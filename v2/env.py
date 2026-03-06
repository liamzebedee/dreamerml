"""Environment: frozen Qwen2.5-0.5B-Instruct with LoRA basis weight edits and probe signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRABasis(nn.Module):
    """K rank-1 LoRA basis directions applied to attention output (wo) and MLP output (fc2).

    Each basis direction k is parameterized by vectors (a_k, b_k) producing
    rank-1 edit: a_k @ b_k.T. An action vector alpha ∈ R^K produces:
        Δw = Σ_k alpha_k * (a_k @ b_k.T)
    """

    def __init__(self, model, K=16, scale=0.01):
        super().__init__()
        self.K = K
        self.target_modules = []  # list of (name, module, weight_shape)

        # Find wo (attn output proj) and fc2 (MLP output proj) layers
        # Only target a spread of layers (every 4th) for more distinct basis directions
        all_targets = []
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.ndim == 2:
                if any(t in name for t in ["o_proj", "down_proj"]):
                    all_targets.append((name, module.weight.shape))

        # Subsample: take every 4th layer's modules for diversity
        if len(all_targets) > 12:
            self.target_modules = [t for i, t in enumerate(all_targets) if i % 4 == 0]
        else:
            self.target_modules = all_targets

        if not self.target_modules:
            raise ValueError("No target modules found. Check model architecture.")

        # Create rank-1 basis vectors for each target module and each basis direction
        self.basis_A = nn.ParameterDict()
        self.basis_B = nn.ParameterDict()
        for name, shape in self.target_modules:
            safe_name = name.replace(".", "_")
            self.basis_A[safe_name] = nn.Parameter(
                torch.randn(K, shape[0]) * scale
            )
            self.basis_B[safe_name] = nn.Parameter(
                torch.randn(K, shape[1]) * scale
            )

        self._module_name_map = {
            name: name.replace(".", "_") for name, _ in self.target_modules
        }

    def compute_deltas(self, action):
        """Compute weight deltas for each target module given action ∈ R^K.

        Returns dict mapping module name -> Δw tensor.
        """
        deltas = {}
        for name, shape in self.target_modules:
            safe = self._module_name_map[name]
            A = self.basis_A[safe]  # (K, out_dim)
            B = self.basis_B[safe]  # (K, in_dim)
            # Δw = Σ_k action_k * (A_k ⊗ B_k) = A.T @ diag(action) @ B
            delta = torch.einsum("k,ko,ki->oi", action, A, B)
            deltas[name] = delta
        return deltas

    def regularization(self, action):
        """||Δw||^2 summed across all target modules."""
        deltas = self.compute_deltas(action)
        return sum((d ** 2).sum() for d in deltas.values())


class BaseModelEnv:
    """Environment wrapping frozen Qwen2.5-0.5B-Instruct.

    Given an action vector, applies LoRA weight edits, runs forward passes
    on a fixed prompt set, and returns probe signals + reward.
    """

    DEFAULT_PROMPTS = [
        "Write a sad poem about losing a friend.",
        "Tell me a scary campfire story.",
        "Write a love letter from Romeo to Juliet.",
        "Roast me like I'm at a comedy show.",
        "Write an angry rant about people who don't use turn signals.",
        "Describe a peaceful morning in a small village.",
        "Write a dramatic monologue from a villain explaining their plan.",
        "Tell me a story about a dog who waits for its owner to come home.",
    ]

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        prompts=None,
        K=16,
        lora_scale=0.01,
        device=None,
        # Reward coefficients (v1-style)
        alpha=1.0,      # R_coarse weight
        beta=0.5,       # R_fine weight
        gamma=0.3,      # R_struct weight
        delta=0.3,      # unused, kept for compat
        eta=0.0001,     # Weight regularization
        kl_target=2.0,  # unused, kept for compat
        lam=0.5,        # λ for R_coarse
        mu=0.3,         # μ for R_fine
        max_seq_len=128,
        use_compile=False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load frozen model in bf16 for faster compute
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # LoRA basis (trainable, stays fp32 for stable gradients)
        self.lora_basis = LoRABasis(self.model, K=K, scale=lora_scale).to(self.device)

        # Tokenize prompts using chat template for instruct models
        self.prompts = prompts or self.DEFAULT_PROMPTS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        formatted_prompts = []
        for p in self.prompts:
            messages = [{"role": "user", "content": p}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)

        self.prompt_inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(self.device)

        # Figure out which layers to hook for coherence probe
        self._setup_hooks()

        # Cache baseline logits and hidden states
        self._cache_baseline()

        # Reward coefficients
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.kl_target = kl_target
        self.lam = lam
        self.mu = mu

    def _setup_hooks(self):
        """Register forward hooks on early and late transformer layers.

        Only captures 2 hidden states instead of all ~24, saving memory.
        """
        # Find the transformer layers
        if hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise ValueError("Can't find transformer layers for hooks")

        n = len(layers)
        self._early_idx = n // 4
        self._late_idx = n - 1

        self._hook_outputs = {}

        def make_hook(name):
            def hook(module, input, output):
                # output is (hidden_states, ...) for Qwen
                h = output[0] if isinstance(output, tuple) else output
                self._hook_outputs[name] = h.detach()
            return hook

        self._hooks = [
            layers[self._early_idx].register_forward_hook(make_hook("early")),
            layers[self._late_idx].register_forward_hook(make_hook("late")),
        ]

    @torch.inference_mode()
    def _cache_baseline(self):
        """Cache logits and hidden states from the unmodified model."""
        self._hook_outputs.clear()
        outputs = self.model(**self.prompt_inputs)
        self.baseline_logits = outputs.logits.detach()
        self.baseline_early_hidden = self._hook_outputs["early"]
        self.baseline_late_hidden = self._hook_outputs["late"]
        # Baseline entropy per token
        probs = F.softmax(self.baseline_logits, dim=-1)
        self.baseline_entropy = -(probs * (probs + 1e-10).log()).sum(-1).mean()

    def _get_weight(self, name):
        """Get the weight tensor for a named module."""
        module = self.model
        for part in name.split("."):
            module = getattr(module, part)
        return module.weight

    def _apply_deltas(self, deltas):
        """Apply weight deltas to model (in-place)."""
        for name, delta in deltas.items():
            w = self._get_weight(name)
            w.data.add_(delta.to(w.dtype))

    def _remove_deltas(self, deltas):
        """Remove weight deltas from model (in-place)."""
        for name, delta in deltas.items():
            w = self._get_weight(name)
            w.data.sub_(delta.to(w.dtype))

    @torch.inference_mode()
    def compute_probes(self, action):
        """Compute 4 probe signals for a given action. Returns probes tensor (4,).

        Probes: [S_global, S_var, S_entropy, S_coherence]
        Batched forward pass, hook-based hidden state capture.
        """
        deltas = self.lora_basis.compute_deltas(action)
        self._apply_deltas(deltas)

        try:
            self._hook_outputs.clear()
            outputs = self.model(**self.prompt_inputs)

            logits = outputs.logits          # (B, T, V)
            mask = self.prompt_inputs["attention_mask"]  # (B, T)
            token_counts = mask.sum(-1).clamp(min=1)     # (B,)

            # KL divergence per prompt
            log_p_new = F.log_softmax(logits, dim=-1)
            log_p_old = F.log_softmax(self.baseline_logits, dim=-1)
            kl = F.kl_div(log_p_new, log_p_old, log_target=True, reduction='none')
            kl = kl.sum(-1) * mask  # (B, T)
            kl_per_prompt = kl.sum(-1) / token_counts  # (B,)
            S_global = kl_per_prompt.mean()
            S_var = kl_per_prompt.var()

            # Entropy
            probs_new = F.softmax(logits, dim=-1)
            ent = -(probs_new * (probs_new + 1e-10).log()).sum(-1)  # (B, T)
            S_entropy = (ent * mask).sum() / token_counts.sum() - self.baseline_entropy

            # Coherence from hooks (only 2 layers captured, not all)
            early_h = self._hook_outputs["early"]  # (B, T, D)
            late_h = self._hook_outputs["late"]     # (B, T, D)
            cos_new = F.cosine_similarity(
                early_h.flatten(1), late_h.flatten(1), dim=-1
            ).mean()
            cos_old = F.cosine_similarity(
                self.baseline_early_hidden.flatten(1),
                self.baseline_late_hidden.flatten(1),
                dim=-1,
            ).mean()
            S_coherence = cos_new - cos_old

            probes = torch.stack([S_global, S_var, S_entropy, S_coherence])

        finally:
            self._remove_deltas(deltas)

        return probes

    def compute_reward(self, action, probes=None):
        """Compute reward R(a) from probe signals.

        Rewards structured behavioral shifts: moderate KL (not too small, not
        destruction), high prompt variance (different prompts affected differently),
        coherence preservation, and stable entropy.
        """
        if probes is None:
            probes = self.compute_probes(action)

        S_global, S_var, S_entropy, S_coherence = probes.unbind(0)

        # Sweet-spot KL: Gaussian reward centered at kl_target
        # Peaks at kl_target, falls off for too low or too high
        R_kl = torch.exp(-((S_global - self.kl_target) ** 2) / (2 * (self.kl_target * 0.5) ** 2))

        # Coherence preservation
        R_struct = S_coherence

        # Regularization
        R_reg = -self.eta * self.lora_basis.regularization(action)

        reward = (self.alpha * R_kl
                  + self.gamma * R_struct
                  + R_reg)
        return reward, probes

    def step(self, action):
        """Full environment step: compute probes and reward for action ∈ R^K.

        Returns: (reward, probes) where probes is tensor of shape (4,).
        """
        probes = self.compute_probes(action)
        reward, probes = self.compute_reward(action, probes)
        return reward, probes

    @torch.inference_mode()
    def generate(self, action, prompts=None, max_new_tokens=50, temperature=0.7):
        """Generate text continuations with weight edit applied."""
        prompts = prompts or self.prompts

        # Format as chat messages for instruct models
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = self.tokenizer(
            formatted, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(self.device)

        if action is not None:
            deltas = self.lora_basis.compute_deltas(action)
            self._apply_deltas(deltas)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            if action is not None:
                self._remove_deltas(deltas)

        # Decode only the new tokens
        results = []
        for i, output in enumerate(outputs):
            input_len = inputs["input_ids"][i].shape[0]
            new_tokens = output[input_len:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        return results
