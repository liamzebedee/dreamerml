"""Environment: frozen Qwen2.5-0.5B with LoRA basis weight edits and probe signals."""

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
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight.ndim == 2:
                # Qwen2.5 uses o_proj for attn output, down_proj for MLP output
                if any(t in name for t in ["o_proj", "down_proj"]):
                    self.target_modules.append((name, module.weight.shape))

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
    """Environment wrapping frozen Qwen2.5-0.5B.

    Given an action vector, applies LoRA weight edits, runs forward passes
    on a fixed prompt set, and returns probe signals + reward.
    """

    DEFAULT_PROMPTS = [
        "The theory of relativity states that",
        "In a surprising turn of events, the government",
        "The most important thing about machine learning is",
        "When writing Python code, you should always",
        "The capital of France is Paris, which is known for",
        "To solve this equation, we first need to",
        "The history of ancient Rome begins with",
        "According to recent scientific research,",
    ]

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",
        prompts=None,
        K=16,
        lora_scale=0.01,
        device=None,
        # Reward coefficients
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        eta=0.01,
        lam=0.5,    # λ for R_coarse
        mu=0.3,     # μ for R_fine
        max_seq_len=64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load frozen model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # LoRA basis (trainable)
        self.lora_basis = LoRABasis(self.model, K=K, scale=lora_scale).to(self.device)

        # Tokenize prompts
        self.prompts = prompts or self.DEFAULT_PROMPTS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.prompt_inputs = self.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(self.device)

        # Cache baseline logits and hidden states
        self._cache_baseline()

        # Reward coefficients
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.lam = lam
        self.mu = mu

    @torch.no_grad()
    def _cache_baseline(self):
        """Cache logits and hidden states from the unmodified model."""
        outputs = self.model(
            **self.prompt_inputs,
            output_hidden_states=True,
        )
        self.baseline_logits = outputs.logits.detach()
        hidden = outputs.hidden_states
        self.baseline_early_hidden = hidden[len(hidden) // 4].detach()
        self.baseline_late_hidden = hidden[-1].detach()
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
            self._get_weight(name).data.add_(delta)

    def _remove_deltas(self, deltas):
        """Remove weight deltas from model (in-place)."""
        for name, delta in deltas.items():
            self._get_weight(name).data.sub_(delta)

    def compute_probes(self, action):
        """Compute 4 probe signals for a given action. Returns (probes, info_dict).

        Probes: [S_global, S_var, S_entropy, S_coherence]
        Processes prompts one at a time to avoid OOM with large vocabs.
        """
        deltas = self.lora_basis.compute_deltas(action)
        self._apply_deltas(deltas)

        try:
            B = self.prompt_inputs["input_ids"].shape[0]
            kl_per_prompt = []
            entropy_sum = 0.0
            entropy_count = 0
            cos_new_list = []

            for i in range(B):
                inputs_i = {k: v[i:i+1] for k, v in self.prompt_inputs.items()}
                mask_i = inputs_i["attention_mask"].squeeze(0)  # (T,)
                token_count = mask_i.sum().clamp(min=1)

                with torch.no_grad():
                    outputs = self.model(**inputs_i, output_hidden_states=True)

                logits_i = outputs.logits.squeeze(0)         # (T, V)
                baseline_i = self.baseline_logits[i]          # (T, V)

                # KL divergence per prompt
                log_p_new = F.log_softmax(logits_i, dim=-1)
                log_p_old = F.log_softmax(baseline_i, dim=-1)
                kl = F.kl_div(log_p_new, log_p_old, log_target=True, reduction='none')
                kl = kl.sum(-1) * mask_i  # (T,)
                kl_per_prompt.append(kl.sum() / token_count)

                # Entropy
                probs_new = F.softmax(logits_i, dim=-1)
                ent = -(probs_new * (probs_new + 1e-10).log()).sum(-1)  # (T,)
                entropy_sum += (ent * mask_i).sum()
                entropy_count += token_count

                # Coherence: early/late hidden states
                hidden = outputs.hidden_states
                early_h = hidden[len(hidden) // 4].flatten(1)
                late_h = hidden[-1].flatten(1)
                cos_new_list.append(
                    F.cosine_similarity(early_h, late_h, dim=-1).squeeze()
                )

                del outputs, logits_i, log_p_new, log_p_old, probs_new

            kl_per_prompt = torch.stack(kl_per_prompt)
            S_global = kl_per_prompt.mean()
            S_var = kl_per_prompt.var()

            S_entropy = (entropy_sum / entropy_count) - self.baseline_entropy

            cos_new = torch.stack(cos_new_list).mean()
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

        R(a) = α·R_coarse + β·R_fine + γ·R_struct + R_reg
        """
        if probes is None:
            probes = self.compute_probes(action)

        S_global, S_var, S_entropy, S_coherence = probes.unbind(0)

        R_coarse = S_global - self.lam * S_var
        R_fine = S_var - self.mu * S_global
        R_struct = S_coherence
        R_reg = -self.eta * self.lora_basis.regularization(action)

        reward = self.alpha * R_coarse + self.beta * R_fine + self.gamma * R_struct + R_reg
        return reward, probes

    def step(self, action):
        """Full environment step: compute probes and reward for action ∈ R^K.

        Returns: (reward, probes) where probes is tensor of shape (4,).
        """
        probes = self.compute_probes(action)
        reward, probes = self.compute_reward(action, probes)
        return reward, probes

    @torch.no_grad()
    def generate(self, action, prompts=None, max_new_tokens=50, temperature=0.7):
        """Generate text continuations with weight edit applied.

        Args:
            action: R^K action vector (or None for baseline)
            prompts: list of strings (defaults to self.prompts)
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature

        Returns:
            list of generated strings
        """
        prompts = prompts or self.prompts
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=64,
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
