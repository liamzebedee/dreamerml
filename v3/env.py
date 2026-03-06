"""Base model wrapper and LoRA perturbation basis for frozen TinyStories."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LoRABasis(nn.Module):
    """K rank-1 LoRA directions applied to attention out_proj and MLP c_proj.

    ΔW(a) = α Σ_k a_k A_k B_k^T
    where a_k ∈ [-1,1], A_k ∈ R^out, B_k ∈ R^in.
    """

    def __init__(self, model, K=16, alpha=2.0, init_std=0.02):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.targets = []

        n_layers = len(model.transformer.h)
        start = int(n_layers * 0.3)
        end = int(n_layers * 0.7)

        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()

        for i in range(start, end + 1):
            for attr in ["attn.attention.out_proj", "mlp.c_proj"]:
                module = model.transformer.h[i]
                for part in attr.split("."):
                    module = getattr(module, part)
                full_name = f"transformer.h.{i}.{attr}"
                out_d, in_d = module.weight.shape
                safe = full_name.replace(".", "_")
                self.A[safe] = nn.Parameter(torch.randn(K, out_d) * init_std)
                self.B[safe] = nn.Parameter(torch.randn(K, in_d) * init_std)
                self.targets.append((full_name, safe))

    def compute_deltas(self, action):
        """action: (K,) in [-1,1]. Returns {module_name: ΔW tensor}."""
        deltas = {}
        for full_name, safe in self.targets:
            delta = self.alpha * torch.einsum(
                "k,ko,ki->oi", action, self.A[safe], self.B[safe]
            )
            deltas[full_name] = delta
        return deltas


class DreamEnv:
    """Frozen TinyStories model with LoRA perturbation basis."""

    PROMPTS = [
        "Once upon a time there was a little",
        "The brave knight walked into the dark",
        "A tiny bird sat on the",
        "The old wizard opened his book and",
        "In a small village by the sea",
        "The little girl found a magic",
        "One sunny morning the cat decided to",
        "Deep in the forest there lived a",
    ]

    def __init__(self, model_name="roneneldan/TinyStories-1M", K=16, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.lora = LoRABasis(self.model, K=K).to(self.device)
        self.prompt_inputs = self.tokenizer(
            self.PROMPTS, return_tensors="pt", padding=True,
            truncation=True, max_length=32,
        ).to(self.device)

    def _get_weight(self, name):
        module = self.model
        for part in name.split("."):
            module = getattr(module, part)
        return module.weight

    def apply_perturbation(self, deltas):
        for name, delta in deltas.items():
            w = self._get_weight(name)
            w.data.add_(delta.to(w.dtype))

    def remove_perturbation(self, deltas):
        for name, delta in deltas.items():
            w = self._get_weight(name)
            w.data.sub_(delta.to(w.dtype))

    @torch.inference_mode()
    def generate(self, action=None, prompts=None, max_new_tokens=48,
                 temperature=0.9, top_p=0.95):
        prompts = prompts or self.PROMPTS
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=32,
        ).to(self.device)

        deltas = None
        if action is not None:
            deltas = self.lora.compute_deltas(action)
            self.apply_perturbation(deltas)
        try:
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            if deltas is not None:
                self.remove_perturbation(deltas)

        results = []
        for i, o in enumerate(out):
            n = inputs["input_ids"][i].shape[0]
            results.append(self.tokenizer.decode(o[n:], skip_special_tokens=True))
        return results
