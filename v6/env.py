"""
TinyStories wrapper, LoRA basis creation, action application/revert, generation.
Optimized: fp16, batched generation, per-sample perturbation via hooks.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, List, Tuple
from contextlib import contextmanager


MODEL_NAME = "roneneldan/TinyStories-33M"
K = 32              # number of LoRA directions
ALPHA = 0.06        # global scale
L2_CLAMP = 2.0      # max L2 norm of action vector
N_DREAM = 40        # dream tokens
TEMPERATURE = 0.9
TOP_P = 0.95


@dataclass
class LoRABasis:
    """Fixed rank-1 LoRA basis for weight perturbation."""
    directions: dict  # {layer_name: (A, B)}
    # Precomputed ΔW_k = A_k @ B_k for each layer, shape (K, out, in)
    delta_weights: dict  # {layer_name: Tensor(K, out, in)}
    target_layers: List[str]


def get_target_layers(model) -> List[str]:
    """Get target layer names: out_proj and c_proj (MLP down) in middle band."""
    num_layers = model.config.num_layers
    start = int(0.3 * num_layers)
    end = int(0.7 * num_layers)
    targets = []
    for l in range(start, end + 1):
        targets.append(f"transformer.h.{l}.attn.attention.out_proj")
        targets.append(f"transformer.h.{l}.mlp.c_proj")
    return targets


def get_module_by_name(model, name: str) -> nn.Module:
    parts = name.split(".")
    mod = model
    for p in parts:
        mod = getattr(mod, p)
    return mod


def create_lora_basis(model, device="cuda", dtype=torch.float16) -> LoRABasis:
    """Create K rank-1 LoRA directions for each target layer."""
    target_layers = get_target_layers(model)
    directions = {}
    delta_weights = {}

    for layer_name in target_layers:
        mod = get_module_by_name(model, layer_name)
        out_dim, in_dim = mod.weight.shape

        A = torch.randn(K, out_dim, 1, device=device, dtype=dtype) * 0.02
        B = torch.randn(K, 1, in_dim, device=device, dtype=dtype) * 0.02
        directions[layer_name] = (A, B)
        # Precompute ΔW_k = A_k @ B_k -> (K, out, in)
        delta_weights[layer_name] = (A @ B).squeeze()  # (K, out, in) if K>1

    return LoRABasis(directions=directions, delta_weights=delta_weights, target_layers=target_layers)


def clamp_action(a: torch.Tensor) -> torch.Tensor:
    """Clamp action: per-component to [-1,1], then L2 norm to 2.0. Works batched."""
    a = a.clamp(-1.0, 1.0)
    if a.dim() == 1:
        norm = a.norm(2)
        if norm > L2_CLAMP:
            a = a * (L2_CLAMP / norm)
    else:
        norms = a.norm(2, dim=-1, keepdim=True)
        a = torch.where(norms > L2_CLAMP, a * L2_CLAMP / norms, a)
    return a


class DreamEnv:
    """Environment wrapping TinyStories model with LoRA perturbation."""

    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        print(f"Loading {MODEL_NAME} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=dtype
        ).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        for p in self.model.parameters():
            p.requires_grad = False

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params/1e6:.1f}M params, {self.model.config.num_layers} layers, "
              f"hidden={self.model.config.hidden_size}")

        self.basis = create_lora_basis(self.model, device, dtype)
        print(f"LoRA basis: K={K}, targets={self.basis.target_layers}")

        self._original_weights = {}
        self._hooks = []

    # ---- Single-action weight modification (for inference/eval) ----

    def apply_action(self, action: torch.Tensor):
        """Apply weight perturbation W' = W + alpha * sum_k a_k * ΔW_k."""
        action = clamp_action(action.to(self.device, self.dtype))
        for layer_name in self.basis.target_layers:
            mod = get_module_by_name(self.model, layer_name)
            self._original_weights[layer_name] = mod.weight.data.clone()
            dw = self.basis.delta_weights[layer_name]  # (K, out, in)
            delta = torch.einsum("k,koi->oi", action, dw)
            mod.weight.data += ALPHA * delta

    def revert_action(self):
        for layer_name, orig in self._original_weights.items():
            get_module_by_name(self.model, layer_name).weight.data.copy_(orig)
        self._original_weights.clear()

    # ---- Per-sample batched perturbation via hooks ----

    @contextmanager
    def batched_perturbation(self, actions: torch.Tensor):
        """
        Context manager: install hooks for per-sample perturbation.
        actions: (batch, K) - different action per batch element.
        During forward pass, each sample gets its own weight perturbation.
        """
        actions = clamp_action(actions.to(self.device, self.dtype))
        hooks = []

        for layer_name in self.basis.target_layers:
            mod = get_module_by_name(self.model, layer_name)
            dw = self.basis.delta_weights[layer_name]  # (K, out, in)

            def make_hook(dw_ref):
                def hook_fn(module, input, output):
                    # input[0]: (batch, seq, in_dim) or (batch, 1, in_dim)
                    x = input[0]
                    B = x.shape[0]
                    # Per-sample delta: actions (B,K) @ dw (K, out, in) -> (B, out, in)
                    # Then: delta_y = einsum("boi, bsi -> bso", per_dw, x) * ALPHA
                    per_dw = torch.einsum("bk,koi->boi", actions[:B], dw_ref)
                    delta_y = torch.einsum("boi,bsi->bso", per_dw, x) * ALPHA
                    return output + delta_y
                return hook_fn

            h = mod.register_forward_hook(make_hook(dw))
            hooks.append(h)

        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    # ---- Generation ----

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, n_tokens: int = N_DREAM,
                 temperature: float = TEMPERATURE, top_p: float = TOP_P) -> torch.Tensor:
        """Generate n_tokens using HF generate. Returns new tokens only."""
        attn_mask = (input_ids != self.pad_id).long()
        output = self.model.generate(
            input_ids, attention_mask=attn_mask,
            max_new_tokens=n_tokens, min_new_tokens=n_tokens,
            do_sample=True, temperature=temperature, top_p=top_p,
            pad_token_id=self.pad_id,
        )
        return output[:, input_ids.shape[-1]:]

    @torch.no_grad()
    def generate_batched_manual(self, input_ids: torch.Tensor, n_tokens: int = N_DREAM,
                                 temperature: float = TEMPERATURE, top_p: float = TOP_P) -> torch.Tensor:
        """
        Manual batched autoregressive generation (no KV cache).
        Fast for small models with short sequences.
        Returns new tokens only: (batch, n_tokens).
        """
        cur = input_ids
        new_tokens = []
        for _ in range(n_tokens):
            attn_mask = (cur != self.pad_id).long()
            logits = self.model(cur, attention_mask=attn_mask).logits[:, -1, :] / temperature
            # Top-p
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cumprobs = sorted_logits.softmax(-1).cumsum(-1)
            mask = (cumprobs - sorted_logits.softmax(-1)) >= top_p
            sorted_logits[mask] = float("-inf")
            probs = sorted_logits.softmax(-1)
            tok_idx = torch.multinomial(probs, 1)
            tok = sorted_idx.gather(-1, tok_idx)
            new_tokens.append(tok)
            cur = torch.cat([cur, tok], dim=-1)
        return torch.cat(new_tokens, dim=-1)

    # ---- Helpers ----

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode list of texts, left-pad to same length."""
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             padding_side="left", truncation=True, max_length=128)
        return enc["input_ids"].to(self.device)

    def decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids[0] if ids.dim() > 1 else ids, skip_special_tokens=True)

    def decode_batch(self, ids: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    import time
    env = DreamEnv()

    prompt = "Once upon a time, there was a little cat who"
    ids = env.encode(prompt)

    # Single generation
    gen = env.generate(ids, n_tokens=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {env.decode(gen)}")

    # Batched generation with per-sample perturbation
    B = 32
    prompts = [prompt] * B
    batch_ids = env.encode_batch(prompts)
    actions = torch.randn(B, K, device="cuda") * 0.3
    actions = torch.tanh(actions)

    t0 = time.time()
    with env.batched_perturbation(actions):
        dreams = env.generate_batched_manual(batch_ids, n_tokens=40)
    dt = time.time() - t0
    print(f"\nBatched {B} dreams in {dt:.3f}s ({B/dt:.0f} dreams/s)")
    for i in range(3):
        print(f"  Dream {i}: {env.decode(dreams[i])}")
