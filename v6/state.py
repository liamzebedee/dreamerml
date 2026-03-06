"""
Batched state extractor: obs(context) -> s ∈ R^64.
Supports batch extraction for fast data collection.
"""

import torch
import torch.nn.functional as F
from typing import List


STATE_DIM = 64
CONTEXT_DIM = 64


class RunningNormalizer:
    """EMA-based running mean/std for state normalization."""

    def __init__(self, dim: int, momentum: float = 0.001, device="cuda"):
        self.dim = dim
        self.momentum = momentum
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 0
        self.frozen = False

    def update_batch(self, x: torch.Tensor):
        """Update with a batch of states (B, dim)."""
        if self.frozen:
            return
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False) + 1e-8
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
            self.var = (1 - self.momentum) * self.var + self.momentum * batch_var
        self.count += 1

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def freeze(self):
        self.frozen = True

    def state_dict(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(),
                "count": self.count, "frozen": self.frozen}

    def load_state_dict(self, d, device="cuda"):
        self.mean = d["mean"].to(device)
        self.var = d["var"].to(device)
        self.count = d["count"]
        self.frozen = d["frozen"]


def get_probe_layers(num_layers: int) -> List[int]:
    return [int(0.25 * num_layers), int(0.50 * num_layers), int(0.75 * num_layers)]


class StateExtractor:
    """Extract 64-dim cognitive state. Fully batched."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.num_layers = model.config.num_layers
        self.probe_layers = get_probe_layers(self.num_layers)
        self.normalizer = RunningNormalizer(STATE_DIM, device=device)
        self.pad_id = model.config.eos_token_id  # pad token

    @torch.no_grad()
    def extract_batch(self, input_ids: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract states for a batch. input_ids: (B, seq). Returns (B, 64).
        Trajectory deltas are zeroed for batch extraction (used in collection).
        """
        B = input_ids.shape[0]
        attn_mask = (input_ids != self.pad_id).long()

        outputs = self.model(
            input_ids, attention_mask=attn_mask,
            output_hidden_states=True, output_attentions=True,
        )

        logits = outputs.logits  # (B, seq, vocab)
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        # Find last real token position per sample
        seq_lens = attn_mask.sum(-1)  # (B,)
        last_idx = (seq_lens - 1).clamp(min=0)  # (B,)

        # Get logits at last position
        last_logits = logits[torch.arange(B, device=self.device), last_idx]  # (B, vocab)
        last_tokens = input_ids[torch.arange(B, device=self.device), last_idx]  # (B,)

        features = []

        # A) Logit stats (10 dims)
        features.append(self._logit_stats_batch(last_logits, last_tokens))

        # B) Hidden state stats at 3 layers (3*8=24 dims)
        for li in self.probe_layers:
            h = hidden_states[li + 1]  # (B, seq, hidden)
            features.append(self._hidden_stats_batch(h, last_idx, attn_mask))

        # C) Attention stats at 3 layers (3*6=18 dims)
        for li in self.probe_layers:
            attn = attentions[li]  # (B, heads, seq, seq)
            features.append(self._attention_stats_batch(attn, last_idx))

        # D) Trajectory deltas (12 dims) - zero for batch collection
        features.append(torch.zeros(B, 12, device=self.device, dtype=last_logits.dtype))

        s = torch.cat(features, dim=-1)  # (B, 64)

        if normalize:
            self.normalizer.update_batch(s.float())
            s = self.normalizer.normalize(s.float())

        s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
        return s  # (B, 64)

    def _logit_stats_batch(self, logits: torch.Tensor, last_tokens: torch.Tensor) -> torch.Tensor:
        """(B, vocab) -> (B, 10)"""
        logits_f = logits.float()
        probs = F.softmax(logits_f, dim=-1)
        log_probs = F.log_softmax(logits_f, dim=-1)

        entropy = -(probs * log_probs).sum(-1)  # (B,)
        sorted_probs, _ = probs.sort(descending=True)

        top1 = sorted_probs[:, 0]
        top5 = sorted_probs[:, :5].sum(-1)
        top20 = sorted_probs[:, :20].sum(-1)
        top200 = sorted_probs[:, :200].sum(-1)

        sorted_logits_f, _ = logits_f.sort(descending=True)
        gap = sorted_logits_f[:, 0] - sorted_logits_f[:, 1]
        mean_logit = logits_f.mean(-1)
        std_logit = logits_f.std(-1)
        ppl = torch.exp(entropy)
        rep_risk = probs[torch.arange(probs.shape[0], device=probs.device), last_tokens]

        return torch.stack([entropy, top1, top5, top20, gap,
                           mean_logit, std_logit, top200, ppl, rep_risk], dim=-1)

    def _hidden_stats_batch(self, h: torch.Tensor, last_idx: torch.Tensor,
                             attn_mask: torch.Tensor) -> torch.Tensor:
        """(B, seq, hidden) -> (B, 8)"""
        B, S, H = h.shape
        h_f = h.float()

        # Mask padded positions
        mask_3d = attn_mask.unsqueeze(-1).float()  # (B, seq, 1)
        h_masked = h_f * mask_3d
        counts = attn_mask.sum(-1, keepdim=True).unsqueeze(-1).float().clamp(min=1)  # (B,1,1)

        mean_all = (h_masked.sum(1) / counts.squeeze(-1)).mean(-1)  # (B,)
        var_all = ((h_masked ** 2).sum(1) / counts.squeeze(-1) - (h_masked.sum(1) / counts.squeeze(-1)) ** 2).mean(-1).clamp(min=0)

        h_abs = h_f.abs() * mask_3d
        mean_abs = (h_abs.sum(1) / counts.squeeze(-1)).mean(-1)
        var_abs = ((h_abs ** 2).sum(1) / counts.squeeze(-1) - (h_abs.sum(1) / counts.squeeze(-1)) ** 2).mean(-1).clamp(min=0)

        # Last token hidden
        h_last = h_f[torch.arange(B, device=h.device), last_idx]  # (B, hidden)
        mean_last = h_last.mean(-1)
        var_last = h_last.var(-1)

        # Cosine sim: last token vs mean of recent 16
        recent_means = []
        for b in range(B):
            start = max(0, last_idx[b].item() - 15)
            end = last_idx[b].item() + 1
            recent_means.append(h_f[b, start:end].mean(0))
        recent_mean = torch.stack(recent_means)  # (B, hidden)
        cos_sim = F.cosine_similarity(h_last, recent_mean, dim=-1)

        norm_last = h_last.norm(dim=-1)

        return torch.stack([mean_all, var_all, mean_abs, var_abs,
                           mean_last, var_last, cos_sim, norm_last], dim=-1)

    def _attention_stats_batch(self, attn: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        """(B, heads, seq, seq) -> (B, 6)"""
        B, H, S, _ = attn.shape
        attn_f = attn.float()

        # Get last token's attention for each sample
        last_attn = attn_f[torch.arange(B, device=attn.device), :, last_idx]  # (B, heads, seq)

        eps = 1e-10
        attn_entropy = -(last_attn * (last_attn + eps).log()).sum(-1)  # (B, heads)
        mean_entropy = attn_entropy.mean(-1)
        std_entropy = attn_entropy.std(-1)

        max_attn = last_attn.max(-1).values
        mean_max = max_attn.mean(-1)
        std_max = max_attn.std(-1)

        positions = torch.arange(S, device=attn.device, dtype=attn_f.dtype)
        expected_pos = (last_attn * positions).sum(-1)  # (B, heads)
        mean_dist = expected_pos.mean(-1)
        std_dist = expected_pos.std(-1)

        return torch.stack([mean_entropy, std_entropy, mean_max, std_max,
                           mean_dist, std_dist], dim=-1)

    @torch.no_grad()
    def extract_context_batch(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract context embeddings. (B, seq) -> (B, hidden_dim)."""
        attn_mask = (input_ids != self.pad_id).long()
        outputs = self.model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
        B = input_ids.shape[0]
        last_idx = (attn_mask.sum(-1) - 1).clamp(min=0)
        return outputs.hidden_states[-1][torch.arange(B, device=self.device), last_idx]


class ContextProjector(torch.nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int = CONTEXT_DIM):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


if __name__ == "__main__":
    from env import DreamEnv
    import time

    env = DreamEnv()
    extractor = StateExtractor(env.model, device="cuda")

    prompts = ["Once upon a time, there was a little cat."] * 64
    batch_ids = env.encode_batch(prompts)

    t0 = time.time()
    states = extractor.extract_batch(batch_ids)
    dt = time.time() - t0
    print(f"Batch state extraction: {batch_ids.shape} -> {states.shape} in {dt*1000:.1f}ms")

    t0 = time.time()
    ctx = extractor.extract_context_batch(batch_ids)
    dt = time.time() - t0
    print(f"Batch context extraction: {ctx.shape} in {dt*1000:.1f}ms")
