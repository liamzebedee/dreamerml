"""State feature extraction: compact summary of the model's cognitive state."""

import torch
import torch.nn.functional as F

STATE_DIM = 16


@torch.inference_mode()
def extract_state(logits, hidden_states, attention_mask):
    """Extract 16-dim state vector from model outputs.

    8 features at last token + 8 sequence-averaged:
      logit entropy, top-1 prob, top-5 mass, top-20 mass,
      hidden mean, hidden var, hidden norm, logit std.
    """
    B, T, V = logits.shape
    mask = attention_mask.float()
    seq_lens = mask.sum(-1).clamp(min=1)
    batch_idx = torch.arange(B, device=logits.device)
    last_idx = (seq_lens - 1).long()

    hidden = hidden_states[-1]  # last layer: (B, T, D)

    def features_at(lg, hd):
        """8 features from logits (B, V) and hidden (B, D)."""
        p = F.softmax(lg, dim=-1)
        sp, _ = p.sort(dim=-1, descending=True)
        ent = -(p * (p + 1e-10).log()).sum(-1)
        return torch.stack([
            ent.mean(),
            sp[:, 0].mean(),
            sp[:, :5].sum(-1).mean(),
            sp[:, :min(20, V)].sum(-1).mean(),
            hd.mean(-1).mean(),
            hd.var(-1).mean(),
            hd.norm(dim=-1).mean(),
            lg.std(-1).mean(),
        ])

    # Last-token features
    f_last = features_at(
        logits[batch_idx, last_idx],
        hidden[batch_idx, last_idx],
    )

    # Sequence-averaged features
    logits_avg = (logits * mask.unsqueeze(-1)).sum(1) / seq_lens.unsqueeze(-1)
    hidden_avg = (hidden * mask.unsqueeze(-1)).sum(1) / seq_lens.unsqueeze(-1)
    f_seq = features_at(logits_avg, hidden_avg)

    return torch.cat([f_last, f_seq])  # (16,)
