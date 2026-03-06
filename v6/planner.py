"""
Planner: target proposal via forward model F, action solve via inverse model I.
M=64 proposals, 1 refinement step.
"""

import torch
import torch.nn.functional as F
from env import K, clamp_action
from state import STATE_DIM, CONTEXT_DIM
from forward_model import ForwardModel
from inverse_model import InverseModel


M = 64  # number of candidate proposals


def compute_objective(s: torch.Tensor, thresholds: dict = None) -> torch.Tensor:
    """
    Compute planning objective J(s) from state vector.
    s: (batch, 64) or (64,)

    State dims (from state.py):
    0: entropy, 1: top1, 2: top5, 3: top20, 4: gap,
    5: mean_logit, 6: std_logit, 7: top200, 8: ppl_proxy, 9: rep_risk
    10-17: hidden stats layer 1
    18-25: hidden stats layer 2
    26-33: hidden stats layer 3
    34-39: attn stats layer 1 (mean_entropy, std_entropy, mean_max, std_max, mean_dist, std_dist)
    40-45: attn stats layer 2
    46-51: attn stats layer 3
    52-63: trajectory deltas
    """
    if s.dim() == 1:
        s = s.unsqueeze(0)

    if thresholds is None:
        thresholds = {
            "entropy_low": -1.0,    # 5th percentile in normalized space (approx)
            "rep_high": 1.5,
            "attn_collapse": 0.1,
        }

    # Novelty proxy: + entropy + (1 - top1_prob) + (1 - repetition_risk)
    # In normalized space, higher values = higher novelty
    j_nov = s[:, 0] + (-s[:, 1]) + (-s[:, 9])

    # Coherence proxy: penalties
    penalty_low_ent = torch.clamp(thresholds["entropy_low"] - s[:, 0], min=0) * 2
    penalty_high_rep = torch.clamp(s[:, 9] - thresholds["rep_high"], min=0) * 2

    # Attention collapse: if attn entropy std near 0 across all probe layers
    attn_std_mean = (s[:, 35].abs() + s[:, 41].abs() + s[:, 47].abs()) / 3
    penalty_attn = torch.clamp(thresholds["attn_collapse"] - attn_std_mean, min=0) * 2

    j_coh = -(penalty_low_ent + penalty_high_rep + penalty_attn)

    j = 1.0 * j_nov + 1.5 * j_coh
    return j.squeeze()


class Planner:
    """Plan actions using forward + inverse dynamics models."""

    def __init__(self, forward_model: ForwardModel, inverse_model: InverseModel, device="cuda"):
        self.F = forward_model
        self.I = inverse_model
        self.device = device
        self.F.eval()
        self.I.eval()

    @torch.no_grad()
    def plan(self, s_t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Choose action using forward proposal + inverse solve.

        s_t: (64,) current state
        c: (64,) context embedding
        Returns: a* (32,) clamped action
        """
        s_t = s_t.unsqueeze(0).expand(M, -1)  # (M, 64)
        c = c.unsqueeze(0).expand(M, -1)       # (M, 64)

        # 1. Sample candidate actions
        a_cand = torch.randn(M, K, device=self.device) * 0.35
        a_cand = torch.tanh(a_cand)
        # L2 clamp each
        norms = a_cand.norm(dim=-1, keepdim=True)
        a_cand = torch.where(norms > 2.0, a_cand * 2.0 / norms, a_cand)

        # 2. Predict next states
        s_pred = self.F(s_t, a_cand, c)  # (M, 64)

        # 3. Score
        scores = compute_objective(s_pred)  # (M,)

        # 4. Choose best
        best_idx = scores.argmax()
        s_star = s_pred[best_idx:best_idx+1]  # (1, 64)

        # 5. Solve for action via inverse dynamics
        a_star = self.I(s_t[:1], s_star, c[:1]).squeeze(0)  # (32,)

        # 6. Optional refinement
        s_check = self.F(s_t[:1], a_star.unsqueeze(0), c[:1])
        j_check = compute_objective(s_check)
        j_star = scores[best_idx]

        if j_check < j_star:
            # One correction step
            s_corrected = s_check + (s_star - s_check)
            a_corr = self.I(s_t[:1], s_corrected, c[:1]).squeeze(0)
            a_star = 0.5 * a_star + 0.5 * a_corr

        return clamp_action(a_star)


if __name__ == "__main__":
    # Quick test with random models
    device = "cuda"
    fwd = ForwardModel().to(device)
    inv = InverseModel().to(device)
    planner = Planner(fwd, inv, device)

    s = torch.randn(64, device=device)
    c = torch.randn(64, device=device)
    a = planner.plan(s, c)
    print(f"Planned action: shape={a.shape}, norm={a.norm():.3f}")
    print(f"Action: {a}")
