"""Perturbation policy: maps cognitive state to LoRA mixture coefficients.

Architecture: 16 → 128 → 128 → 16, ~30k parameters.
Output squashed through tanh to [-1, 1]^K.
"""

import torch
import torch.nn as nn


class PerturbationPolicy(nn.Module):

    def __init__(self, state_dim=16, hidden_dim=128, K=16):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, K)
        self.log_std = nn.Parameter(torch.zeros(K))

    def get_dist(self, state):
        h = self.net(state)
        mean = self.mean_head(h)
        std = self.log_std.exp().clamp(min=0.01)
        return torch.distributions.Normal(mean, std)

    def sample(self, state, G):
        """Sample G actions. Returns (actions, log_probs, raw_samples)."""
        dist = self.get_dist(state)
        raw = dist.rsample((G,))
        log_probs = dist.log_prob(raw).sum(-1)
        actions = torch.tanh(raw)
        return actions, log_probs, raw

    def log_prob(self, state, raw_actions):
        dist = self.get_dist(state)
        return dist.log_prob(raw_actions).sum(-1)

    def deterministic(self, state):
        h = self.net(state)
        return torch.tanh(self.mean_head(h))
