"""Actor: Gaussian policy with GRPO (Group Relative Policy Optimization) updates."""

import torch
import torch.nn as nn
import copy


class HierarchicalGating(nn.Module):
    """Splits action into [a_coarse, a_fine] and applies gating:
        a_fine_effective = sigmoid(W @ a_coarse) * a_fine
    """

    def __init__(self, K):
        super().__init__()
        self.K = K
        self.K_coarse = K // 2
        self.K_fine = K - self.K_coarse
        self.gate = nn.Linear(self.K_coarse, self.K_fine)

    def forward(self, action):
        """Apply hierarchical gating. Returns gated action of same shape."""
        a_coarse = action[..., :self.K_coarse]
        a_fine = action[..., self.K_coarse:]
        gate_values = torch.sigmoid(self.gate(a_coarse))
        a_fine_gated = gate_values * a_fine
        return torch.cat([a_coarse, a_fine_gated], dim=-1)


def _build_policy_net(K, hidden_dim):
    """Build the shared policy network architecture."""
    return nn.Sequential(
        nn.Linear(K, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )


class Actor(nn.Module):
    """Gaussian policy π_φ(a|z) with GRPO updates.

    Outputs mean μ and log-std for a K-dimensional action.
    Uses Group Relative Policy Optimization: sample G actions per step,
    normalize rewards within group for advantages, clipped surrogate objective.
    """

    def __init__(
        self,
        K=16,
        hidden_dim=128,
        G=8,           # group size for GRPO
        clip_eps=0.2,
        kl_beta=0.01,  # KL penalty coefficient
        lr=3e-4,
        use_gating=True,
    ):
        super().__init__()
        self.K = K
        self.G = G
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.hidden_dim = hidden_dim

        # Policy network (no conditioning input for minimal version)
        self.net = _build_policy_net(K, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, K)
        self.log_std = nn.Parameter(torch.zeros(K))

        # Hierarchical gating
        self.gating = HierarchicalGating(K) if use_gating else None

        # Reference policy — separate network, never optimized
        self.ref_net = _build_policy_net(K, hidden_dim)
        self.ref_mean_head = nn.Linear(hidden_dim, K)
        self.ref_log_std = nn.Parameter(torch.zeros(K))
        # Exclude ref params from optimization
        for p in [*self.ref_net.parameters(), *self.ref_mean_head.parameters(), self.ref_log_std]:
            p.requires_grad_(False)

        # Optimizer (only covers main policy params + gating)
        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=lr
        )

        self.save_reference()

    def _get_distribution(self):
        """Get current Gaussian distribution."""
        h = self.net(torch.zeros(self.K, device=self.log_std.device))
        mu = self.mean_head(h)
        std = self.log_std.exp()
        return torch.distributions.Normal(mu, std)

    def _get_ref_distribution(self):
        """Get reference Gaussian distribution (no grad)."""
        with torch.no_grad():
            h = self.ref_net(torch.zeros(self.K, device=self.ref_log_std.device))
            mu = self.ref_mean_head(h)
            std = self.ref_log_std.exp()
        return torch.distributions.Normal(mu, std)

    def log_prob(self, actions):
        """Compute log prob of actions under current policy."""
        dist = self._get_distribution()
        return dist.log_prob(actions).sum(-1)

    def sample_raw(self, n=None):
        """Sample raw (pre-gating) actions. Returns (raw_actions, gated_actions, log_probs)."""
        dist = self._get_distribution()
        shape = (n or self.G,)
        raw = dist.rsample(shape)
        log_probs = dist.log_prob(raw).sum(-1)
        gated = self.gating(raw) if self.gating is not None else raw
        return raw, gated, log_probs

    def save_reference(self):
        """Snapshot current params into reference network."""
        self.ref_net.load_state_dict(self.net.state_dict())
        self.ref_mean_head.load_state_dict(self.mean_head.state_dict())
        self.ref_log_std.data.copy_(self.log_std.data)

    def grpo_update(self, raw_actions, old_log_probs, rewards):
        """Perform one GRPO update step.

        Args:
            raw_actions: (G, K) pre-gating actions (detached)
            old_log_probs: (G,) log probs under sampling policy (detached)
            rewards: (G,) rewards for each action (detached)

        Returns:
            dict with loss info
        """
        raw_actions = raw_actions.detach()
        old_log_probs = old_log_probs.detach()
        rewards = rewards.detach()

        # Normalize rewards within group → advantages
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Current log probs (has grad)
        new_log_probs = self.log_prob(raw_actions)

        # Importance ratio
        ratio = (new_log_probs - old_log_probs).exp()

        # Clipped surrogate
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty against reference (ref has no grad)
        ref_dist = self._get_ref_distribution()
        ref_lp = ref_dist.log_prob(raw_actions).sum(-1)
        kl = (new_log_probs - ref_lp).mean()
        kl_loss = self.kl_beta * kl

        loss = policy_loss + kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "kl": kl.item(),
            "total_loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
        }
