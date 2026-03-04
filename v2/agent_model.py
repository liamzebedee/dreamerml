"""World Model: learned landscape predictor M_ψ(a) → predicted probe vector, with self-attention."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Attention-based model mapping action a ∈ R^K → predicted probe vector p̂ ∈ R^4.

    Treats each action dimension as a token, applies self-attention to capture
    interactions between basis directions, then predicts probe signals.

    Trained with MSE on real rollout data. Used for "dreaming" — the actor
    can train on predicted probes without running real model forward passes.
    """

    def __init__(self, K=16, d_model=64, n_heads=4, n_layers=2, lr=1e-3):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # Project each action scalar to d_model (one token per basis direction)
        self.input_proj = nn.Linear(1, d_model)
        # Learned positional embeddings for each basis direction
        self.pos_embed = nn.Parameter(torch.randn(K, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Readout: pool over tokens → 4 probe outputs
        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, action):
        """Predict probe vector from action.

        action: (B, K) or (K,) → (B, 4) or (4,)
        """
        squeeze = action.ndim == 1
        if squeeze:
            action = action.unsqueeze(0)

        B = action.shape[0]
        # (B, K) → (B, K, 1) → (B, K, d_model)
        x = self.input_proj(action.unsqueeze(-1))
        x = x + self.pos_embed.unsqueeze(0)

        # Self-attention over basis directions
        x = self.encoder(x)  # (B, K, d_model)

        # Mean pool over tokens
        x = x.mean(dim=1)  # (B, d_model)
        out = self.readout(x)  # (B, 4)

        if squeeze:
            out = out.squeeze(0)
        return out

    def predict_probes(self, action):
        """Predict probes without gradient tracking."""
        with torch.no_grad():
            return self.forward(action)

    def train_step(self, actions, real_probes):
        """One gradient step on MSE loss.

        Args:
            actions: (B, K) batch of actions
            real_probes: (B, 4) corresponding real probe values

        Returns:
            float: MSE loss value
        """
        predicted = self.forward(actions)
        loss = F.mse_loss(predicted, real_probes)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


class ReplayBuffer:
    """Simple replay buffer storing (action, probes, reward) tuples."""

    def __init__(self, capacity=10000, K=16):
        self.capacity = capacity
        self.K = K
        self.actions = []
        self.probes = []
        self.rewards = []

    def add(self, action, probes, reward):
        """Add a single transition. Tensors are detached and moved to CPU."""
        self.actions.append(action.detach().cpu())
        self.probes.append(probes.detach().cpu())
        self.rewards.append(reward.detach().cpu() if torch.is_tensor(reward) else torch.tensor(reward))

        if len(self.actions) > self.capacity:
            self.actions.pop(0)
            self.probes.pop(0)
            self.rewards.pop(0)

    def add_batch(self, actions, probes, rewards):
        """Add a batch of transitions."""
        for i in range(actions.shape[0]):
            self.add(actions[i], probes[i], rewards[i])

    def sample(self, batch_size, device="cpu"):
        """Sample a random batch. Returns (actions, probes, rewards) tensors."""
        n = len(self.actions)
        if n == 0:
            return None
        idx = torch.randint(0, n, (min(batch_size, n),))
        acts = torch.stack([self.actions[i] for i in idx]).to(device)
        prbs = torch.stack([self.probes[i] for i in idx]).to(device)
        rews = torch.stack([self.rewards[i] for i in idx]).to(device)
        return acts, prbs, rews

    def __len__(self):
        return len(self.actions)
