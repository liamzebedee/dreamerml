"""
Inverse dynamics model I: infers action a from (s_t, s_{t+1}, c).
MLP: [s_t(64), s_{t+1}(64), c(64)] -> 192 -> 512 -> 512 -> 32, tanh output.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from state import STATE_DIM, CONTEXT_DIM
from env import K


class InverseModel(nn.Module):
    """Infer action from (state, next_state, context)."""

    def __init__(self, state_dim=STATE_DIM, action_dim=K, context_dim=CONTEXT_DIM):
        super().__init__()
        input_dim = state_dim * 2 + context_dim  # 192

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh(),
        )

    def forward(self, s, s_next, c):
        """
        s: (batch, 64), s_next: (batch, 64), c: (batch, 64)
        Returns action in [-1, 1]^32
        """
        x = torch.cat([s, s_next, c], dim=-1)
        return self.net(x)


def train_inverse_model(
    data_path: str = "data",
    epochs: int = 3,
    batch_size: int = 4096,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    save_path: str = "checkpoints/inverse_model.pt",
):
    """Train inverse model on quality=1 transitions only."""
    device = "cuda"

    from forward_model import load_all_transitions
    data = load_all_transitions(data_path, device)
    states, actions, next_states, quality, contexts = data

    # Filter to quality=1 only
    mask = quality > 0.5
    states = states[mask]
    actions = actions[mask]
    next_states = next_states[mask]
    contexts = contexts[mask]

    print(f"Training inverse model on {len(states)} quality transitions")

    dataset = TensorDataset(states, actions, next_states, contexts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = InverseModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for s, a, s_next, c in loader:
            pred_a = model(s, s_next, c)
            loss = ((a - pred_a) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved inverse model to {save_path}")
    return model


if __name__ == "__main__":
    train_inverse_model()
