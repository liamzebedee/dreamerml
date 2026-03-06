"""
Forward dynamics model F: predicts s_{t+1} = s_t + F(s_t, a_t, c).
MLP with residual: [s(64), a(32), c(64)] -> 160 -> 512 -> 512 -> 64.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
from state import STATE_DIM, CONTEXT_DIM
from env import K


class ForwardModel(nn.Module):
    """Predict delta-state from (state, action, context)."""

    def __init__(self, state_dim=STATE_DIM, action_dim=K, context_dim=CONTEXT_DIM):
        super().__init__()
        input_dim = state_dim + action_dim + context_dim  # 160

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim),
        )

    def forward(self, s, a, c):
        """
        s: (batch, 64), a: (batch, 32), c: (batch, 64)
        Returns predicted s_{t+1} = s + delta
        """
        x = torch.cat([s, a, c], dim=-1)
        delta = self.net(x)
        return s + delta  # residual prediction


def train_forward_model(
    data_path: str = "data",
    epochs: int = 3,
    batch_size: int = 4096,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    save_path: str = "checkpoints/forward_model.pt",
):
    """Train forward model on collected transitions."""
    device = "cuda"

    # Load data
    data = load_all_transitions(data_path, device)
    states, actions, next_states, quality, contexts = data

    print(f"Training forward model on {len(states)} transitions")
    print(f"Quality rate: {quality.mean():.2%}")

    dataset = TensorDataset(states, actions, next_states, quality, contexts)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ForwardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for s, a, s_next, q, c in loader:
            pred = model(s, a, c)
            # Quality-gated MSE
            loss = (q.unsqueeze(-1) * (s_next - pred) ** 2).mean()
            loss += weight_decay * sum(p.pow(2).sum() for p in model.parameters())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved forward model to {save_path}")
    return model


def load_all_transitions(data_path: str, device: str = "cuda"):
    """Load and concatenate all transition files."""
    files = sorted([f for f in os.listdir(data_path) if f.startswith("transitions_") and f.endswith(".pt")])
    if not files:
        raise FileNotFoundError(f"No transition files in {data_path}")

    all_s, all_a, all_ns, all_q, all_c = [], [], [], [], []
    for f in files:
        d = torch.load(os.path.join(data_path, f), map_location=device)
        all_s.append(d["states"])
        all_a.append(d["actions"])
        all_ns.append(d["next_states"])
        all_q.append(d["quality"])
        all_c.append(d["contexts"])

    return (
        torch.cat(all_s),
        torch.cat(all_a),
        torch.cat(all_ns),
        torch.cat(all_q),
        torch.cat(all_c),
    )


if __name__ == "__main__":
    train_forward_model()
