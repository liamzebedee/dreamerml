#!/usr/bin/env python3
"""Stage 2: Train sparse autoencoder on cached activations."""

import json, time, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, dict_size):
        super().__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        return F.relu(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

def main():
    if SAE_FILE.exists():
        print(f"SKIP: SAE already trained ({SAE_FILE})")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(SHAPE_FILE.read_text())
    n, d = meta["num_tokens"], meta["hidden_size"]
    print(f"Training SAE: {d}d -> {SAE_DICT_SIZE} features, {n} samples, {SAE_EPOCHS} epochs")

    acts = np.memmap(str(ACT_CACHE_FILE), dtype=np.float16, mode='r', shape=(n, d))

    print("  Computing normalization...")
    sum_x = np.zeros(d, dtype=np.float64)
    sum_x2 = np.zeros(d, dtype=np.float64)
    for i in range(0, n, 100000):
        block = acts[i:i+100000].astype(np.float32)
        sum_x += block.sum(0)
        sum_x2 += (block ** 2).sum(0)
    act_mean = (sum_x / n).astype(np.float32)
    act_std = np.sqrt(sum_x2 / n - act_mean ** 2).clip(min=1e-6).astype(np.float32)
    act_mean_t = torch.from_numpy(act_mean).to(DEVICE)
    act_std_t = torch.from_numpy(act_std).to(DEVICE)

    sae = SparseAutoencoder(d, SAE_DICT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(sae.parameters(), lr=SAE_LR)

    indices = np.arange(n)
    t0 = time.time()
    for epoch in range(SAE_EPOCHS):
        np.random.shuffle(indices)
        total_recon = total_l1 = 0
        total_alive = 0
        nb = 0

        for i in range(0, n - SAE_BATCH + 1, SAE_BATCH):
            idx = indices[i:i+SAE_BATCH]
            idx.sort()
            batch = torch.from_numpy(acts[idx].astype(np.float32)).to(DEVICE, non_blocking=True)
            batch = (batch - act_mean_t) / act_std_t

            x_hat, f = sae(batch)
            recon = F.mse_loss(x_hat, batch)
            l1 = f.abs().mean()
            loss = recon + SAE_SPARSITY_COEFF * l1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                sae.decoder.weight.copy_(F.normalize(sae.decoder.weight, dim=0))

            total_recon += recon.item()
            total_l1 += l1.item()
            total_alive += (f > 0).any(dim=0).sum().item()
            nb += 1

        print(f"  Epoch {epoch+1}/{SAE_EPOCHS}: recon={total_recon/nb:.6f} "
              f"l1={total_l1/nb:.4f} alive={total_alive/nb:.0f}/{SAE_DICT_SIZE}")

    torch.save({
        "state_dict": sae.state_dict(),
        "act_mean": act_mean_t.cpu(),
        "act_std": act_std_t.cpu(),
        "config": {"input_dim": d, "dict_size": SAE_DICT_SIZE, "layer": TARGET_LAYER},
    }, SAE_FILE)
    print(f"SAE saved: {SAE_FILE} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
