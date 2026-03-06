#!/usr/bin/env python3
"""Stage 3: Compute feature stats + high-activation examples (single pass)."""

import json, time, numpy as np, torch
import torch.nn.functional as F
from config import *

# Import SAE class for loading
from stage2_sae import SparseAutoencoder

def main():
    if STATS_FILE.exists():
        print(f"SKIP: Feature stats already computed ({STATS_FILE})")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(SHAPE_FILE.read_text())
    n, d = meta["num_tokens"], meta["hidden_size"]
    token_counts = json.loads(TOKEN_MAP_FILE.read_text())

    print("Loading SAE...")
    ckpt = torch.load(SAE_FILE, map_location=DEVICE, weights_only=True)
    sae = SparseAutoencoder(d, SAE_DICT_SIZE).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    act_mean = ckpt["act_mean"].to(DEVICE)
    act_std = ckpt["act_std"].to(DEVICE)

    acts = np.memmap(str(ACT_CACHE_FILE), dtype=np.float16, mode='r', shape=(n, d))
    dict_size = SAE_DICT_SIZE
    batch_size = 8192
    ek = 8  # examples per feature

    # Streaming accumulators
    sum_act = torch.zeros(dict_size)
    count_pos = torch.zeros(dict_size)
    sum_pos = torch.zeros(dict_size)
    max_act = torch.full((dict_size,), float("-inf"))
    topk_vals = torch.full((dict_size, ek), float("-inf"))
    topk_pos = torch.zeros((dict_size, ek), dtype=torch.long)

    print(f"Computing stats over {n} tokens...")
    t0 = time.time()
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = torch.from_numpy(acts[i:end].astype(np.float32)).to(DEVICE)
        batch = (batch - act_mean) / act_std
        with torch.no_grad():
            f = sae.encode(batch).cpu()

        sum_act += f.sum(0)
        mask = f > 0
        count_pos += mask.sum(0)
        sum_pos += (f * mask.float()).sum(0)
        max_act = torch.max(max_act, f.max(0).values)

        bs = f.shape[0]
        positions = torch.arange(i, end).unsqueeze(0).expand(dict_size, -1)
        combined_vals = torch.cat([topk_vals, f.t()], dim=1)
        combined_pos = torch.cat([topk_pos, positions], dim=1)
        top = combined_vals.topk(ek, dim=1)
        topk_vals = top.values
        topk_pos = combined_pos.gather(1, top.indices)

        if (i // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{n} ({100*i/n:.0f}%) - {i/max(elapsed,1):.0f} tok/s")

    freq = count_pos / n
    conditional_mean = sum_pos / count_pos.clamp(min=1)

    # Only keep features with <10% frequency (sparse/selective)
    score = conditional_mean.clone()
    score[freq >= 0.10] = 0
    score[freq < 0.005] = 0

    nonzero = (score > 0).sum().item()
    k = min(SWEEP_TOP_FEATURES, nonzero)
    top_indices = score.argsort(descending=True)[:k]
    print(f"  {nonzero} features with freq <10%, selecting top {k}")

    # Load texts for high-activation mapping
    from datasets import load_dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    texts = [s["text"] for s in dataset.select(range(min(ACT_NUM_SAMPLES, len(dataset))))]
    cumsum = np.cumsum(token_counts)

    stats = []
    for idx in top_indices:
        fi = idx.item()
        examples = []
        for j in range(ek):
            val = topk_vals[fi, j].item()
            pos = topk_pos[fi, j].item()
            if val <= 0:
                continue
            si = int(np.searchsorted(cumsum, pos, side='right'))
            if si < len(texts):
                lp = pos - (int(cumsum[si - 1]) if si > 0 else 0)
                examples.append({"text": texts[si][:500], "pos": int(lp), "act": round(val, 3)})

        stats.append({
            "feature_idx": fi,
            "frequency": round(freq[fi].item(), 6),
            "mean_act": round((sum_act[fi] / n).item(), 6),
            "max_act": round(max_act[fi].item(), 3),
            "cond_mean": round(conditional_mean[fi].item(), 4),
            "score": round(score[fi].item(), 6),
            "examples": examples,
        })

    STATS_FILE.write_text(json.dumps(stats, indent=2))
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s. Top features: {[s['feature_idx'] for s in stats[:10]]}")
    print(f"Frequency range: {stats[0]['frequency']:.4f} - {stats[-1]['frequency']:.4f}")
    print(f"Saved: {STATS_FILE}")

if __name__ == "__main__":
    main()
