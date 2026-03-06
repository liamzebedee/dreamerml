"""
Fully batched transition collection.
Each iteration: B prompts, B actions, all parallel.
"""

import torch
import numpy as np
import os
import time

from env import DreamEnv, K, clamp_action, N_DREAM
from state import StateExtractor, ContextProjector, CONTEXT_DIM

BATCH_SIZE = 48  # prompts per iteration

STORY_STARTERS = [
    "Once upon a time, there was a",
    "One day, a little girl named",
    "There was a big dog who",
    "The cat and the bird were",
    "A brave knight went to the",
    "In a small village, there lived",
    "The sun was shining and the",
    "A funny monkey found a",
    "The princess looked out the window and",
    "Tom and his friend went to",
    "The little bunny hopped into the",
    "A magic fish swam in the",
    "The old tree had a secret",
    "One morning, the baby bird",
    "Sara loved to play with her",
    "The farmer had a big red",
    "In the garden, there was a",
    "The boy found a shiny",
    "Lily and her mom went to the",
    "A tiny mouse lived in a",
    "Write a scary story about a cat in a forest.",
    "Write a funny story about a dog who found a key.",
    "Write a happy story about a girl and a rabbit.",
    "Write a sad story about a boy who lost his ball.",
    "Write a story about a knight who went to the mountain.",
    "Write a story about a princess and a dragon.",
    "Write a story about a wizard in a cave.",
    "Write a story about a bird who could talk.",
    "Write a story about a mouse who was very brave.",
    "Write a story about a bear who loved honey.",
]


def sample_random_actions(B: int, device="cuda", dtype=torch.float16) -> torch.Tensor:
    """Sample B random actions. (B, K)"""
    # Half uniform, half normal+tanh
    a = torch.empty(B, K, device=device, dtype=dtype)
    half = B // 2
    a[:half] = torch.rand(half, K, device=device, dtype=dtype) * 2 - 1
    a[half:] = torch.tanh(torch.randn(B - half, K, device=device, dtype=dtype) * 0.5)
    return clamp_action(a)


def check_quality_batch(tokens: torch.Tensor, tokenizer) -> torch.Tensor:
    """Check quality for a batch of token sequences. Returns (B,) float."""
    B = tokens.shape[0]
    quality = torch.ones(B, dtype=torch.float32)

    for b in range(B):
        toks = tokens[b].tolist()
        # 4-gram repetition
        if len(toks) >= 8:
            ngrams = set()
            repeated = 0
            for i in range(len(toks) - 3):
                ng = tuple(toks[i:i+4])
                if ng in ngrams:
                    repeated += 1
                ngrams.add(ng)
            if repeated / max(1, len(toks) - 3) > 0.4:
                quality[b] = 0
                continue

        # Punctuation spam
        text = tokenizer.decode(toks, skip_special_tokens=True)
        if len(text) > 0:
            punct = sum(1 for c in text if c in '.,!?\n\r\t')
            if punct / len(text) > 0.5:
                quality[b] = 0

    return quality


def collect_transitions(n_transitions: int = 50000, save_dir: str = "data",
                         batch_size: int = BATCH_SIZE, t_base: int = 24):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda"
    dtype = torch.float16

    env = DreamEnv(device=device, dtype=dtype)
    extractor = StateExtractor(env.model, device=device)
    projector = ContextProjector(env.model.config.hidden_size).to(device).half()

    all_s, all_a, all_ns, all_q, all_c = [], [], [], [], []
    n_collected = 0
    n_quality = 0
    start_time = time.time()

    while n_collected < n_transitions:
        B = min(batch_size, n_transitions - n_collected)

        # Sample B prompts
        prompts = [np.random.choice(STORY_STARTERS) for _ in range(B)]
        prompt_ids = env.encode_batch(prompts)  # (B, seq)

        # Extract context embeddings (batch)
        with torch.no_grad():
            c_raw = extractor.extract_context_batch(prompt_ids)  # (B, hidden)
            c = projector(c_raw)  # (B, 64)

        # Generate base tokens (batch, unperturbed)
        base_tokens = env.generate_batched_manual(prompt_ids, n_tokens=t_base)  # (B, t_base)
        full_ids = torch.cat([prompt_ids, base_tokens], dim=-1)  # (B, seq+t_base)

        # Extract pre-dream states (batch, unperturbed)
        s_t = extractor.extract_batch(full_ids, normalize=True)  # (B, 64)

        # Sample B random actions
        actions = sample_random_actions(B, device, dtype)  # (B, K)

        # Dream generation with per-sample perturbation (batch)
        with env.batched_perturbation(actions):
            dream_tokens = env.generate_batched_manual(full_ids, n_tokens=N_DREAM)  # (B, 40)

        # Build dream contexts
        dream_ids = torch.cat([full_ids, dream_tokens], dim=-1)  # (B, seq+t_base+40)

        # Extract post-dream states (batch, UNPERTURBED)
        s_next = extractor.extract_batch(dream_ids, normalize=True)  # (B, 64)

        # Check quality (batch)
        quality = check_quality_batch(dream_tokens, env.tokenizer)  # (B,)

        # Store
        all_s.append(s_t.float().cpu())
        all_a.append(actions.float().cpu())
        all_ns.append(s_next.float().cpu())
        all_q.append(quality)
        all_c.append(c.float().detach().cpu())

        n_collected += B
        n_quality += quality.sum().item()

        if n_collected % (batch_size * 5) < batch_size:
            elapsed = time.time() - start_time
            rate = n_collected / elapsed
            qr = n_quality / n_collected
            eta = (n_transitions - n_collected) / max(rate, 0.01) / 60
            print(f"[{n_collected}/{n_transitions}] {rate:.1f} trans/s, quality={qr:.1%}, ETA={eta:.1f}min")

    # Save
    data = {
        "states": torch.cat(all_s),
        "actions": torch.cat(all_a),
        "next_states": torch.cat(all_ns),
        "quality": torch.cat(all_q),
        "contexts": torch.cat(all_c),
    }
    path = os.path.join(save_dir, f"transitions_{n_collected}.pt")
    torch.save(data, path)
    print(f"Saved {n_collected} transitions to {path}")

    # Save normalizer + projector
    torch.save(extractor.normalizer.state_dict(), os.path.join(save_dir, "normalizer.pt"))
    torch.save(projector.state_dict(), os.path.join(save_dir, "projector.pt"))

    elapsed = time.time() - start_time
    print(f"\nDone! {n_collected} transitions in {elapsed:.0f}s ({n_collected/elapsed:.1f}/s), "
          f"quality={n_quality/n_collected:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--save_dir", type=str, default="data")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    collect_transitions(n_transitions=args.n, save_dir=args.save_dir, batch_size=args.batch)
