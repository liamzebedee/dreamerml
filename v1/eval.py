"""Artifact generation: sweep discovered weight directions and show behavioral changes."""

import argparse
import torch
from env import BaseModelEnv

EVAL_PROMPTS = [
    "Once upon a time, there was a",
    "The little dog ran to the",
    "She looked at the big red",
]

STRENGTHS = [-3.0, -1.0, 0.0, 1.0, 3.0]

DIVIDER = "─" * 80


def print_header(text):
    print(f"\n{'═' * 80}")
    print(f"  {text}")
    print(f"{'═' * 80}")


def probe_summary(env, action):
    """Return formatted probe string for an action."""
    probes = env.compute_probes(action)
    names = ["KL_global", "KL_var", "Entropy_Δ", "Coherence_Δ"]
    return " | ".join(f"{n}={v:+.4f}" for n, v in zip(names, probes.tolist()))


def sweep_direction(env, direction_idx, K, strengths=STRENGTHS, prompts=EVAL_PROMPTS):
    """Sweep a single basis direction at multiple strengths."""
    print_header(f"Direction {direction_idx}")

    for strength in strengths:
        action = torch.zeros(K, device=env.device)
        action[direction_idx] = strength

        if strength == 0.0:
            label = "BASELINE (strength=0)"
            probes_str = "n/a"
        else:
            label = f"strength={strength:+.1f}"
            probes_str = probe_summary(env, action)

        print(f"\n  [{label}]  {probes_str}")
        print(f"  {DIVIDER}")

        generations = env.generate(
            action if strength != 0.0 else None,
            prompts=prompts,
            max_new_tokens=40,
            temperature=0.7,
        )

        for prompt, gen in zip(prompts, generations):
            # Truncate for readability
            gen_short = gen[:120].replace("\n", " ")
            print(f"    \"{prompt}\" → {gen_short}")


def sweep_hierarchy(env, coarse_idx, fine_idx, K, prompts=EVAL_PROMPTS):
    """Demonstrate hierarchical control: fix coarse, sweep fine."""
    print_header(f"Hierarchy: Coarse=dir[{coarse_idx}], Fine=dir[{fine_idx}]")

    for coarse_val in [-2.0, 2.0]:
        print(f"\n  ┌── Coarse dir[{coarse_idx}] = {coarse_val:+.1f}")

        for fine_val in [-2.0, 0.0, 2.0]:
            action = torch.zeros(K, device=env.device)
            action[coarse_idx] = coarse_val
            action[fine_idx] = fine_val

            probes_str = probe_summary(env, action)
            print(f"  │")
            print(f"  ├─ Fine dir[{fine_idx}] = {fine_val:+.1f}  |  {probes_str}")

            generations = env.generate(action, prompts=prompts[:2], max_new_tokens=40)
            for prompt, gen in zip(prompts[:2], generations):
                gen_short = gen[:100].replace("\n", " ")
                print(f"  │    \"{prompt}\" → {gen_short}")

        print(f"  └──")


def probe_landscape(env, K, n_random=50):
    """Sample random actions at various scales and show probe statistics."""
    print_header("Probe Landscape (random actions)")

    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        probes_list = []
        for _ in range(n_random):
            action = torch.randn(K, device=env.device) * scale
            probes = env.compute_probes(action)
            probes_list.append(probes)

        probes_t = torch.stack(probes_list)  # (n, 4)
        means = probes_t.mean(0)
        stds = probes_t.std(0)
        names = ["KL_global", "KL_var", "Entropy_Δ", "Coherence_Δ"]

        print(f"\n  Scale={scale:.1f}  (n={n_random})")
        for name, m, s in zip(names, means, stds):
            bar_len = min(int(abs(m.item()) * 200), 40)
            bar = "█" * bar_len if m > 0 else "░" * bar_len
            print(f"    {name:14s}: {m:+.6f} ± {s:.6f}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="DreamerML Eval: Sweep weight directions")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pt (uses random basis if not given)")
    parser.add_argument("--lora-scale", type=float, default=0.1,
                        help="LoRA basis init scale (only used without checkpoint)")
    parser.add_argument("--directions", type=int, nargs="+", default=None,
                        help="Which directions to sweep (default: first 4)")
    parser.add_argument("--landscape", action="store_true",
                        help="Run probe landscape analysis")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = BaseModelEnv(K=args.K, device=device, lora_scale=args.lora_scale)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        env.lora_basis.load_state_dict(ckpt["lora_basis"])
        print("Loaded trained basis directions.")
    else:
        print(f"Using random basis (scale={args.lora_scale})")

    K = args.K

    # Probe landscape first to find interesting scales
    if args.landscape:
        probe_landscape(env, K)

    # Sweep individual directions
    dirs_to_sweep = args.directions or list(range(min(4, K)))
    for d in dirs_to_sweep:
        sweep_direction(env, d, K)

    # Hierarchy demo: first coarse direction + first fine direction
    K_coarse = K // 2
    sweep_hierarchy(env, coarse_idx=0, fine_idx=K_coarse, K=K)

    print_header("Done")


if __name__ == "__main__":
    main()
