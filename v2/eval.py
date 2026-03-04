"""Artifact generation: sweep discovered weight directions and show behavioral changes.

Produces the artifact described in spec2.md: a readable map of discovered
internal axes, showing how each learned knob changes the model's behavior.
"""

import argparse
import torch
from env import BaseModelEnv
from agent import Actor

# Diverse prompts that reveal different behavioral dimensions
EVAL_PROMPTS = [
    "The theory of relativity states that",
    "She walked into the room and immediately noticed",
    "To solve this problem, we need to",
    "The most controversial aspect of this debate is",
    "In the year 2050, humanity will",
]

# Wide sweep to find where behavioral shifts become visible
STRENGTHS = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]

DIVIDER = "─" * 90
DOUBLE = "═" * 90


def print_header(text):
    print(f"\n{DOUBLE}")
    print(f"  {text}")
    print(f"{DOUBLE}")


def probe_summary(probes):
    """Format probe values."""
    names = ["KL", "KL_var", "Ent_Δ", "Coh_Δ"]
    return " | ".join(f"{n}={v:+.4f}" for n, v in zip(names, probes.tolist()))


def sweep_direction(env, direction_idx, K, strengths=STRENGTHS, prompts=EVAL_PROMPTS):
    """Sweep a single basis direction at multiple strengths."""
    print_header(f"AXIS {direction_idx}  (coarse)" if direction_idx < K // 2 else f"AXIS {direction_idx}  (fine)")

    for strength in strengths:
        action = torch.zeros(K, device=env.device)
        action[direction_idx] = strength

        if strength == 0.0:
            label = "BASELINE"
            probes_str = ""
        else:
            probes = env.compute_probes(action)
            label = f"{strength:+.0f}"
            probes_str = f"  [{probe_summary(probes)}]"

        print(f"\n  ▸ strength={label}{probes_str}")
        print(f"  {DIVIDER}")

        generations = env.generate(
            action if strength != 0.0 else None,
            prompts=prompts,
            max_new_tokens=50,
            temperature=0.8,
        )

        for prompt, gen in zip(prompts, generations):
            gen_short = gen[:140].replace("\n", " ").strip()
            print(f"    {prompt}")
            print(f"      → {gen_short}")


def sweep_hierarchy(env, coarse_idx, fine_idx, K, prompts=EVAL_PROMPTS):
    """Demonstrate hierarchical gating: fix coarse, sweep fine."""
    print_header(f"HIERARCHY: Coarse=axis[{coarse_idx}] × Fine=axis[{fine_idx}]")

    for coarse_val in [-5.0, 0.0, 5.0]:
        print(f"\n  ┌── Coarse axis[{coarse_idx}] = {coarse_val:+.0f}")

        for fine_val in [-5.0, 0.0, 5.0]:
            action = torch.zeros(K, device=env.device)
            action[coarse_idx] = coarse_val
            action[fine_idx] = fine_val

            if coarse_val == 0.0 and fine_val == 0.0:
                probes_str = "BASELINE"
            else:
                probes = env.compute_probes(action)
                probes_str = probe_summary(probes)

            print(f"  │")
            print(f"  ├─ Fine axis[{fine_idx}] = {fine_val:+.0f}  |  {probes_str}")

            generations = env.generate(action if (coarse_val != 0 or fine_val != 0) else None,
                                       prompts=prompts[:3], max_new_tokens=40)
            for prompt, gen in zip(prompts[:3], generations):
                gen_short = gen[:100].replace("\n", " ").strip()
                print(f"  │    {prompt}")
                print(f"  │      → {gen_short}")

        print(f"  └──")


def actor_best_directions(env, actor, K):
    """Sample from the trained actor and show what directions it prefers."""
    print_header("ACTOR'S LEARNED POLICY: What directions does the agent prefer?")

    with torch.no_grad():
        _, gated, _ = actor.sample_raw(n=20)

    # Show the mean action and top-activated directions
    mean_action = gated.mean(0)
    std_action = gated.std(0)

    print("\n  Mean action (policy center):")
    for i in range(K):
        bar_len = min(int(abs(mean_action[i].item()) * 10), 30)
        sign = "+" if mean_action[i] > 0 else "-"
        bar = "█" * bar_len
        kind = "coarse" if i < K // 2 else "fine  "
        print(f"    axis[{i:2d}] ({kind}): {mean_action[i]:+.3f} ± {std_action[i]:.3f}  {sign}{bar}")

    # Generate with the actor's mean action
    print(f"\n  Generations from actor's mean action:")
    print(f"  {DIVIDER}")
    probes = env.compute_probes(mean_action)
    print(f"  [{probe_summary(probes)}]")
    gens = env.generate(mean_action, prompts=EVAL_PROMPTS, max_new_tokens=50, temperature=0.8)
    for prompt, gen in zip(EVAL_PROMPTS, gens):
        gen_short = gen[:140].replace("\n", " ").strip()
        print(f"    {prompt}")
        print(f"      → {gen_short}")

    # Also show a few individual samples
    print(f"\n  3 individual actor samples:")
    for s in range(3):
        action = gated[s]
        print(f"\n  Sample {s+1}: action norm={action.norm():.3f}")
        gens = env.generate(action, prompts=EVAL_PROMPTS[:2], max_new_tokens=40, temperature=0.8)
        for prompt, gen in zip(EVAL_PROMPTS[:2], gens):
            gen_short = gen[:120].replace("\n", " ").strip()
            print(f"    {prompt}")
            print(f"      → {gen_short}")


def main():
    parser = argparse.ArgumentParser(description="DreamerML v2 Eval: Map of Internal Axes")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lora-scale", type=float, default=0.02)
    parser.add_argument("--directions", type=int, nargs="+", default=None)
    parser.add_argument("--no-hierarchy", action="store_true")
    parser.add_argument("--no-actor", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    env = BaseModelEnv(K=args.K, device=device, lora_scale=args.lora_scale)

    actor = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        env.lora_basis.load_state_dict(ckpt["lora_basis"])
        if "actor" in ckpt and not args.no_actor:
            actor = Actor(K=args.K, use_gating=True).to(device)
            actor.load_state_dict(ckpt["actor"])
            actor.eval()
        print("Loaded.")

    K = args.K

    # Sweep individual directions (2 coarse + 2 fine by default)
    K_coarse = K // 2
    dirs_to_sweep = args.directions or [0, 1, K_coarse, K_coarse + 1]
    for d in dirs_to_sweep:
        sweep_direction(env, d, K)

    # Hierarchy demo
    if not args.no_hierarchy:
        sweep_hierarchy(env, coarse_idx=0, fine_idx=K_coarse, K=K)

    # Actor's learned policy
    if actor is not None:
        actor_best_directions(env, actor, K)

    print_header("END")


if __name__ == "__main__":
    main()
