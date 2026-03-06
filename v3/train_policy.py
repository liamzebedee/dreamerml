"""Phase 1: Train perturbation policy via GRPO.

Reward = KL(p_dream || p_base) − λ · collapse_penalty

Goal: discover weight perturbations that produce coherent but distinct
reasoning trajectories. After training, freeze the policy.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from env import DreamEnv
from policy import PerturbationPolicy
from state import extract_state, STATE_DIM


def compute_reward(base_logits, perturbed_logits, attention_mask,
                   collapse_lambda=2.0, kl_target=0.5):
    """Sweet-spot novelty reward minus collapse penalty.

    R_kl = exp(-(KL - target)^2 / (2 * sigma^2))  — peaks at kl_target.
    collapse = fraction of positions with entropy < 0.5.
    """
    mask = attention_mask.float()
    total = mask.sum().clamp(min=1)

    log_p = F.log_softmax(perturbed_logits, dim=-1)
    log_q = F.log_softmax(base_logits, dim=-1)

    kl = (log_p.exp() * (log_p - log_q)).sum(-1)  # (B, T)
    kl_mean = (kl * mask).sum() / total

    # Gaussian sweet spot: want KL near target, not too small or too large
    sigma = kl_target * 0.5
    R_kl = torch.exp(-((kl_mean - kl_target) ** 2) / (2 * sigma ** 2))

    # Variance across prompts: different prompts should be affected differently
    kl_per_prompt = (kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    R_var = kl_per_prompt.std()

    # Collapse detector
    probs = F.softmax(perturbed_logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(-1)
    low_ent = ((entropy < 0.5).float() * mask).sum() / total

    R = R_kl + 0.3 * R_var - collapse_lambda * low_ent
    return R.item(), kl_mean.item(), low_ent.item()


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = DreamEnv(model_name=args.model, K=args.K, device=device)
    policy = PerturbationPolicy(state_dim=STATE_DIM, K=args.K).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "policy_train.jsonl")
    log_f = open(log_path, "w")

    # Cache full-batch baseline
    with torch.inference_mode():
        base_out = env.model(
            **env.prompt_inputs, output_hidden_states=True,
        )
        base_logits = base_out.logits.detach()

    print(f"K={args.K}, G={args.G}, steps={args.steps}")
    print(f"LoRA targets: {len(env.lora.targets)} modules")
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"LoRA basis params: {sum(p.numel() for p in env.lora.parameters()):,}")
    print()

    best_reward = float("-inf")
    t0 = time.time()

    for step in range(args.steps):
        # Random prompt subset per step for diversity
        n_p = min(4, len(env.PROMPTS))
        idx = torch.randperm(len(env.PROMPTS))[:n_p]

        input_ids = env.prompt_inputs["input_ids"][idx]
        attn_mask = env.prompt_inputs["attention_mask"][idx]

        with torch.inference_mode():
            out = env.model(
                input_ids=input_ids, attention_mask=attn_mask,
                output_hidden_states=True,
            )
            state = extract_state(out.logits, out.hidden_states, attn_mask)
            batch_base = base_logits[idx]

        # Clone state out of inference mode for autograd
        state = state.clone().detach().requires_grad_(False)

        # Sample G actions from policy
        actions, log_probs, raw_actions = policy.sample(state, args.G)

        # Evaluate each action
        rewards, novs, cols = [], [], []
        for g in range(args.G):
            action = actions[g].detach()
            with torch.inference_mode():
                deltas = env.lora.compute_deltas(action)
                env.apply_perturbation(deltas)
                p_out = env.model(input_ids=input_ids, attention_mask=attn_mask)
                env.remove_perturbation(deltas)

            r, nov, col = compute_reward(
                batch_base, p_out.logits, attn_mask,
                collapse_lambda=args.collapse_lambda,
                kl_target=args.kl_target,
            )
            rewards.append(r)
            novs.append(nov)
            cols.append(col)

        rewards_t = torch.tensor(rewards, device=device)

        # GRPO clipped surrogate update
        adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        new_lp = policy.log_prob(state, raw_actions)
        ratio = (new_lp - log_probs.detach()).exp()
        s1 = ratio * adv
        s2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
        loss = -torch.min(s1, s2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        mean_r = rewards_t.mean().item()
        if mean_r > best_reward:
            best_reward = mean_r

        entry = {
            "step": step,
            "reward": mean_r,
            "reward_std": rewards_t.std().item(),
            "novelty": sum(novs) / len(novs),
            "collapse": sum(cols) / len(cols),
            "loss": loss.item(),
        }
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()

        if step % args.print_every == 0:
            elapsed = time.time() - t0
            print(
                f"Step {step:4d} | R={mean_r:+.4f} (±{rewards_t.std():.3f}) | "
                f"nov={entry['novelty']:.4f} col={entry['collapse']:.4f} | "
                f"loss={loss.item():.4f} | {elapsed:.0f}s"
            )

        # Periodic text generation for qualitative check
        if step > 0 and step % args.gen_every == 0:
            with torch.no_grad():
                action = policy.deterministic(state)
            print(f"\n--- Step {step} generations ---")
            base_texts = env.generate(
                None, prompts=env.PROMPTS[:3], max_new_tokens=40,
            )
            dream_texts = env.generate(
                action, prompts=env.PROMPTS[:3], max_new_tokens=40,
            )
            for i in range(3):
                print(f"  [{env.PROMPTS[i]}]")
                print(f"    Base:  {base_texts[i][:100]}")
                print(f"    Dream: {dream_texts[i][:100]}")
            print()

    # ── Final evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    with torch.inference_mode():
        out = env.model(**env.prompt_inputs, output_hidden_states=True)
        state = extract_state(
            out.logits, out.hidden_states,
            env.prompt_inputs["attention_mask"],
        )

    with torch.no_grad():
        action = policy.deterministic(state)

    print(f"\nLearned action: [{', '.join(f'{v:.3f}' for v in action.cpu())}]")
    print(f"Policy log_std: [{', '.join(f'{v:.2f}' for v in policy.log_std.data.cpu())}]")

    # Multiple trials to show consistency
    for trial in range(3):
        print(f"\n─── Trial {trial + 1} ───")
        base_texts = env.generate(None, max_new_tokens=60)
        dream_texts = env.generate(action, max_new_tokens=60)
        for i in range(len(env.PROMPTS)):
            print(f"\n  [{i}] {env.PROMPTS[i]}")
            print(f"      Base:  {base_texts[i][:120]}")
            print(f"      Dream: {dream_texts[i][:120]}")

    # ── Direction sweep ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DIRECTION SWEEP: individual LoRA basis directions at strength ±0.8")
    print("=" * 70)

    sweep_prompts = env.PROMPTS[:2]
    for k in range(min(args.K, 8)):
        for sign in [+0.8, -0.8]:
            a = torch.zeros(args.K, device=device)
            a[k] = sign
            texts = env.generate(a, prompts=sweep_prompts, max_new_tokens=40)
            label = f"d{k:02d}={'+'if sign>0 else '-'}"
            for i, t in enumerate(texts):
                print(f"  {label} | {sweep_prompts[i]}{t[:80]}")
        print()

    # Save checkpoint
    ckpt = {
        "policy": policy.state_dict(),
        "lora": env.lora.state_dict(),
        "args": vars(args),
    }
    ckpt_path = os.path.join(args.out_dir, "policy.pt")
    torch.save(ckpt, ckpt_path)
    log_f.close()

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Log: {log_path}")
    print(f"Best reward: {best_reward:+.4f}")
    print(f"Total time: {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser(description="DreamerML v3 - Phase 1 policy training")
    p.add_argument("--model", default="roneneldan/TinyStories-1M")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--collapse-lambda", type=float, default=2.0)
    p.add_argument("--kl-target", type=float, default=0.5)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--gen-every", type=int, default=50)
    p.add_argument("--out-dir", default="runs/v3")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
