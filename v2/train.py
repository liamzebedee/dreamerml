"""Training loop: unified real+dream with ramping dream ratio."""

import argparse
import json
import os
import time

import torch

from env import BaseModelEnv
from agent import Actor
from agent_model import WorldModelEnsemble, ReplayBuffer


def compute_reward_from_probes(probes, alpha=1.0, gamma=0.3, kl_target=0.3, **kwargs):
    """Compute reward from probe tensor without needing the env (for dreaming).

    probes: (..., 4) tensor [S_global, S_var, S_entropy, S_coherence]
    """
    S_global = probes[..., 0]
    S_coherence = probes[..., 3]

    R_kl = torch.exp(-((S_global - kl_target) ** 2) / (2 * (kl_target * 0.5) ** 2))
    R_struct = S_coherence

    return alpha * R_kl + gamma * R_struct


def real_rollout(env, actor, replay_buffer, device):
    """Sample G actions, run through real env, collect data."""
    raw_actions, gated_actions, old_log_probs = actor.sample_raw()
    gated_detached = gated_actions.detach()

    rewards = []
    all_probes = []
    for i in range(actor.G):
        reward, probes = env.step(gated_detached[i])
        rewards.append(reward)
        all_probes.append(probes)
        replay_buffer.add(gated_detached[i], probes, reward)

    rewards = torch.stack(rewards).to(device)
    all_probes = torch.stack(all_probes).to(device)

    if device == "cuda":
        torch.cuda.empty_cache()

    return raw_actions, old_log_probs, rewards, all_probes, gated_actions


def dream_rollout(actor, world_model_ensemble, reward_fn, pessimism, device):
    """Sample G actions, predict probes with ensemble, compute pessimistic rewards."""
    raw_actions, gated_actions, old_log_probs = actor.sample_raw()

    with torch.no_grad():
        rewards, predicted_probes = world_model_ensemble.predict_pessimistic(
            gated_actions, reward_fn, pessimism=pessimism
        )

    return raw_actions, old_log_probs, rewards, predicted_probes


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # --- Setup ---
    print(f"Loading environment (Qwen2.5-0.5B-Instruct)...")
    env = BaseModelEnv(
        K=args.K,
        device=device,
        lora_scale=args.lora_scale,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        eta=args.eta,
        kl_target=args.kl_target,
        use_compile=args.compile,
    )

    actor = Actor(
        K=args.K,
        hidden_dim=args.actor_hidden,
        G=args.G,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        lr=args.actor_lr,
        use_gating=args.use_gating,
    ).to(device)
    actor.save_reference()

    world_model = WorldModelEnsemble(
        n_models=args.n_ensemble,
        K=args.K,
        d_model=args.wm_d_model,
        n_heads=args.wm_n_heads,
        n_layers=args.wm_n_layers,
        lr=args.wm_lr,
    ).to(device)

    replay_buffer = ReplayBuffer(capacity=args.buffer_size, K=args.K)

    reward_kwargs = dict(
        alpha=args.alpha, gamma=args.gamma,
        kl_target=args.kl_target,
    )
    reward_fn = lambda probes: compute_reward_from_probes(probes, **reward_kwargs)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    def log(entry):
        entry["timestamp"] = time.time()
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    print(f"K={args.K}, G={args.G}, steps={args.total_steps}, "
          f"warmup={args.warmup_steps}, ensemble={args.n_ensemble}")

    # --- Unified training loop ---
    best_real_reward = float("-inf")
    real_reward_ema = None

    for step in range(args.total_steps):
        # Dream ratio ramps from 0 to max_dream_ratio after warmup
        if step < args.warmup_steps:
            dream_ratio = 0.0
        else:
            progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
            dream_ratio = min(args.max_dream_ratio, progress * args.max_dream_ratio)

        is_dream = (step >= args.warmup_steps and torch.rand(1).item() < dream_ratio)

        if is_dream:
            # Dream rollout with pessimistic ensemble
            raw_actions, old_log_probs, rewards, probes = dream_rollout(
                actor, world_model, reward_fn, args.pessimism, device
            )
            actor_info = actor.grpo_update(raw_actions, old_log_probs, rewards)

            entry = {
                "step": step, "type": "dream",
                "dream_reward": actor_info["mean_reward"],
                "policy_loss": actor_info["policy_loss"],
                "kl": actor_info["kl"],
                "dream_ratio": dream_ratio,
            }
            log(entry)

            if step % args.print_every == 0:
                print(f"  Step {step:4d} [DREAM] | R={actor_info['mean_reward']:+.4f} | "
                      f"KL={actor_info['kl']:.4f} | dream%={dream_ratio:.0%}")

        else:
            # Real rollout
            raw_actions, old_log_probs, rewards, probes, gated_actions = real_rollout(
                env, actor, replay_buffer, device
            )
            actor_info = actor.grpo_update(raw_actions, old_log_probs, rewards)

            # Train world model ensemble on replay buffer
            wm_loss = 0.0
            if len(replay_buffer) >= args.wm_batch_size:
                for _ in range(args.wm_updates_per_step):
                    batch = replay_buffer.sample(args.wm_batch_size, device)
                    wm_loss = world_model.train_step(batch[0], batch[1])

            # Track real reward
            mean_r = actor_info["mean_reward"]
            if real_reward_ema is None:
                real_reward_ema = mean_r
            else:
                real_reward_ema = 0.9 * real_reward_ema + 0.1 * mean_r
            if mean_r > best_real_reward:
                best_real_reward = mean_r

            # Update reference policy periodically
            if (step + 1) % args.ref_update_freq == 0:
                actor.save_reference()

            entry = {
                "step": step, "type": "real",
                "mean_reward": mean_r,
                "std_reward": actor_info["std_reward"],
                "policy_loss": actor_info["policy_loss"],
                "kl": actor_info["kl"],
                "wm_loss": wm_loss,
                "probes_mean": probes.mean(0).tolist(),
                "reward_ema": real_reward_ema,
                "dream_ratio": dream_ratio,
            }
            log(entry)

            if step % args.print_every == 0:
                print(
                    f"  Step {step:4d} [REAL]  | R={mean_r:+.4f} "
                    f"(±{actor_info['std_reward']:.4f}) | EMA={real_reward_ema:+.4f} | "
                    f"WM={wm_loss:.4f} | KL={actor_info['kl']:.4f} | "
                    f"Probes={[f'{v:.3f}' for v in probes.mean(0).tolist()]}"
                )

        # Periodic real validation during dreaming
        if is_dream and (step + 1) % args.validation_freq == 0:
            raw_a, old_lp, real_rewards, real_probes, gated_a = real_rollout(
                env, actor, replay_buffer, device
            )
            # Retrain world model on fresh data
            for _ in range(args.wm_updates_per_step * 2):
                batch = replay_buffer.sample(args.wm_batch_size, device)
                if batch is not None:
                    world_model.train_step(batch[0], batch[1])

            val_r = real_rewards.mean().item()
            print(f"  [VAL] Step {step:4d} | Real R={val_r:+.4f}")
            log({"step": step, "type": "validation", "real_reward": val_r,
                 "probes_mean": real_probes.mean(0).tolist()})

    # --- Generate sample outputs ---
    print("\n=== Sample Generations ===")
    # Get the current best action (mean of policy)
    with torch.no_grad():
        dist = actor._get_distribution()
        best_action = dist.mean
        if actor.gating:
            best_action = actor.gating(best_action)

    print("\n--- Baseline (no edit) ---")
    baseline_texts = env.generate(None, max_new_tokens=60)
    for i, t in enumerate(baseline_texts[:3]):
        print(f"  [{i}] {env.prompts[i]}{t}")

    print("\n--- With learned edit ---")
    edited_texts = env.generate(best_action, max_new_tokens=60)
    for i, t in enumerate(edited_texts[:3]):
        print(f"  [{i}] {env.prompts[i]}{t}")

    # --- Save checkpoint ---
    ckpt_path = os.path.join(args.log_dir, "checkpoint.pt")
    torch.save({
        "actor": actor.state_dict(),
        "world_model": world_model.state_dict(),
        "lora_basis": env.lora_basis.state_dict(),
        "args": vars(args),
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    log_file.close()
    print(f"Log saved to {log_path}")
    print(f"Best real reward: {best_real_reward:+.4f}, EMA: {real_reward_ema:+.4f}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="DreamerML v2 Training")

    # Architecture
    parser.add_argument("--K", type=int, default=16, help="Number of LoRA basis directions")
    parser.add_argument("--G", type=int, default=32, help="GRPO group size")
    parser.add_argument("--lora-scale", type=float, default=0.1, help="LoRA basis init scale")
    parser.add_argument("--actor-hidden", type=int, default=128)
    parser.add_argument("--wm-d-model", type=int, default=64)
    parser.add_argument("--wm-n-heads", type=int, default=4)
    parser.add_argument("--wm-n-layers", type=int, default=2)
    parser.add_argument("--n-ensemble", type=int, default=3, help="World model ensemble size")
    parser.add_argument("--use-gating", action="store_true", default=True)
    parser.add_argument("--no-gating", dest="use_gating", action="store_false")

    # Training
    parser.add_argument("--total-steps", type=int, default=400)
    parser.add_argument("--warmup-steps", type=int, default=80, help="Real-only warmup before dreaming")
    parser.add_argument("--max-dream-ratio", type=float, default=0.7, help="Max fraction of dream steps")
    parser.add_argument("--pessimism", type=float, default=1.0, help="Ensemble disagreement penalty")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--wm-lr", type=float, default=1e-3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)

    # Reward coefficients
    parser.add_argument("--alpha", type=float, default=1.0, help="KL sweet-spot weight")
    parser.add_argument("--beta", type=float, default=0.5, help="Selectivity weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Coherence weight")
    parser.add_argument("--delta", type=float, default=0.3, help="Entropy stability weight")
    parser.add_argument("--eta", type=float, default=0.001, help="Weight regularization")
    parser.add_argument("--kl-target", type=float, default=2.0, help="Target KL for sweet spot")

    # World model
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--wm-batch-size", type=int, default=32)
    parser.add_argument("--wm-updates-per-step", type=int, default=4)

    # Schedule
    parser.add_argument("--validation-freq", type=int, default=20)
    parser.add_argument("--ref-update-freq", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=5)

    # System
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
