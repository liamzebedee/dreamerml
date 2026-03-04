"""Training loop: Phase 1 (real rollouts) + Phase 2 (dreaming)."""

import argparse
import json
import os
import time

import torch

from env import BaseModelEnv
from agent import Actor
from agent_model import WorldModel, ReplayBuffer


def compute_reward_from_probes(probes, alpha=1.0, beta=0.5, gamma=0.3, lam=0.5, mu=0.3):
    """Compute reward from probe tensor without needing the env (for dreaming).

    probes: (..., 4) tensor [S_global, S_var, S_entropy, S_coherence]
    """
    S_global = probes[..., 0]
    S_var = probes[..., 1]
    S_coherence = probes[..., 3]

    R_coarse = S_global - lam * S_var
    R_fine = S_var - mu * S_global
    R_struct = S_coherence

    return alpha * R_coarse + beta * R_fine + gamma * R_struct


def real_rollout(env, actor, replay_buffer, device):
    """Sample G actions, run through real env, collect data.

    Returns dict with rewards and probes.
    """
    raw_actions, gated_actions, old_log_probs = actor.sample_raw()
    # Detach for env rollout — no backprop through frozen model
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

    return raw_actions, old_log_probs, rewards, all_probes, gated_actions


def dream_rollout(actor, world_model, reward_kwargs, device):
    """Sample G actions, predict probes with world model, compute rewards.

    No real env needed — pure dreaming.
    """
    raw_actions, gated_actions, old_log_probs = actor.sample_raw()

    with torch.no_grad():
        predicted_probes = world_model(gated_actions)
        rewards = compute_reward_from_probes(predicted_probes, **reward_kwargs)

    return raw_actions, old_log_probs, rewards, predicted_probes


@torch.no_grad()
def _update_basis_es(env, actor, reward_kwargs, lr, n_perturb=4, sigma=0.01):
    """Update LoRA basis directions using evolutionary strategy (finite differences).

    Perturb each basis param, measure reward change, update in improving direction.
    """
    # Get baseline reward with current basis
    _, gated, _ = actor.sample_raw(n=2)
    base_rewards = []
    for i in range(gated.shape[0]):
        r, _ = env.step(gated[i].detach())
        base_rewards.append(r)
    base_r = torch.stack(base_rewards).mean()

    for p in env.lora_basis.parameters():
        grad_est = torch.zeros_like(p)
        for _ in range(n_perturb):
            noise = torch.randn_like(p) * sigma
            # Positive perturbation
            p.data.add_(noise)
            _, gated_p, _ = actor.sample_raw(n=2)
            r_plus = []
            for i in range(gated_p.shape[0]):
                r, _ = env.step(gated_p[i].detach())
                r_plus.append(r)
            r_plus = torch.stack(r_plus).mean()
            # Restore
            p.data.sub_(noise)
            grad_est += (r_plus - base_r) * noise / (sigma ** 2)
        grad_est /= n_perturb
        # Gradient ascent (maximize reward)
        p.data.add_(lr * grad_est)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup ---
    print("Loading environment (Qwen2.5-0.5B)...")
    env = BaseModelEnv(
        K=args.K,
        device=device,
        lora_scale=args.lora_scale,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        eta=args.eta,
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

    world_model = WorldModel(
        K=args.K,
        d_model=args.wm_d_model,
        n_heads=args.wm_n_heads,
        n_layers=args.wm_n_layers,
        lr=args.wm_lr,
    ).to(device)

    replay_buffer = ReplayBuffer(capacity=args.buffer_size, K=args.K)

    reward_kwargs = dict(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        lam=env.lam, mu=env.mu,
    )

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    def log(entry):
        entry["timestamp"] = time.time()
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()

    print(f"K={args.K}, G={args.G}, Phase1={args.phase1_steps}, Phase2={args.phase2_steps}")

    # --- Phase 1: Real rollouts ---
    print("\n=== Phase 1: Real Rollouts ===")
    for step in range(args.phase1_steps):
        raw_actions, old_log_probs, rewards, probes, gated_actions = real_rollout(
            env, actor, replay_buffer, device
        )

        # Update actor with GRPO
        actor_info = actor.grpo_update(raw_actions, old_log_probs, rewards)

        # Train world model on replay buffer
        wm_loss = 0.0
        if len(replay_buffer) >= args.wm_batch_size:
            for _ in range(args.wm_updates_per_step):
                batch = replay_buffer.sample(args.wm_batch_size, device)
                wm_loss = world_model.train_step(batch[0], batch[1])

        # Update reference policy periodically
        if (step + 1) % args.ref_update_freq == 0:
            actor.save_reference()

        # Basis training disabled for v1 — random basis at proper scale works well
        # ES-based basis update was causing instability (weight explosion)

        entry = {
            "phase": 1, "step": step,
            "mean_reward": actor_info["mean_reward"],
            "std_reward": actor_info["std_reward"],
            "policy_loss": actor_info["policy_loss"],
            "kl": actor_info["kl"],
            "wm_loss": wm_loss,
            "probes_mean": probes.mean(0).tolist(),
        }
        log(entry)

        if step % args.print_every == 0:
            print(
                f"  Step {step:4d} | R={actor_info['mean_reward']:+.4f} "
                f"(±{actor_info['std_reward']:.4f}) | "
                f"WM={wm_loss:.4f} | KL={actor_info['kl']:.4f} | "
                f"Probes={[f'{v:.3f}' for v in probes.mean(0).tolist()]}"
            )

    # --- Phase 2: Dreaming ---
    print("\n=== Phase 2: Dreaming ===")
    for step in range(args.phase2_steps):
        # Dream: use world model
        raw_actions, old_log_probs, dream_rewards, pred_probes = dream_rollout(
            actor, world_model, reward_kwargs, device
        )
        actor_info = actor.grpo_update(raw_actions, old_log_probs, dream_rewards)

        # Periodic real validation
        if (step + 1) % args.validation_freq == 0:
            raw_a, old_lp, real_rewards, real_probes, gated_a = real_rollout(
                env, actor, replay_buffer, device
            )
            # Re-train world model on new real data
            for _ in range(args.wm_updates_per_step * 2):
                batch = replay_buffer.sample(args.wm_batch_size, device)
                if batch is not None:
                    world_model.train_step(batch[0], batch[1])

            val_entry = {
                "phase": 2, "step": step, "type": "validation",
                "real_mean_reward": real_rewards.mean().item(),
                "dream_mean_reward": actor_info["mean_reward"],
                "probes_mean": real_probes.mean(0).tolist(),
            }
            log(val_entry)
            print(
                f"  [VAL] Step {step:4d} | Real R={real_rewards.mean():+.4f} | "
                f"Dream R={actor_info['mean_reward']:+.4f}"
            )

        # Update reference policy
        if (step + 1) % args.ref_update_freq == 0:
            actor.save_reference()

        entry = {
            "phase": 2, "step": step,
            "dream_reward": actor_info["mean_reward"],
            "policy_loss": actor_info["policy_loss"],
            "kl": actor_info["kl"],
        }
        log(entry)

        if step % args.print_every == 0:
            print(
                f"  Step {step:4d} | Dream R={actor_info['mean_reward']:+.4f} | "
                f"KL={actor_info['kl']:.4f}"
            )

    # --- Save checkpoint ---
    ckpt_path = os.path.join(args.log_dir, "checkpoint.pt")
    torch.save({
        "actor": actor.state_dict(),
        "world_model": world_model.state_dict(),
        "lora_basis": env.lora_basis.state_dict(),
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")

    log_file.close()
    print(f"Log saved to {log_path}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="DreamerML v1 Training")

    # Architecture
    parser.add_argument("--K", type=int, default=16, help="Number of LoRA basis directions")
    parser.add_argument("--G", type=int, default=8, help="GRPO group size")
    parser.add_argument("--lora-scale", type=float, default=0.1, help="LoRA basis init scale")
    parser.add_argument("--actor-hidden", type=int, default=128)
    parser.add_argument("--wm-d-model", type=int, default=64)
    parser.add_argument("--wm-n-heads", type=int, default=4)
    parser.add_argument("--wm-n-layers", type=int, default=2)
    parser.add_argument("--use-gating", action="store_true", default=True)
    parser.add_argument("--no-gating", dest="use_gating", action="store_false")

    # Training
    parser.add_argument("--phase1-steps", type=int, default=100)
    parser.add_argument("--phase2-steps", type=int, default=200)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--wm-lr", type=float, default=1e-3)
    parser.add_argument("--basis-lr", type=float, default=1e-3)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--train-basis", action="store_true", default=True)
    parser.add_argument("--no-train-basis", dest="train_basis", action="store_false")

    # Reward coefficients
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=0.0001)

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

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
