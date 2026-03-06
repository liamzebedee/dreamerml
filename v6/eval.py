"""
Evaluation: A/B/C validation metrics.
A) Dream usage is selective
B) Dream improves reward
C) Inverse dynamics matters (planner > random)
"""

import torch
import numpy as np
import json
import os
import time
from typing import Dict, List

from env import DreamEnv, K, clamp_action
from state import StateExtractor, ContextProjector
from planner import Planner, compute_objective
from forward_model import ForwardModel
from inverse_model import InverseModel
from dream_exec import DreamExecutor
from collect_transitions import check_quality, sample_random_action
from tasks import get_random_task, compute_reward


def evaluate_all(
    n_prompts: int = 100,
    checkpoint_dir: str = "checkpoints",
    data_dir: str = "data",
    device: str = "cuda",
) -> Dict:
    """Run all three validation checks."""
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    env = DreamEnv(device=device)
    extractor = StateExtractor(env.model, device=device)

    projector = ContextProjector(env.model.config.hidden_size).to(device)
    proj_path = f"{data_dir}/projector.pt"
    if os.path.exists(proj_path):
        projector.load_state_dict(torch.load(proj_path, map_location=device))

    norm_path = f"{data_dir}/normalizer.pt"
    if os.path.exists(norm_path):
        extractor.normalizer.load_state_dict(torch.load(norm_path, map_location=device), device=device)
        extractor.normalizer.freeze()

    fwd = ForwardModel().to(device)
    inv = InverseModel().to(device)

    fwd_path = f"{checkpoint_dir}/forward_model.pt"
    inv_path = f"{checkpoint_dir}/inverse_model.pt"

    if os.path.exists(fwd_path):
        fwd.load_state_dict(torch.load(fwd_path, map_location=device))
    if os.path.exists(inv_path):
        inv.load_state_dict(torch.load(inv_path, map_location=device))

    planner_obj = Planner(fwd, inv, device)

    results = {}

    # Test A: Dream selectivity (state-dependent)
    print("\n--- Test A: Dream Selectivity ---")
    results["A"] = test_selectivity(env, extractor, projector, planner_obj, n_prompts, device)

    # Test B: Dream improves outcomes
    print("\n--- Test B: Dream vs No-Dream ---")
    results["B"] = test_dream_benefit(env, extractor, projector, planner_obj, n_prompts // 2, device)

    # Test C: Planner vs Random
    print("\n--- Test C: Planner vs Random ---")
    results["C"] = test_planner_vs_random(env, extractor, projector, planner_obj, n_prompts // 2, device)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        for k, v in test_results.items():
            print(f"  {k}: {v}")

    # Save
    save_path = os.path.join(checkpoint_dir, "eval_results.json")
    with open(save_path, "w") as f:
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
    print(f"\nSaved results to {save_path}")
    return results


@torch.no_grad()
def test_selectivity(env, extractor, projector, planner, n_prompts, device):
    """Test A: Does the planner prefer dreaming in harder states?"""
    high_diff_scores = []
    low_diff_scores = []

    for i in range(n_prompts):
        prompt, _ = get_random_task()
        ids = env.encode(prompt)
        extractor.reset_history()

        # Generate some tokens
        gen = env.generate(ids, n_tokens=24)
        ctx = torch.cat([ids, gen.unsqueeze(0) if gen.dim() == 1 else gen], dim=-1)

        s = extractor.extract(ctx, normalize=True)

        # Difficulty: low top1 + high entropy + high rep_risk
        # (using raw features before normalization... in normalized space these are just the values)
        entropy = s[0].item()
        top1 = s[1].item()
        rep_risk = s[9].item()
        difficulty = entropy + (-top1) + rep_risk

        # Compute planning score (how beneficial is a dream here?)
        c_raw = extractor.extract_context_embedding(ids)
        c = projector(c_raw).squeeze(0)
        a = planner.plan(s, c)

        # Score: how much does the planned action improve the objective?
        s_pred = planner.F(s.unsqueeze(0), a.unsqueeze(0), c.unsqueeze(0))
        j_before = compute_objective(s.unsqueeze(0)).item()
        j_after = compute_objective(s_pred).item()
        improvement = j_after - j_before

        if difficulty > np.median([entropy]) if i == 0 else 0:
            high_diff_scores.append(improvement)
        else:
            low_diff_scores.append(improvement)

    # Split at median difficulty
    all_scores = high_diff_scores + low_diff_scores
    median_imp = np.median(all_scores) if all_scores else 0

    high_mean = np.mean(high_diff_scores) if high_diff_scores else 0
    low_mean = np.mean(low_diff_scores) if low_diff_scores else 0

    print(f"High difficulty states: mean improvement = {high_mean:.4f} (n={len(high_diff_scores)})")
    print(f"Low difficulty states: mean improvement = {low_mean:.4f} (n={len(low_diff_scores)})")
    print(f"Selective: {high_mean > low_mean}")

    return {
        "high_diff_improvement": high_mean,
        "low_diff_improvement": low_mean,
        "is_selective": high_mean > low_mean,
    }


@torch.no_grad()
def test_dream_benefit(env, extractor, projector, planner, n_prompts, device):
    """Test B: Dream-enabled vs dream-disabled."""
    dream_rewards = []
    no_dream_rewards = []

    for i in range(n_prompts):
        prompt, task_meta = get_random_task()
        ids = env.encode(prompt)

        # No dream: just generate
        extractor.reset_history()
        gen_no_dream = env.generate(ids, n_tokens=100)
        text_no_dream = env.decode(gen_no_dream)
        r_no = compute_reward(text_no_dream, task_meta)
        no_dream_rewards.append(r_no)

        # With dream: generate, dream at token 30, continue
        extractor.reset_history()
        gen_pre = env.generate(ids, n_tokens=30)
        ctx = torch.cat([ids, gen_pre.unsqueeze(0) if gen_pre.dim() == 1 else gen_pre], dim=-1)

        s = extractor.extract(ctx, normalize=True)
        c_raw = extractor.extract_context_embedding(ids)
        c = projector(c_raw).squeeze(0)
        a = planner.plan(s, c)

        env.apply_action(a)
        dream_tokens = env.generate(ctx, n_tokens=40)
        env.revert_action()

        quality = check_quality(dream_tokens, env)
        if quality == 1:
            dream_ctx = torch.cat([ctx, dream_tokens.unsqueeze(0) if dream_tokens.dim() == 1 else dream_tokens], dim=-1)
        else:
            dream_ctx = ctx

        gen_post = env.generate(dream_ctx, n_tokens=30)
        full_dream = torch.cat([ctx[:, ids.shape[-1]:],
                                dream_tokens.unsqueeze(0) if dream_tokens.dim() == 1 else dream_tokens,
                                gen_post.unsqueeze(0) if gen_post.dim() == 1 else gen_post], dim=-1)
        text_dream = env.decode(full_dream)
        r_dream = compute_reward(text_dream, task_meta)
        dream_rewards.append(r_dream)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_prompts}] dream={np.mean(dream_rewards):.3f} no_dream={np.mean(no_dream_rewards):.3f}")

    dream_mean = np.mean(dream_rewards)
    no_dream_mean = np.mean(no_dream_rewards)
    print(f"Dream reward: {dream_mean:.4f} ± {np.std(dream_rewards):.4f}")
    print(f"No-dream reward: {no_dream_mean:.4f} ± {np.std(no_dream_rewards):.4f}")
    print(f"Improvement: {dream_mean - no_dream_mean:.4f}")

    return {
        "dream_reward": dream_mean,
        "no_dream_reward": no_dream_mean,
        "improvement": dream_mean - no_dream_mean,
        "dream_wins": dream_mean > no_dream_mean,
    }


@torch.no_grad()
def test_planner_vs_random(env, extractor, projector, planner, n_prompts, device):
    """Test C: Planned actions vs random actions."""
    planned_rewards = []
    random_rewards = []
    planned_degen = 0
    random_degen = 0

    for i in range(n_prompts):
        prompt, task_meta = get_random_task()
        ids = env.encode(prompt)

        extractor.reset_history()
        gen_pre = env.generate(ids, n_tokens=30)
        ctx = torch.cat([ids, gen_pre.unsqueeze(0) if gen_pre.dim() == 1 else gen_pre], dim=-1)

        s = extractor.extract(ctx, normalize=True)
        c_raw = extractor.extract_context_embedding(ids)
        c = projector(c_raw).squeeze(0)

        # Planned action
        a_plan = planner.plan(s, c)
        env.apply_action(a_plan)
        dream_plan = env.generate(ctx, n_tokens=40)
        env.revert_action()
        q_plan = check_quality(dream_plan, env)
        if q_plan == 0:
            planned_degen += 1

        # Random action
        a_rand = sample_random_action(device)
        env.apply_action(a_rand)
        dream_rand = env.generate(ctx, n_tokens=40)
        env.revert_action()
        q_rand = check_quality(dream_rand, env)
        if q_rand == 0:
            random_degen += 1

        # Complete and reward
        for dream_tokens, rewards_list, q in [
            (dream_plan, planned_rewards, q_plan),
            (dream_rand, random_rewards, q_rand)
        ]:
            if q == 1:
                dream_ctx = torch.cat([ctx, dream_tokens.unsqueeze(0) if dream_tokens.dim() == 1 else dream_tokens], dim=-1)
            else:
                dream_ctx = ctx
            gen_post = env.generate(dream_ctx, n_tokens=30)
            full = torch.cat([ctx[:, ids.shape[-1]:],
                              dream_tokens.unsqueeze(0) if dream_tokens.dim() == 1 else dream_tokens,
                              gen_post.unsqueeze(0) if gen_post.dim() == 1 else gen_post], dim=-1)
            text = env.decode(full)
            r = compute_reward(text, task_meta)
            rewards_list.append(r)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_prompts}] planned={np.mean(planned_rewards):.3f} random={np.mean(random_rewards):.3f}")

    plan_mean = np.mean(planned_rewards)
    rand_mean = np.mean(random_rewards)
    print(f"Planned reward: {plan_mean:.4f} ± {np.std(planned_rewards):.4f}")
    print(f"Random reward: {rand_mean:.4f} ± {np.std(random_rewards):.4f}")
    print(f"Planned degen rate: {planned_degen/n_prompts:.2%}")
    print(f"Random degen rate: {random_degen/n_prompts:.2%}")

    return {
        "planned_reward": plan_mean,
        "random_reward": rand_mean,
        "planned_degen_rate": planned_degen / n_prompts,
        "random_degen_rate": random_degen / n_prompts,
        "planner_wins": plan_mean > rand_mean,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    evaluate_all(n_prompts=args.n)
