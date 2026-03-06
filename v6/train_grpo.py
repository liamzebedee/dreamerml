"""
GRPO training of meta-adapter: teaches model when to dream and how to use dream text.
LoRA rank 8 on q_proj, v_proj all layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from typing import List, Dict

from env import DreamEnv, K
from state import StateExtractor, ContextProjector
from planner import Planner
from forward_model import ForwardModel
from inverse_model import InverseModel
from dream_exec import DreamExecutor
from tasks import get_random_task, compute_reward


LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
GROUP_SIZE = 8
CLIP_RATIO = 0.2
LR = 1e-4
MAX_GEN_TOKENS = 150


class LoRALayer(nn.Module):
    """LoRA adapter for a linear layer."""

    def __init__(self, original: nn.Linear, rank: int = LORA_RANK, alpha: int = LORA_ALPHA):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_dim = original.in_features
        out_dim = original.out_features

        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))
        self.dropout = nn.Dropout(LORA_DROPOUT)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out


def apply_lora_adapter(model) -> List[nn.Parameter]:
    """Apply LoRA to q_proj and v_proj in all layers. Returns trainable params."""
    trainable_params = []

    for i in range(model.config.num_layers):
        block = model.transformer.h[i]
        attn = block.attn.attention

        # Replace q_proj
        q_lora = LoRALayer(attn.q_proj)
        attn.q_proj = q_lora
        trainable_params.extend([q_lora.lora_A, q_lora.lora_B])

        # Replace v_proj
        v_lora = LoRALayer(attn.v_proj)
        attn.v_proj = v_lora
        trainable_params.extend([v_lora.lora_A, v_lora.lora_B])

    return trainable_params


def train_grpo(
    n_prompts: int = 1000,
    checkpoint_dir: str = "checkpoints",
    data_dir: str = "data",
    save_path: str = "checkpoints/adapter.pt",
    device: str = "cuda",
):
    """Train meta-adapter with GRPO."""
    # Load environment and models
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

    # Load F and I
    fwd = ForwardModel().to(device)
    fwd_path = f"{checkpoint_dir}/forward_model.pt"
    if os.path.exists(fwd_path):
        fwd.load_state_dict(torch.load(fwd_path, map_location=device))

    inv = InverseModel().to(device)
    inv_path = f"{checkpoint_dir}/inverse_model.pt"
    if os.path.exists(inv_path):
        inv.load_state_dict(torch.load(inv_path, map_location=device))

    planner_obj = Planner(fwd, inv, device)

    # Apply LoRA adapter
    adapter_params = apply_lora_adapter(env.model)
    optimizer = torch.optim.Adam(adapter_params, lr=LR)

    dream_exec = DreamExecutor(env, extractor, projector, planner_obj, device=device)

    print(f"Training GRPO with {n_prompts} prompts, group size {GROUP_SIZE}")
    print(f"Adapter params: {sum(p.numel() for p in adapter_params)}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_rewards = []

    for prompt_idx in range(n_prompts):
        prompt, task_meta = get_random_task()
        prompt_ids = env.encode(prompt)

        # Generate GROUP_SIZE completions
        group_rewards = []
        group_log_probs = []

        for g in range(GROUP_SIZE):
            dream_exec.reset()
            context_ids = prompt_ids.clone()
            total_log_prob = torch.tensor(0.0, device=device)
            n_gen = 0
            dream_inserted = False

            # Generate tokens one at a time
            for t in range(MAX_GEN_TOKENS):
                outputs = env.model(context_ids)
                logits = outputs.logits[:, -1, :] / 0.9  # temperature

                # Top-p sampling
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumprobs - torch.softmax(sorted_logits, dim=-1) >= 0.95
                sorted_logits[mask] = float("-inf")
                probs = torch.softmax(sorted_logits, dim=-1)

                token_idx = torch.multinomial(probs, 1)
                token = sorted_idx.gather(-1, token_idx)

                # Log prob for policy gradient
                log_p = F.log_softmax(logits, dim=-1)
                token_log_prob = log_p.gather(-1, token).squeeze()
                total_log_prob = total_log_prob + token_log_prob

                context_ids = torch.cat([context_ids, token], dim=-1)
                n_gen += 1

                # Maybe dream at token 30-50
                if not dream_inserted and 30 <= t <= 50 and np.random.random() < 0.3:
                    if dream_exec.dream_count < dream_exec.max_dreams:
                        dream_text = dream_exec.execute_dream(context_ids, prompt_ids)
                        if dream_text:
                            dream_ids = env.encode(dream_text)
                            context_ids = torch.cat([context_ids, dream_ids], dim=-1)
                            dream_inserted = True

            # Compute reward
            full_text = env.decode(context_ids[:, prompt_ids.shape[-1]:])
            reward = compute_reward(full_text, task_meta)

            group_rewards.append(reward)
            group_log_probs.append(total_log_prob)

        # GRPO update
        rewards_t = torch.tensor(group_rewards, device=device)
        baseline = rewards_t.mean()
        advantages = rewards_t - baseline

        # Normalize advantages
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        # Policy gradient loss
        log_probs_t = torch.stack(group_log_probs)
        loss = -(advantages.detach() * log_probs_t).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
        optimizer.step()

        all_rewards.extend(group_rewards)

        if (prompt_idx + 1) % 10 == 0:
            recent = all_rewards[-80:]
            print(f"[{prompt_idx+1}/{n_prompts}] "
                  f"reward={np.mean(recent):.3f}±{np.std(recent):.3f} "
                  f"loss={loss.item():.4f}")

        if (prompt_idx + 1) % 100 == 0:
            torch.save({p: p.data for p in adapter_params}, save_path)
            print(f"Saved adapter checkpoint")

    # Final save
    torch.save({p: p.data for p in adapter_params}, save_path)
    print(f"\nGRPO training done. Final reward: {np.mean(all_rewards[-100:]):.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    train_grpo(n_prompts=args.n)
