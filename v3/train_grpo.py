"""Phase 2: Train meta-adapter with GRPO to decide when to dream.

The perturbation policy is frozen. Only the meta-adapter (LoRA on q/v proj)
and dream decision head learn. The adapter learns to integrate dream outputs
and improve task performance.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import DreamEnv
from policy import PerturbationPolicy
from state import extract_state, STATE_DIM
from dream_executor import DreamExecutor


class MetaAdapter(nn.Module):
    """LoRA on q_proj/v_proj (rank 8) + dream decision head."""

    def __init__(self, model, rank=8, lora_alpha=16):
        super().__init__()
        self.rank = rank
        self.scale = lora_alpha / rank
        self.adapters = nn.ModuleDict()
        self.hooks = []

        for i, block in enumerate(model.transformer.h):
            for proj_name in ["q_proj", "v_proj"]:
                proj = getattr(block.attn.attention, proj_name)
                in_d = proj.in_features
                out_d = proj.out_features
                safe = f"h{i}_{proj_name}"
                self.adapters[f"{safe}_A"] = nn.Linear(in_d, rank, bias=False)
                self.adapters[f"{safe}_B"] = nn.Linear(rank, out_d, bias=False)
                nn.init.zeros_(self.adapters[f"{safe}_B"].weight)

        hidden_dim = model.config.hidden_size
        self.dream_head = nn.Linear(hidden_dim, 1)

    def install(self, model):
        """Install forward hooks that add LoRA output."""
        self.uninstall()
        for i, block in enumerate(model.transformer.h):
            for proj_name in ["q_proj", "v_proj"]:
                proj = getattr(block.attn.attention, proj_name)
                safe = f"h{i}_{proj_name}"
                A = self.adapters[f"{safe}_A"]
                B = self.adapters[f"{safe}_B"]
                s = self.scale

                def make_hook(A, B, s):
                    def hook(mod, inp, out):
                        return out + s * B(A(inp[0]))
                    return hook

                h = proj.register_forward_hook(make_hook(A, B, s))
                self.hooks.append(h)

    def uninstall(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def dream_probability(self, hidden_state):
        """P(dream) from last-layer hidden state."""
        return torch.sigmoid(self.dream_head(hidden_state))


def judge_story(text):
    """Heuristic story quality score ∈ [0, 1]."""
    words = text.split()
    if len(words) < 3:
        return 0.0

    score = 0.0
    # Length bonus
    score += min(len(words) / 30.0, 0.3)
    # Vocabulary diversity
    score += (len(set(words)) / len(words)) * 0.3
    # Punctuation = sentence structure
    score += 0.2 * min(sum(1 for c in text if c in ".!?,;") / 5.0, 1.0)
    # Penalize consecutive repeats
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])
    score -= repeats * 0.05

    return max(0.0, min(1.0, score))


def generate_with_dream(env, adapter, executor, prompt, device):
    """Generate text, inserting a dream after the first segment."""
    inputs = env.tokenizer(
        prompt, return_tensors="pt", max_length=32, truncation=True,
    ).to(device)

    # First segment: generate 15 tokens with adapter
    adapter.install(env.model)
    try:
        seg1 = env.model.generate(
            **inputs, max_new_tokens=15, temperature=0.9,
            top_p=0.95, do_sample=True,
            pad_token_id=env.tokenizer.pad_token_id,
        )

        # Check dream probability
        hidden_out = env.model(seg1, output_hidden_states=True)
        last_h = hidden_out.hidden_states[-1][:, -1, :]
        p_dream = adapter.dream_probability(last_h).item()
    finally:
        adapter.uninstall()

    seg1_text = env.tokenizer.decode(
        seg1[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )

    # Dream insertion
    dream_text = ""
    if p_dream > 0.5:
        dream_text, _, _ = executor.dream(
            seg1, torch.ones_like(seg1), max_tokens=48,
        )

    # Second segment: continue with dream context
    full_ctx = prompt + seg1_text
    if dream_text:
        full_ctx += " " + dream_text.strip()

    ctx_inputs = env.tokenizer(
        full_ctx, return_tensors="pt", max_length=128, truncation=True,
    ).to(device)

    adapter.install(env.model)
    try:
        seg2 = env.model.generate(
            **ctx_inputs, max_new_tokens=30, temperature=0.9,
            top_p=0.95, do_sample=True,
            pad_token_id=env.tokenizer.pad_token_id,
        )
    finally:
        adapter.uninstall()

    seg2_text = env.tokenizer.decode(
        seg2[0, ctx_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )

    final_text = seg1_text + (" [" + dream_text.strip() + "] "
                              if dream_text else "") + seg2_text
    return final_text, p_dream


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env = DreamEnv(model_name=args.model, K=args.K, device=device)

    # Load frozen policy
    policy = PerturbationPolicy(state_dim=STATE_DIM, K=args.K).to(device)
    ckpt = torch.load(args.policy_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    env.lora.load_state_dict(ckpt["lora"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    for p in env.lora.parameters():
        p.requires_grad_(False)

    executor = DreamExecutor(env, policy)
    adapter = MetaAdapter(env.model, rank=args.rank).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    task_prompts = [
        "Once upon a time there was a brave little",
        "The magical forest was full of",
        "A young boy named Tom wanted to",
        "In a kingdom far away the queen",
        "The three friends went on an adventure to find",
        "On a rainy day the old man told a story about",
    ]

    n_adapter = sum(p.numel() for p in adapter.parameters())
    print(f"Meta-adapter params: {n_adapter:,}")
    print(f"Training for {args.steps} steps, G={args.G}\n")

    for step in range(args.steps):
        prompt = task_prompts[step % len(task_prompts)]

        # Generate G completions, score each
        rewards = []
        texts = []
        dream_log_probs = []
        for g in range(args.G):
            text, p_dream = generate_with_dream(
                env, adapter, executor, prompt, device,
            )
            r = judge_story(text)
            rewards.append(r)
            texts.append(text)
            # Store dream probability for REINFORCE gradient
            dream_log_probs.append(
                torch.log(torch.tensor(p_dream + 1e-8, device=device))
            )

        rewards_t = torch.tensor(rewards, device=device)

        # Train dream head via supervised signal: dream when reward is high
        # Compute differentiable loss through the adapter
        adapter.install(env.model)
        try:
            inp = env.tokenizer(
                prompt, return_tensors="pt", max_length=32, truncation=True,
            ).to(device)
            h_out = env.model(inp["input_ids"], output_hidden_states=True)
            last_h = h_out.hidden_states[-1][:, -1, :]
            p_d = adapter.dream_probability(last_h)
            # Target: dream more when mean reward is high
            target = torch.tensor(
                [[1.0 if rewards_t.mean() > 0.5 else 0.0]],
                device=device,
            )
            dream_loss = F.binary_cross_entropy(p_d, target)
        finally:
            adapter.uninstall()

        opt.zero_grad()
        dream_loss.backward()
        opt.step()

        if step % args.print_every == 0:
            best_idx = rewards_t.argmax().item()
            print(
                f"Step {step:4d} | R={rewards_t.mean():.3f} "
                f"(±{rewards_t.std():.3f}) | best: {texts[best_idx][:80]}"
            )

    # Comparison: base vs adapter vs adapter+dream
    print("\n" + "=" * 70)
    print("COMPARISON: base vs adapter vs adapter+dream")
    print("=" * 70)
    for prompt in task_prompts[:3]:
        print(f"\n  Prompt: {prompt}")

        # Base
        base = env.generate(None, prompts=[prompt], max_new_tokens=50)
        print(f"  Base:          {base[0][:100]}")

        # Adapter only (no dream)
        adapter.install(env.model)
        try:
            inp = env.tokenizer(
                prompt, return_tensors="pt", max_length=32, truncation=True,
            ).to(device)
            with torch.inference_mode():
                out = env.model.generate(
                    **inp, max_new_tokens=50, temperature=0.9,
                    top_p=0.95, do_sample=True,
                    pad_token_id=env.tokenizer.pad_token_id,
                )
            adapter_text = env.tokenizer.decode(
                out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True,
            )
        finally:
            adapter.uninstall()
        print(f"  Adapter:       {adapter_text[:100]}")

        # Adapter + dream
        dream_text, _ = generate_with_dream(
            env, adapter, executor, prompt, device,
        )
        print(f"  Adapter+Dream: {dream_text[:100]}")

    torch.save({
        "adapter": adapter.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out_dir, "adapter.pt"))
    print(f"\nSaved adapter to {args.out_dir}/adapter.pt")


def main():
    p = argparse.ArgumentParser(description="DreamerML v3 - Phase 2 meta-adapter")
    p.add_argument("--model", default="roneneldan/TinyStories-1M")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--policy-path", default="runs/v3/policy.pt")
    p.add_argument("--out-dir", default="runs/v3")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
