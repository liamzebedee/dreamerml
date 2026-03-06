#!/usr/bin/env python3
"""
Stage 6: Dreamer GRPO training.

Dreamer is a small network that learns to modulate 10 interpretable SAE features
during generation. It operates ONLY on the labeled features from the meta-index,
so every gain is human-readable.

Training uses GRPO with Gaussian exploration noise. The reward combines coherence
(neg perplexity) with lexical diversity to create pressure toward creative,
varied text — forcing the Dreamer to discover which features help.

Reports are formatted as research logs showing per-feature gain trajectories,
side-by-side generations, and emergence narratives across training.
"""

import json, time, os, math, re, textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from config import *
from stage2_sae import SparseAutoencoder

# --- Dreamer Config ----------------------------------------------------------

DREAMER_HIDDEN = 128
DREAMER_LR = 3e-4
GRPO_K = 6              # rollouts per prompt
GRPO_STEPS = 500
GRPO_BATCH = 8           # prompts per step
GAIN_REG_COEFF = 0.005   # light reg — let gains grow
NOISE_STD = 0.3          # exploration noise (per-feature, only 10 dims)
GEN_TOKENS = 80
REPORT_EVERY = 50        # emit report every N steps
DIVERSITY_WEIGHT = 3.0   # reward weight for lexical diversity

DREAMER_DIR = OUT_DIR / "dreamer"
DREAMER_CKPT = DREAMER_DIR / "dreamer.pt"

# Fixed prompts used in EVERY report for consistent comparison
EVAL_PROMPTS = [
    "One day, a little girl named Lily found",
    "The brave knight walked into the dark cave",
    "Once upon a time, in a land full of",
    "A tiny mouse named Max wanted to",
    "One morning, the sun came up and",
]


# --- Load meta-index --------------------------------------------------------

def load_meta_index():
    """Load labeled feature indices and names from meta-index."""
    features = []  # list of (idx, shortname, description)
    if META_INDEX_FILE.exists():
        for line in META_INDEX_FILE.read_text().split("\n"):
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                idx = int(parts[0])
                name = parts[2]
                # Get description (everything after the shortname)
                desc_start = line.find(name) + len(name)
                desc = line[desc_start:].strip()
                features.append((idx, name, desc))
    return features


# --- Dreamer Network ---------------------------------------------------------

class DreamerNet(nn.Module):
    """MLP: full SAE features (2048) -> gains for N labeled features only."""
    def __init__(self, input_dim, n_features, hidden_dim):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features),
        )
        # Init near-zero
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, f):
        """f: (B, T, input_dim) -> g: (B, T, n_features)"""
        return self.net(f)


# --- Reward ------------------------------------------------------------------

def compute_reward(model, tokenizer, texts, prompt_lens):
    """Reward = coherence + diversity bonus.

    coherence = -perplexity (negative, higher is better)
    diversity = unique_words / total_words in generated text (0 to 1)

    Combined: R = coherence + DIVERSITY_WEIGHT * diversity
    This creates tension: pure repetition is coherent but low-diversity,
    pure randomness is diverse but incoherent. The Dreamer must find features
    that produce varied yet fluent text.
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                   truncation=True, max_length=512).to(DEVICE)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    rewards = []
    for i in range(len(texts)):
        pl = prompt_lens[i]
        gen_logits = logits[i, pl-1:-1]
        gen_targets = input_ids[i, pl:]
        mask = attention_mask[i, pl:]

        if mask.sum() == 0:
            rewards.append(0.0)
            continue

        # Coherence: -perplexity
        loss = F.cross_entropy(gen_logits, gen_targets, reduction='none')
        loss = (loss * mask.float()).sum() / mask.sum()
        ppl = loss.exp().clamp(max=100.0)
        coherence = -ppl.item()

        # Diversity: unique words / total words in generation
        gen_text = tokenizer.decode(input_ids[i, pl:], skip_special_tokens=True)
        words = gen_text.lower().split()
        if len(words) > 0:
            diversity = len(set(words)) / len(words)
        else:
            diversity = 0.0

        rewards.append(coherence + DIVERSITY_WEIGHT * diversity)

    return rewards


# --- Generation with Dreamer ------------------------------------------------

def generate_with_dreamer(model, tokenizer, sae, dreamer, prompts, target_layer,
                          act_mean, act_std, decoder_w, feature_indices,
                          max_tokens, collect_gains=False, explore_noise=None):
    """Generate text with Dreamer modulating only the labeled features.

    feature_indices: list of SAE feature indices the Dreamer controls.
    explore_noise: optional (n_features,) tensor added for exploration.
    Returns: texts, prompt_lens, mean_gains_per_feature (n_features,) or None.
    """
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                   truncation=True, max_length=64).to(DEVICE)
    prompt_lens = [enc["attention_mask"][i].sum().item() for i in range(len(prompts))]

    feat_idx_tensor = torch.tensor(feature_indices, device=DEVICE)
    all_gains = []

    def dreamer_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h_float = h.float()
        h_norm = (h_float - act_mean) / act_std
        with torch.no_grad():
            f = sae.encode(h_norm)  # (B, T, 2048)

        # Dreamer predicts gains for labeled features only
        g_small = dreamer(f.detach())  # (B, T, n_features)

        if explore_noise is not None:
            g_small = g_small + explore_noise

        if collect_gains:
            all_gains.append(g_small.detach().cpu())

        # Scatter into full gain vector
        B, T, _ = g_small.shape
        g_full = torch.zeros(B, T, SAE_DICT_SIZE, device=g_small.device)
        g_full[:, :, feat_idx_tensor] = g_small

        # Apply: delta = D @ g
        delta = torch.einsum("hd,btd->bth", decoder_w.float(), g_full)
        delta = delta * act_std.unsqueeze(0)
        h = h + delta.to(h.dtype)

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    hook = model.transformer.h[target_layer].register_forward_hook(dreamer_hook)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    hook.remove()

    texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out]

    # Compute mean signed gains per feature
    if all_gains:
        stacked = torch.cat(all_gains, dim=1)  # (B, T_total, n_features)
        mean_gains = stacked.mean(dim=(0, 1))   # (n_features,)
    else:
        mean_gains = None

    return texts, prompt_lens, mean_gains


# --- GRPO Training -----------------------------------------------------------

def grpo_step(model, tokenizer, sae, dreamer, dreamer_opt, prompts,
              target_layer, act_mean, act_std, decoder_w, feature_indices):
    """GRPO with Gaussian exploration noise over labeled features only."""

    n_feat = len(feature_indices)
    all_rewards = []
    all_eps = []

    for k in range(GRPO_K):
        eps_k = torch.randn(n_feat, device=DEVICE) * NOISE_STD
        all_eps.append(eps_k)

        texts, prompt_lens, _ = generate_with_dreamer(
            model, tokenizer, sae, dreamer, prompts, target_layer,
            act_mean, act_std, decoder_w, feature_indices,
            GEN_TOKENS, collect_gains=False, explore_noise=eps_k
        )
        rewards = compute_reward(model, tokenizer, texts, prompt_lens)
        all_rewards.append(rewards)

    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)  # (K, B)

    # GRPO advantages
    group_mean = rewards_tensor.mean(dim=0, keepdim=True)
    group_std = rewards_tensor.std(dim=0, keepdim=True).clamp(min=1e-4)
    advantages = (rewards_tensor - group_mean) / group_std
    avg_adv = advantages.mean(dim=1)  # (K,)

    # Forward pass with gradients
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                   truncation=True, max_length=64).to(DEVICE)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        outputs = model.transformer(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
    h = outputs.hidden_states[target_layer + 1].float()
    h_norm = (h - act_mean) / act_std
    with torch.no_grad():
        f = sae.encode(h_norm)

    dreamer_opt.zero_grad(set_to_none=True)
    g_mean = dreamer(f.detach())  # (B, T, n_feat)

    # REINFORCE for Gaussian policy
    policy_loss = torch.tensor(0.0, device=DEVICE)
    for k in range(GRPO_K):
        dot = (g_mean * all_eps[k]).sum(dim=-1).mean()
        policy_loss -= avg_adv[k] * dot / (NOISE_STD ** 2)
    policy_loss /= GRPO_K

    reg_loss = GAIN_REG_COEFF * g_mean.abs().mean()

    loss = policy_loss + reg_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dreamer.parameters(), 1.0)
    dreamer_opt.step()

    mean_reward = rewards_tensor.mean().item()
    per_feature_gain = g_mean.detach().mean(dim=(0, 1)).cpu().numpy()  # (n_feat,)
    return mean_reward, per_feature_gain, loss.item()


# --- Reporting ---------------------------------------------------------------

def sparkline(values, width=20):
    """Render a list of floats as a text sparkline."""
    if len(values) == 0:
        return " " * width
    blocks = " .:;+=xX#@"
    mn, mx = min(values), max(values)
    if mx == mn:
        return blocks[0] * min(len(values), width)
    # Sample down to width
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    out = ""
    for v in sampled:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        out += blocks[idx]
    return out


def wrap_text(text, indent=4, width=72):
    """Word-wrap text with indent."""
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                        subsequent_indent=prefix)


def emit_report(step, model, tokenizer, sae, dreamer, target_layer,
                act_mean, act_std, decoder_w, labeled_features,
                feature_indices, gain_history_per_feature, reward_history):
    """Emit a beautifully formatted research log entry."""

    n_feat = len(labeled_features)
    report_path = DREAMER_DIR / f"report_step{step:04d}.txt"

    L = []  # lines
    L.append("")
    L.append("=" * 72)
    L.append(f"  DREAMER TRAINING LOG  --  Step {step} / {GRPO_STEPS}")
    L.append("=" * 72)
    L.append("")

    # --- Section 1: Feature Gain Profile ---
    L.append("-" * 72)
    L.append("  FEATURE GAIN PROFILE")
    L.append("  What the Dreamer has learned to do with each feature")
    L.append("-" * 72)
    L.append("")

    # Get current gains by running eval prompts
    dreamer.eval()
    texts_dreamer, prompt_lens, mean_gains = generate_with_dreamer(
        model, tokenizer, sae, dreamer, EVAL_PROMPTS, target_layer,
        act_mean, act_std, decoder_w, feature_indices,
        GEN_TOKENS, collect_gains=True
    )

    if mean_gains is not None:
        gains_np = mean_gains.numpy()
    else:
        gains_np = np.zeros(n_feat)

    # Sort by absolute gain (most active first)
    order = np.argsort(-np.abs(gains_np))

    for rank, fi in enumerate(order):
        idx, name, desc = labeled_features[fi]
        g = gains_np[fi]
        mag = abs(g)

        if mag > 0.05:
            if g > 0:
                arrow = ">>>"
                action = "AMPLIFYING"
            else:
                arrow = "<<<"
                action = "SUPPRESSING"
        elif mag > 0.01:
            if g > 0:
                arrow = " > "
                action = "weak amplify"
            else:
                arrow = " < "
                action = "weak suppress"
        else:
            arrow = " . "
            action = "inactive"

        # Sparkline of this feature's gain history
        hist = [h[fi] for h in gain_history_per_feature]
        spark = sparkline(hist)

        L.append(f"  {arrow} {g:+.3f}  {name:<28s} {action:<16s} [{spark}]")

    L.append("")
    L.append(f"  Total |gain| budget: {np.abs(gains_np).sum():.3f}")
    L.append("")

    # --- Section 2: Side-by-side Generations ---
    L.append("-" * 72)
    L.append("  SIDE-BY-SIDE GENERATIONS")
    L.append("  Same prompts, same temperature — only difference is the Dreamer")
    L.append("-" * 72)

    # Generate baseline (no Dreamer)
    tokenizer.padding_side = "left"
    enc = tokenizer(EVAL_PROMPTS, return_tensors="pt", padding=True,
                   truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        out_base = model.generate(**enc, max_new_tokens=GEN_TOKENS, do_sample=True,
                                  temperature=0.8, top_p=0.9,
                                  pad_token_id=tokenizer.eos_token_id)
    texts_base = [tokenizer.decode(o, skip_special_tokens=True) for o in out_base]

    for i in range(len(EVAL_PROMPTS)):
        L.append("")
        L.append(f"  Prompt {i+1}: \"{EVAL_PROMPTS[i]}\"")
        L.append("")
        L.append("    BASELINE:")
        L.append(wrap_text(texts_base[i], indent=6, width=68))
        L.append("")
        L.append("    DREAMER:")
        L.append(wrap_text(texts_dreamer[i], indent=6, width=68))

        # Show which features were active for this generation
        if mean_gains is not None:
            active = [(labeled_features[j][1], gains_np[j])
                      for j in range(n_feat) if abs(gains_np[j]) > 0.01]
            if active:
                parts = [f"{name} {g:+.2f}" for name, g in
                         sorted(active, key=lambda x: -abs(x[1]))]
                L.append(f"    Active: {', '.join(parts)}")
        L.append("")

    # --- Section 3: Reward & Training Trajectory ---
    L.append("-" * 72)
    L.append("  TRAINING TRAJECTORY")
    L.append("-" * 72)
    L.append("")

    # Reward curve
    L.append("  Reward history (coherence + diversity):")
    L.append(f"    [{sparkline(reward_history, width=50)}]")
    if len(reward_history) >= 10:
        L.append(f"    Start: {np.mean(reward_history[:10]):.3f}  "
                 f"Now: {np.mean(reward_history[-10:]):.3f}  "
                 f"Best: {max(reward_history):.3f}")
    L.append("")

    # Per-feature gain evolution
    L.append("  Per-feature gain evolution (signed mean across eval prompts):")
    L.append("")
    for fi in order:
        idx, name, desc = labeled_features[fi]
        hist = [h[fi] for h in gain_history_per_feature]
        spark = sparkline([abs(v) for v in hist], width=40)
        if len(hist) > 0:
            L.append(f"    {name:<28s} {hist[-1]:+.3f}  [{spark}]")
    L.append("")

    # --- Section 4: Emergence Notes ---
    L.append("-" * 72)
    L.append("  EMERGENCE NOTES")
    L.append("-" * 72)
    L.append("")

    # Auto-generate observations about what's happening
    amplified = [(labeled_features[j][1], gains_np[j])
                 for j in range(n_feat) if gains_np[j] > 0.03]
    suppressed = [(labeled_features[j][1], gains_np[j])
                  for j in range(n_feat) if gains_np[j] < -0.03]
    inactive = [(labeled_features[j][1], gains_np[j])
                for j in range(n_feat) if abs(gains_np[j]) <= 0.03]

    if amplified:
        names = ", ".join(n for n, _ in sorted(amplified, key=lambda x: -x[1]))
        L.append(f"  The Dreamer is actively amplifying: {names}")
    if suppressed:
        names = ", ".join(n for n, _ in sorted(suppressed, key=lambda x: x[1]))
        L.append(f"  The Dreamer is actively suppressing: {names}")
    if inactive:
        names = ", ".join(n for n, _ in inactive)
        L.append(f"  Still inactive: {names}")

    # Check for recent emergence (feature crossed threshold in last 50 steps)
    if len(gain_history_per_feature) > 50:
        old_gains = gain_history_per_feature[-50]
        for fi in range(n_feat):
            name = labeled_features[fi][1]
            old_g = abs(old_gains[fi])
            new_g = abs(gains_np[fi])
            if old_g < 0.02 and new_g > 0.05:
                L.append(f"  ** NEW: {name} just emerged (was {old_g:.3f}, now {new_g:.3f})")
            elif old_g > 0.05 and new_g < 0.02:
                L.append(f"  ** GONE: {name} went inactive (was {old_g:.3f}, now {new_g:.3f})")

    if len(reward_history) > 50:
        early_r = np.mean(reward_history[:20])
        late_r = np.mean(reward_history[-20:])
        delta_r = late_r - early_r
        if delta_r > 0.1:
            L.append(f"  Reward improving: {early_r:.3f} -> {late_r:.3f} (+{delta_r:.3f})")
        elif delta_r < -0.1:
            L.append(f"  Reward declining: {early_r:.3f} -> {late_r:.3f} ({delta_r:.3f})")
        else:
            L.append(f"  Reward stable: {early_r:.3f} -> {late_r:.3f}")

    L.append("")
    L.append("=" * 72)
    L.append("")

    report_path.write_text("\n".join(L))
    print(f"  Report: {report_path}")
    dreamer.train()


# --- Main --------------------------------------------------------------------

def main():
    DREAMER_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Loading SAE...")
    ckpt = torch.load(SAE_FILE, map_location=DEVICE, weights_only=True)
    sae = SparseAutoencoder(ckpt["config"]["input_dim"], ckpt["config"]["dict_size"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    act_mean = ckpt["act_mean"].to(DEVICE)
    act_std = ckpt["act_std"].to(DEVICE)
    decoder_w = sae.decoder.weight.detach()

    # Load labeled features
    labeled_features = load_meta_index()
    feature_indices = [f[0] for f in labeled_features]
    n_feat = len(labeled_features)
    print(f"  {n_feat} labeled features: {[f[1] for f in labeled_features]}")

    if n_feat == 0:
        print("ERROR: No labeled features in meta-index. Run stages 1-5 first.")
        return

    # Create Dreamer (input=full SAE, output=labeled features only)
    dreamer = DreamerNet(SAE_DICT_SIZE, n_feat, DREAMER_HIDDEN).to(DEVICE)
    start_step = 0
    reward_history = []
    gain_history_per_feature = []  # list of (n_feat,) arrays

    if DREAMER_CKPT.exists():
        dckpt = torch.load(DREAMER_CKPT, map_location=DEVICE, weights_only=True)
        dreamer.load_state_dict(dckpt["state_dict"])
        start_step = dckpt.get("step", 0)
        reward_history = dckpt.get("reward_history", [])
        # Reconstruct gain history from saved data
        saved_gh = dckpt.get("gain_history_per_feature", [])
        gain_history_per_feature = [np.array(g) for g in saved_gh]
        print(f"Resuming Dreamer from step {start_step}")
    dreamer.train()

    dreamer_opt = torch.optim.Adam(dreamer.parameters(), lr=DREAMER_LR)

    # Load training prompts
    print("Loading prompts...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    all_prompts = []
    for s in dataset.select(range(min(5000, len(dataset)))):
        text = s["text"]
        words = text.split()
        if len(words) >= 5:
            all_prompts.append(" ".join(words[:8]))
    print(f"  {len(all_prompts)} prompts")

    # --- Training loop ---
    print(f"\n{'='*72}")
    print(f"  DREAMER GRPO TRAINING")
    print(f"  {GRPO_STEPS} steps, K={GRPO_K} rollouts, batch={GRPO_BATCH}")
    print(f"  {n_feat} interpretable features, noise_std={NOISE_STD}")
    print(f"  Reward = -perplexity + {DIVERSITY_WEIGHT} * diversity")
    print(f"{'='*72}\n")
    t0 = time.time()

    for step in range(start_step, GRPO_STEPS):
        idx = np.random.choice(len(all_prompts), GRPO_BATCH, replace=False)
        prompts = [all_prompts[i] for i in idx]

        mean_reward, per_feat_gain, loss = grpo_step(
            model, tokenizer, sae, dreamer, dreamer_opt, prompts,
            TARGET_LAYER, act_mean, act_std, decoder_w, feature_indices
        )

        reward_history.append(mean_reward)
        gain_history_per_feature.append(per_feat_gain.copy())

        elapsed = time.time() - t0
        steps_done = step - start_step + 1
        rate = steps_done / elapsed * 60
        eta = (GRPO_STEPS - step - 1) / max(rate / 60, 0.001)

        # Compact per-feature gain summary
        top3_idx = np.argsort(-np.abs(per_feat_gain))[:3]
        top3 = " ".join(f"{labeled_features[i][1].split('/')[-1]}={per_feat_gain[i]:+.3f}"
                        for i in top3_idx)

        print(f"  [{step+1:3d}/{GRPO_STEPS}] R={mean_reward:.3f} "
              f"|g|={np.abs(per_feat_gain).mean():.4f}  "
              f"top: {top3}  "
              f"({rate:.0f} step/min, ETA {eta:.0f}s)")

        # Periodic report
        if (step + 1) % REPORT_EVERY == 0 or step == GRPO_STEPS - 1:
            emit_report(step + 1, model, tokenizer, sae, dreamer, TARGET_LAYER,
                       act_mean, act_std, decoder_w, labeled_features,
                       feature_indices, gain_history_per_feature, reward_history)

            # Save checkpoint
            torch.save({
                "state_dict": dreamer.state_dict(),
                "step": step + 1,
                "reward_history": reward_history,
                "gain_history_per_feature": [g.tolist() for g in gain_history_per_feature],
                "feature_indices": feature_indices,
                "labeled_features": [(idx, name, desc) for idx, name, desc in labeled_features],
            }, DREAMER_CKPT)
            print(f"  Checkpoint saved\n")

    total_time = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  TRAINING COMPLETE: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Final reward: {np.mean(reward_history[-10:]):.3f}")
    print(f"  Reports: {DREAMER_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
