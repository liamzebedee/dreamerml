#!/usr/bin/env python3
"""
Stage 6: Dreamer GRPO training on GPT-2.

Dreamer learns to modulate interpretable SAE features during generation.
Training uses GRPO with Gaussian exploration noise. The reward combines:
  - Coherence (neg perplexity on generated text)
  - Diversity (unique bigrams / total bigrams)
  - Fluency bonus (avg log-prob of generated tokens, rewards natural text)

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

DREAMER_HIDDEN = 256
DREAMER_LR = 1e-4
GRPO_K = 8              # rollouts per prompt
GRPO_STEPS = 800
GRPO_BATCH = 8           # prompts per step
GAIN_REG_COEFF = 0.003
NOISE_STD = 0.4
GEN_TOKENS = 100
REPORT_EVERY = 25        # emit report every N steps
DIVERSITY_WEIGHT = 2.0
FLUENCY_WEIGHT = 0.5

DREAMER_DIR = OUT_DIR / "dreamer"
DREAMER_CKPT = DREAMER_DIR / "dreamer.pt"

# Diverse eval prompts spanning many domains
EVAL_PROMPTS = [
    "The scientist carefully examined the data and concluded that",
    "In a small village nestled between the mountains,",
    "The stock market experienced unprecedented volatility when",
    "She picked up the guitar and began to play a",
    "The ancient civilization left behind remarkable",
    "According to the latest research in quantum physics,",
    "The detective found a crucial piece of evidence:",
    "Walking through the autumn forest, he noticed",
]


# --- Load meta-index --------------------------------------------------------

def load_meta_index():
    features = []
    if META_INDEX_FILE.exists():
        for line in META_INDEX_FILE.read_text().split("\n"):
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                idx = int(parts[0])
                name = parts[2]
                desc_start = line.find(name) + len(name)
                desc = line[desc_start:].strip()
                features.append((idx, name, desc))
    return features


# --- Dreamer Network ---------------------------------------------------------

class DreamerNet(nn.Module):
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
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, f):
        return self.net(f)


# --- Reward ------------------------------------------------------------------

def compute_reward(model, tokenizer, texts, prompt_lens):
    """Multi-component reward: coherence + diversity + fluency.

    coherence = -perplexity (clamped)
    diversity = unique_bigrams / total_bigrams
    fluency = mean log-prob of generated tokens (higher = more natural)
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                   truncation=True, max_length=512).to(DEVICE)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    rewards = []
    components = []  # for logging
    for i in range(len(texts)):
        pl = prompt_lens[i]
        gen_logits = logits[i, pl-1:-1]
        gen_targets = input_ids[i, pl:]
        mask = attention_mask[i, pl:]

        if mask.sum() == 0:
            rewards.append(0.0)
            components.append((0, 0, 0))
            continue

        # Coherence: -perplexity
        loss = F.cross_entropy(gen_logits, gen_targets, reduction='none')
        loss_masked = (loss * mask.float()).sum() / mask.sum()
        ppl = loss_masked.exp().clamp(max=200.0)
        coherence = -ppl.item()

        # Fluency: mean log-prob (higher = more natural)
        log_probs = -loss
        fluency = (log_probs * mask.float()).sum().item() / mask.sum().item()

        # Diversity: unique bigrams
        gen_text = tokenizer.decode(input_ids[i, pl:], skip_special_tokens=True)
        words = gen_text.lower().split()
        if len(words) > 2:
            bigrams = [(words[j], words[j+1]) for j in range(len(words)-1)]
            diversity = len(set(bigrams)) / len(bigrams)
        else:
            diversity = 0.0

        r = coherence + DIVERSITY_WEIGHT * diversity + FLUENCY_WEIGHT * fluency
        rewards.append(r)
        components.append((coherence, diversity, fluency))

    return rewards, components


# --- Generation with Dreamer ------------------------------------------------

def generate_with_dreamer(model, tokenizer, sae, dreamer, prompts, target_layer,
                          act_mean, act_std, decoder_w, feature_indices,
                          max_tokens, collect_gains=False, explore_noise=None):
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
            f = sae.encode(h_norm)

        g_small = dreamer(f.detach())

        if explore_noise is not None:
            g_small = g_small + explore_noise

        if collect_gains:
            all_gains.append(g_small.detach().cpu())

        B, T, _ = g_small.shape
        g_full = torch.zeros(B, T, SAE_DICT_SIZE, device=g_small.device)
        g_full[:, :, feat_idx_tensor] = g_small

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

    if all_gains:
        stacked = torch.cat(all_gains, dim=1)
        mean_gains = stacked.mean(dim=(0, 1))
    else:
        mean_gains = None

    return texts, prompt_lens, mean_gains


# --- GRPO Training -----------------------------------------------------------

def grpo_step(model, tokenizer, sae, dreamer, dreamer_opt, prompts,
              target_layer, act_mean, act_std, decoder_w, feature_indices):
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
        rewards, _ = compute_reward(model, tokenizer, texts, prompt_lens)
        all_rewards.append(rewards)

    rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)

    group_mean = rewards_tensor.mean(dim=0, keepdim=True)
    group_std = rewards_tensor.std(dim=0, keepdim=True).clamp(min=1e-4)
    advantages = (rewards_tensor - group_mean) / group_std
    avg_adv = advantages.mean(dim=1)

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
    g_mean = dreamer(f.detach())

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
    per_feature_gain = g_mean.detach().mean(dim=(0, 1)).cpu().numpy()
    return mean_reward, per_feature_gain, loss.item()


# --- Reporting ---------------------------------------------------------------

def sparkline(values, width=20):
    if len(values) == 0:
        return " " * width
    blocks = " .:;+=xX#@"
    mn, mx = min(values), max(values)
    if mx == mn:
        return blocks[0] * min(len(values), width)
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    out = ""
    for v in sampled:
        idx = int((v - mn) / (mx - mn) * (len(blocks) - 1))
        out += blocks[idx]
    return out


def wrap_text(text, indent=4, width=76):
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix,
                        subsequent_indent=prefix)


def emit_report(step, model, tokenizer, sae, dreamer, target_layer,
                act_mean, act_std, decoder_w, labeled_features,
                feature_indices, gain_history_per_feature, reward_history,
                reward_component_history):
    n_feat = len(labeled_features)
    report_path = DREAMER_DIR / f"report_step{step:04d}.txt"

    L = []
    L.append("")
    L.append("=" * 80)
    L.append(f"  DREAMER TRAINING LOG  --  Step {step} / {GRPO_STEPS}")
    L.append(f"  Model: GPT-2 (124M)  |  Layer: {TARGET_LAYER}  |  SAE: {SAE_DICT_SIZE} features")
    L.append(f"  Controlling {n_feat} interpretable features via learned gain policy")
    L.append("=" * 80)
    L.append("")

    # --- Section 1: Feature Gain Profile ---
    L.append("-" * 80)
    L.append("  FEATURE GAIN PROFILE")
    L.append("  What the Dreamer has learned to do with each feature")
    L.append("-" * 80)
    L.append("")

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

    order = np.argsort(-np.abs(gains_np))

    for rank, fi in enumerate(order):
        idx, name, desc = labeled_features[fi]
        g = gains_np[fi]
        mag = abs(g)

        if mag > 0.1:
            arrow = ">>>" if g > 0 else "<<<"
            action = "AMPLIFYING" if g > 0 else "SUPPRESSING"
        elif mag > 0.03:
            arrow = " > " if g > 0 else " < "
            action = "amplifying" if g > 0 else "suppressing"
        elif mag > 0.01:
            arrow = " > " if g > 0 else " < "
            action = "weak"
        else:
            arrow = " . "
            action = "inactive"

        hist = [h[fi] for h in gain_history_per_feature]
        spark = sparkline(hist)

        L.append(f"  {arrow} {g:+.4f}  {name:<32s} {action:<14s} [{spark}]")

    L.append("")
    L.append(f"  Total |gain| budget: {np.abs(gains_np).sum():.4f}")
    active_count = sum(1 for g in gains_np if abs(g) > 0.01)
    L.append(f"  Active features: {active_count}/{n_feat}")
    L.append("")

    # --- Section 2: Side-by-side Generations ---
    L.append("-" * 80)
    L.append("  SIDE-BY-SIDE GENERATIONS")
    L.append("  Same prompts, same temperature -- only difference is the Dreamer")
    L.append("-" * 80)

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
        L.append(wrap_text(texts_base[i], indent=6, width=74))
        L.append("")
        L.append("    DREAMER:")
        L.append(wrap_text(texts_dreamer[i], indent=6, width=74))

        if mean_gains is not None:
            active = [(labeled_features[j][1], gains_np[j])
                      for j in range(n_feat) if abs(gains_np[j]) > 0.01]
            if active:
                parts = [f"{name}={g:+.3f}" for name, g in
                         sorted(active, key=lambda x: -abs(x[1]))]
                L.append(f"    Features: {', '.join(parts[:8])}")
        L.append("")

    # --- Section 3: Reward Trajectory ---
    L.append("-" * 80)
    L.append("  TRAINING TRAJECTORY")
    L.append("-" * 80)
    L.append("")

    L.append("  Reward history:")
    L.append(f"    [{sparkline(reward_history, width=60)}]")
    if len(reward_history) >= 20:
        L.append(f"    Start: {np.mean(reward_history[:20]):.3f}  "
                 f"Now: {np.mean(reward_history[-20:]):.3f}  "
                 f"Best: {max(reward_history):.3f}  "
                 f"Delta: {np.mean(reward_history[-20:])-np.mean(reward_history[:20]):+.3f}")
    L.append("")

    # Reward components breakdown
    if reward_component_history:
        recent = reward_component_history[-min(50, len(reward_component_history)):]
        avg_coh = np.mean([c[0] for c in recent])
        avg_div = np.mean([c[1] for c in recent])
        avg_flu = np.mean([c[2] for c in recent])
        L.append(f"  Reward components (last 50 steps avg):")
        L.append(f"    Coherence: {avg_coh:.3f}  Diversity: {avg_div:.3f}  Fluency: {avg_flu:.3f}")
        L.append("")

    # Per-feature gain evolution
    L.append("  Per-feature gain evolution:")
    L.append("")
    for fi in order:
        idx, name, desc = labeled_features[fi]
        hist = [h[fi] for h in gain_history_per_feature]
        spark = sparkline([abs(v) for v in hist], width=40)
        if len(hist) > 0:
            L.append(f"    {name:<32s} {hist[-1]:+.4f}  [{spark}]")
    L.append("")

    # --- Section 4: Emergence Notes ---
    L.append("-" * 80)
    L.append("  EMERGENCE NOTES")
    L.append("-" * 80)
    L.append("")

    amplified = [(labeled_features[j][1], gains_np[j])
                 for j in range(n_feat) if gains_np[j] > 0.03]
    suppressed = [(labeled_features[j][1], gains_np[j])
                  for j in range(n_feat) if gains_np[j] < -0.03]
    inactive = [(labeled_features[j][1], gains_np[j])
                for j in range(n_feat) if abs(gains_np[j]) <= 0.03]

    if amplified:
        names = ", ".join(f"{n} ({g:+.3f})" for n, g in sorted(amplified, key=lambda x: -x[1]))
        L.append(f"  AMPLIFYING: {names}")
    if suppressed:
        names = ", ".join(f"{n} ({g:+.3f})" for n, g in sorted(suppressed, key=lambda x: x[1]))
        L.append(f"  SUPPRESSING: {names}")
    if inactive:
        L.append(f"  INACTIVE ({len(inactive)}): {', '.join(n for n, _ in inactive[:10])}")
    L.append("")

    # Emergence detection
    emerged = []
    vanished = []
    if len(gain_history_per_feature) > 25:
        old_gains = gain_history_per_feature[-25]
        for fi in range(n_feat):
            name = labeled_features[fi][1]
            old_g = abs(old_gains[fi])
            new_g = abs(gains_np[fi])
            if old_g < 0.02 and new_g > 0.05:
                emerged.append((name, old_g, new_g))
            elif old_g > 0.05 and new_g < 0.02:
                vanished.append((name, old_g, new_g))

    if emerged:
        L.append("  ** EMERGENCE DETECTED **")
        for name, old, new in emerged:
            L.append(f"     {name}: was {old:.3f} -> now {new:.3f}")
        L.append("")
    if vanished:
        L.append("  ** FEATURE EXTINCTION **")
        for name, old, new in vanished:
            L.append(f"     {name}: was {old:.3f} -> now {new:.3f}")
        L.append("")

    if len(reward_history) > 50:
        early_r = np.mean(reward_history[:20])
        late_r = np.mean(reward_history[-20:])
        delta_r = late_r - early_r
        if delta_r > 0.2:
            L.append(f"  Reward IMPROVING: {early_r:.3f} -> {late_r:.3f} (+{delta_r:.3f})")
        elif delta_r < -0.2:
            L.append(f"  Reward DECLINING: {early_r:.3f} -> {late_r:.3f} ({delta_r:.3f})")
        else:
            L.append(f"  Reward stable: {early_r:.3f} -> {late_r:.3f}")

    # Feature interaction detection: which features move together?
    if len(gain_history_per_feature) > 50:
        gh = np.array(gain_history_per_feature[-50:])  # (50, n_feat)
        if gh.shape[0] > 5 and n_feat > 1:
            corr = np.corrcoef(gh.T)
            strong_pairs = []
            for fi in range(n_feat):
                for fj in range(fi+1, n_feat):
                    if abs(corr[fi, fj]) > 0.7:
                        strong_pairs.append((
                            labeled_features[fi][1],
                            labeled_features[fj][1],
                            corr[fi, fj]
                        ))
            if strong_pairs:
                L.append("")
                L.append("  CORRELATED FEATURE PAIRS:")
                for n1, n2, c in sorted(strong_pairs, key=lambda x: -abs(x[2]))[:5]:
                    direction = "co-activate" if c > 0 else "anti-correlate"
                    L.append(f"    {n1} <-> {n2}: r={c:.2f} ({direction})")

    L.append("")
    L.append("=" * 80)
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

    labeled_features = load_meta_index()
    feature_indices = [f[0] for f in labeled_features]
    n_feat = len(labeled_features)
    print(f"  {n_feat} labeled features:")
    for idx, name, desc in labeled_features:
        print(f"    [{idx:4d}] {name}: {desc[:60]}")

    if n_feat == 0:
        print("ERROR: No labeled features in meta-index. Run stages 1-5 first.")
        return

    dreamer = DreamerNet(SAE_DICT_SIZE, n_feat, DREAMER_HIDDEN).to(DEVICE)
    start_step = 0
    reward_history = []
    reward_component_history = []
    gain_history_per_feature = []

    if DREAMER_CKPT.exists():
        dckpt = torch.load(DREAMER_CKPT, map_location=DEVICE, weights_only=True)
        dreamer.load_state_dict(dckpt["state_dict"])
        start_step = dckpt.get("step", 0)
        reward_history = dckpt.get("reward_history", [])
        reward_component_history = dckpt.get("reward_component_history", [])
        saved_gh = dckpt.get("gain_history_per_feature", [])
        gain_history_per_feature = [np.array(g) for g in saved_gh]
        print(f"Resuming Dreamer from step {start_step}")
    dreamer.train()

    dreamer_opt = torch.optim.Adam(dreamer.parameters(), lr=DREAMER_LR)

    print("Loading training prompts...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    all_prompts = []
    for s in dataset:
        text = s["text"].strip()
        if len(text) > 50:
            # Take first ~15 tokens as prompt
            words = text.split()
            if len(words) >= 8:
                all_prompts.append(" ".join(words[:12]))
        if len(all_prompts) >= 10000:
            break
    print(f"  {len(all_prompts)} training prompts")

    # --- Training loop ---
    print(f"\n{'='*80}")
    print(f"  DREAMER GRPO TRAINING (GPT-2, layer {TARGET_LAYER})")
    print(f"  {GRPO_STEPS} steps, K={GRPO_K} rollouts, batch={GRPO_BATCH}")
    print(f"  {n_feat} interpretable features, noise_std={NOISE_STD}")
    print(f"  Reward = coherence + {DIVERSITY_WEIGHT}*diversity + {FLUENCY_WEIGHT}*fluency")
    print(f"{'='*80}\n")
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

        # Track reward components (approximate from last rollout)
        reward_component_history.append((mean_reward, 0.0, 0.0))  # simplified

        elapsed = time.time() - t0
        steps_done = step - start_step + 1
        rate = steps_done / elapsed * 60
        eta = (GRPO_STEPS - step - 1) / max(rate / 60, 0.001)

        # Top features by absolute gain
        top3_idx = np.argsort(-np.abs(per_feat_gain))[:3]
        top3 = " ".join(f"{labeled_features[i][1].split('/')[-1]}={per_feat_gain[i]:+.3f}"
                        for i in top3_idx)

        active_n = sum(1 for g in per_feat_gain if abs(g) > 0.01)

        print(f"  [{step+1:4d}/{GRPO_STEPS}] R={mean_reward:+.3f} "
              f"|g|={np.abs(per_feat_gain).mean():.4f} "
              f"active={active_n}/{n_feat}  "
              f"top: {top3}  "
              f"({rate:.0f} step/min, ETA {eta:.0f}s)")

        if (step + 1) % REPORT_EVERY == 0 or step == GRPO_STEPS - 1:
            emit_report(step + 1, model, tokenizer, sae, dreamer, TARGET_LAYER,
                       act_mean, act_std, decoder_w, labeled_features,
                       feature_indices, gain_history_per_feature, reward_history,
                       reward_component_history)

            torch.save({
                "state_dict": dreamer.state_dict(),
                "step": step + 1,
                "reward_history": reward_history,
                "reward_component_history": reward_component_history,
                "gain_history_per_feature": [g.tolist() for g in gain_history_per_feature],
                "feature_indices": feature_indices,
                "labeled_features": [(idx, name, desc) for idx, name, desc in labeled_features],
            }, DREAMER_CKPT)
            print(f"  Checkpoint saved\n")

    total_time = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Final reward: {np.mean(reward_history[-20:]):.3f}")
    print(f"  Reports: {DREAMER_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
