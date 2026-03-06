#!/usr/bin/env python3
"""
Stage 7: Feature Exploration — creative probing of the SAE feature space.

Generates text while dynamically varying feature gains DURING generation:
- Ramps: gain rises from 0 to +3 over the course of a story
- Pulses: gain spikes mid-story then returns to zero
- Oscillations: gain swings between positive and negative
- Combos: multiple features modulated simultaneously
- Collisions: opposing features amplified at once
- The Dreamer's own learned strategy

Produces a single beautiful exploration.txt artifact.
"""

import json, time, os, textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *
from stage2_sae import SparseAutoencoder

GEN_TOKENS = 120  # longer generations to see evolution
DREAMER_DIR = OUT_DIR / "dreamer"

PROMPTS = [
    "One day, a little girl named Lily found",
    "The brave knight walked into the dark cave",
    "A tiny mouse named Max wanted to",
]


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


def wrap(text, indent=6, width=68):
    return textwrap.fill(text, width=width, initial_indent=" "*indent,
                        subsequent_indent=" "*indent)


def generate_with_dynamic_gains(model, tokenizer, sae, prompts, target_layer,
                                 act_mean, act_std, decoder_w,
                                 gain_schedule, max_tokens):
    """Generate text with a gain_schedule function.

    gain_schedule(token_idx, total_tokens) -> dict mapping SAE feature index to gain value.
    Called at each forward pass during generation to get current gains.
    """
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                   truncation=True, max_length=64).to(DEVICE)

    token_counter = [0]  # mutable counter for hook

    def dynamic_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        gains = gain_schedule(token_counter[0], max_tokens)
        token_counter[0] += 1

        if not gains:
            if isinstance(output, tuple):
                return output
            return h

        # Build gain vector
        g = torch.zeros(SAE_DICT_SIZE, device=DEVICE)
        for feat_idx, gain_val in gains.items():
            g[feat_idx] = gain_val

        # Apply: delta = D @ g * act_std
        delta = (decoder_w.float() @ g)  # (hidden_dim,)
        delta = delta * act_std
        h = h + delta.to(h.dtype).unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    hook = model.transformer.h[target_layer].register_forward_hook(dynamic_hook)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    hook.remove()
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]


def generate_baseline(model, tokenizer, prompts, max_tokens):
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                   truncation=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                            temperature=0.7, top_p=0.95,
                            pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]


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
    act_mean = ckpt["act_mean"].to(DEVICE)
    act_std = ckpt["act_std"].to(DEVICE)
    decoder_w = sae.decoder.weight.detach()

    labeled_features = load_meta_index()
    feat_by_name = {f[1]: f[0] for f in labeled_features}
    feat_desc = {f[1]: f[2] for f in labeled_features}
    print(f"  {len(labeled_features)} labeled features")

    L = []  # output lines

    L.append("")
    L.append("=" * 76)
    L.append("  DREAMER FEATURE EXPLORATION")
    L.append("  What happens when you turn the knobs during generation?")
    L.append("=" * 76)
    L.append("")

    # ---- Baseline ----
    print("Generating baselines...")
    baseline = generate_baseline(model, tokenizer, PROMPTS, GEN_TOKENS)

    L.append("-" * 76)
    L.append("  BASELINE (no feature modulation)")
    L.append("-" * 76)
    for i, (p, t) in enumerate(zip(PROMPTS, baseline)):
        L.append(f"\n  Prompt: \"{p}\"")
        L.append(wrap(t))
    L.append("")

    # ==== EXPERIMENT 1: Single feature ramps ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 1: FEATURE RAMPS")
    L.append("  Gain ramps from 0 to +3 during generation, then stays at +3.")
    L.append("  Watch the text shift as each feature activates.")
    L.append("=" * 76)

    ramp_features = ["emotion/joy", "narrative/whimsy", "specificity/entities",
                     "tone/wholesome", "childhood/innocence"]

    for fname in ramp_features:
        if fname not in feat_by_name:
            continue
        fidx = feat_by_name[fname]
        print(f"  Ramp: {fname}...")

        def make_ramp(fi):
            def schedule(t, total):
                progress = min(t / (total * 0.5), 1.0)  # ramp over first half
                return {fi: progress * 3.0}
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_ramp(fidx), GEN_TOKENS
        )

        L.append(f"\n  RAMP: {fname} (0 -> +3.0)")
        L.append(f"  {feat_desc.get(fname, '')[:70]}")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 2: Suppression ramps ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 2: SUPPRESSION RAMPS")
    L.append("  Gain ramps from 0 to -3 during generation.")
    L.append("  Watch what disappears from the text.")
    L.append("=" * 76)

    for fname in ["emotion/joy", "narrative/whimsy", "tone/wholesome"]:
        if fname not in feat_by_name:
            continue
        fidx = feat_by_name[fname]
        print(f"  Suppress: {fname}...")

        def make_suppress(fi):
            def schedule(t, total):
                progress = min(t / (total * 0.5), 1.0)
                return {fi: -progress * 3.0}
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_suppress(fidx), GEN_TOKENS
        )

        L.append(f"\n  SUPPRESS: {fname} (0 -> -3.0)")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 3: Oscillations ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 3: OSCILLATIONS")
    L.append("  Gain oscillates between +2 and -2 every ~20 tokens.")
    L.append("  The text should swing between two modes.")
    L.append("=" * 76)

    for fname in ["emotion/joy", "tone/wholesome", "specificity/entities"]:
        if fname not in feat_by_name:
            continue
        fidx = feat_by_name[fname]
        print(f"  Oscillate: {fname}...")

        def make_oscillate(fi):
            def schedule(t, total):
                wave = np.sin(2 * np.pi * t / 40)  # period ~40 tokens
                return {fi: wave * 2.0}
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_oscillate(fidx), GEN_TOKENS
        )

        L.append(f"\n  OSCILLATE: {fname} (sin wave, amplitude 2.0, period ~40 tokens)")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 4: Pulses ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 4: MID-STORY PULSES")
    L.append("  No modulation for 40 tokens, then SPIKE to +5 for 20 tokens,")
    L.append("  then back to 0. Watch the story shift and recover.")
    L.append("=" * 76)

    for fname in ["narrative/whimsy", "emotion/comfort", "character/female-positive"]:
        if fname not in feat_by_name:
            continue
        fidx = feat_by_name[fname]
        print(f"  Pulse: {fname}...")

        def make_pulse(fi):
            def schedule(t, total):
                if 40 <= t < 60:
                    return {fi: 5.0}
                return {}
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_pulse(fidx), GEN_TOKENS
        )

        L.append(f"\n  PULSE: {fname} (+5.0 at tokens 40-60)")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 5: Feature Combinations ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 5: FEATURE COMBINATIONS")
    L.append("  Multiple features modulated simultaneously.")
    L.append("  Exploring the interaction between features.")
    L.append("=" * 76)

    combos = [
        ("Joy + Whimsy", {"emotion/joy": 2.0, "narrative/whimsy": 2.0}),
        ("Wholesome + Protagonist + Entities", {"tone/wholesome": 2.0, "character/protagonist": 2.0, "specificity/entities": 2.0}),
        ("All positive emotions maxed", {"emotion/joy": 3.0, "emotion/comfort": 3.0, "tone/wholesome": 3.0}),
        ("Suppress all emotion, amp entities", {"emotion/joy": -3.0, "emotion/comfort": -3.0, "tone/wholesome": -3.0, "specificity/entities": 3.0}),
        ("Dreamer's strategy (manual)", {"tone/wholesome": 2.0, "narrative/character-focus": 2.0, "character/protagonist": 1.5, "emotion/joy": -2.0, "emotion/comfort": -2.0, "childhood/innocence": -2.0}),
    ]

    for combo_name, combo_gains in combos:
        print(f"  Combo: {combo_name}...")
        resolved_gains = {}
        for fname, gval in combo_gains.items():
            if fname in feat_by_name:
                resolved_gains[feat_by_name[fname]] = gval

        def make_static(gains):
            def schedule(t, total):
                return gains
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_static(resolved_gains), GEN_TOKENS
        )

        gain_str = ", ".join(f"{k} {v:+.1f}" for k, v in combo_gains.items())
        L.append(f"\n  COMBO: {combo_name}")
        L.append(f"  Gains: {gain_str}")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 6: The Crossfade ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 6: THE CROSSFADE")
    L.append("  Start with one feature at +3, crossfade to another feature at +3.")
    L.append("  Watch the story's character change mid-generation.")
    L.append("=" * 76)

    crossfades = [
        ("Joy -> Whimsy", "emotion/joy", "narrative/whimsy"),
        ("Entities -> Innocence", "specificity/entities", "childhood/innocence"),
        ("Comfort -> Protagonist", "emotion/comfort", "character/protagonist"),
    ]

    for xname, f1name, f2name in crossfades:
        if f1name not in feat_by_name or f2name not in feat_by_name:
            continue
        f1idx = feat_by_name[f1name]
        f2idx = feat_by_name[f2name]
        print(f"  Crossfade: {xname}...")

        def make_crossfade(fi1, fi2):
            def schedule(t, total):
                progress = min(t / total, 1.0)
                return {fi1: 3.0 * (1.0 - progress), fi2: 3.0 * progress}
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_crossfade(f1idx, f2idx), GEN_TOKENS
        )

        L.append(f"\n  CROSSFADE: {f1name} -> {f2name}")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 7: Extreme cranking ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 7: EXTREME CRANKING (gain = +10)")
    L.append("  What happens when you push a feature WAY beyond normal range?")
    L.append("  (Normal sweep range is -3 to +3)")
    L.append("=" * 76)

    for fname in ["emotion/joy", "narrative/whimsy", "specificity/entities",
                   "tone/wholesome", "character/protagonist"]:
        if fname not in feat_by_name:
            continue
        fidx = feat_by_name[fname]
        print(f"  Extreme: {fname}...")

        def make_extreme(fi):
            def schedule(t, total):
                return {fi: 10.0}
            return schedule

        texts = generate_with_dynamic_gains(
            model, tokenizer, sae, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w,
            make_extreme(fidx), GEN_TOKENS
        )

        L.append(f"\n  EXTREME +10: {fname}")
        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n    \"{p}\"")
            L.append(wrap(t, indent=6))
        L.append("")

    # ==== EXPERIMENT 8: Trained Dreamer ====
    L.append("")
    L.append("=" * 76)
    L.append("  EXPERIMENT 8: THE TRAINED DREAMER")
    L.append("  The network that learned its own strategy through 500 steps of GRPO.")
    L.append("  It decides which features to modulate based on the current hidden state.")
    L.append("=" * 76)

    dreamer_ckpt = DREAMER_DIR / "dreamer.pt"
    if dreamer_ckpt.exists():
        from stage6_dreamer import DreamerNet, generate_with_dreamer

        dckpt = torch.load(dreamer_ckpt, map_location=DEVICE, weights_only=True)
        feature_indices = dckpt.get("feature_indices", [f[0] for f in labeled_features])
        n_feat = len(feature_indices)

        dreamer = DreamerNet(SAE_DICT_SIZE, n_feat, 128).to(DEVICE)
        dreamer.load_state_dict(dckpt["state_dict"])
        dreamer.eval()

        texts, _, mean_gains = generate_with_dreamer(
            model, tokenizer, sae, dreamer, PROMPTS, TARGET_LAYER,
            act_mean, act_std, decoder_w, feature_indices,
            GEN_TOKENS, collect_gains=True
        )

        if mean_gains is not None:
            gains_np = mean_gains.numpy()
            L.append("\n  Dreamer's learned gain profile:")
            for fi in range(n_feat):
                idx, name, desc = labeled_features[fi]
                g = gains_np[fi]
                direction = "AMPLIFY" if g > 0.03 else "SUPPRESS" if g < -0.03 else "~neutral"
                L.append(f"    {g:+.3f}  {name:<28s}  {direction}")
            L.append("")

        for i, (p, t) in enumerate(zip(PROMPTS, texts)):
            L.append(f"\n  Prompt: \"{p}\"")
            L.append("")
            L.append("    Baseline:")
            L.append(wrap(baseline[i], indent=6))
            L.append("")
            L.append("    Dreamer:")
            L.append(wrap(t, indent=6))
        L.append("")
    else:
        L.append("\n  (No trained Dreamer checkpoint found — run stage 6 first)")

    L.append("")
    L.append("=" * 76)
    L.append(f"  Generated {time.strftime('%Y-%m-%d %H:%M')}")
    L.append("=" * 76)
    L.append("")

    out_path = DREAMER_DIR / "exploration.txt"
    out_path.write_text("\n".join(L))
    print(f"\nExploration complete: {out_path}")
    print(f"  {len(L)} lines")


if __name__ == "__main__":
    main()
