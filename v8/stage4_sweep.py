#!/usr/bin/env python3
"""Stage 4: Gain sweep generation. Saves each feature artifact as it completes."""

import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *
from stage2_sae import SparseAutoencoder

PROMPTS = [
    "Once upon a time, there was a little",
    "The cat sat on the",
    "Sara and Tom went to the",
    "One day, a big bear",
    "The little girl was very",
    "Mom said it was time to",
    "He found a shiny",
    "The dog ran to the",
    "She looked at the pretty",
    "They played in the",
    "It was a dark and",
    "The boy wanted to",
    "In the forest, there was a",
    "The baby started to",
    "When the sun came up,",
    "The old man had a",
    "She was so happy because",
    "The bird flew over the",
    "He didn't want to",
    "The princess lived in a",
]

def batch_generate(model, tokenizer, prompts, target_layer, delta, max_tokens):
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                   truncation=True, max_length=64).to(DEVICE)

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + delta.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    hook = model.transformer.h[target_layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                            temperature=0.8, top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id)
    hook.remove()
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out]

def write_artifact(path, feat_idx, stat, sweeps):
    lines = [f"FEATURE INDEX: {feat_idx}",
             f"Frequency: {stat['frequency']:.4f}",
             f"Mean act: {stat['mean_act']:.4f}",
             f"Max act: {stat['max_act']:.3f}",
             f"Cond mean: {stat['cond_mean']:.4f}",
             f"Score: {stat['score']:.6f}", ""]

    if stat.get("examples"):
        lines.append("HIGH-ACTIVATION EXAMPLES:")
        for ex in stat["examples"][:5]:
            lines.append(f"  act={ex['act']:.3f} pos={ex['pos']}")
            lines.append(f"    {ex['text'][:200]}")
        lines.append("")

    for si, (prompt, gens) in enumerate(sweeps):
        lines.append(f"--- Prompt {si+1}: {prompt[:100]}")
        for gain_str, text in gens.items():
            marker = " [BASELINE]" if gain_str == "0.0" else ""
            lines.append(f"  gain={gain_str}{marker}:")
            lines.append(f"    {text[:300]}")
        lines.append("")

    path.write_text("\n".join(lines))

def main():
    stats = json.loads(STATS_FILE.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check which features already have artifacts
    done = set()
    for s in stats:
        p = OUT_DIR / f"feature_{s['feature_idx']:04d}.txt"
        if p.exists():
            done.add(s["feature_idx"])
    remaining = [s for s in stats if s["feature_idx"] not in done]
    if not remaining:
        print(f"SKIP: All {len(stats)} feature artifacts exist")
        return

    print(f"Sweeping {len(remaining)} features ({len(done)} already done)")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()

    print("Loading SAE...")
    ckpt = torch.load(SAE_FILE, map_location=DEVICE, weights_only=True)
    sae = SparseAutoencoder(ckpt["config"]["input_dim"], SAE_DICT_SIZE).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    act_std = ckpt["act_std"].to(DEVICE)
    decoder_w = sae.decoder.weight.detach()

    t0 = time.time()
    for fi, stat in enumerate(remaining):
        feat_idx = stat["feature_idx"]
        feature_dir = (decoder_w[:, feat_idx] * act_std).to(DTYPE)

        # Batch all prompts per gain value (7 calls instead of 140)
        gain_outputs = {}
        for gain in SWEEP_GAINS:
            delta = gain * feature_dir
            texts = batch_generate(model, tokenizer, PROMPTS, TARGET_LAYER, delta, GEN_MAX_TOKENS)
            gain_outputs[f"{gain:.1f}"] = texts

        sweeps = []
        for pi, prompt in enumerate(PROMPTS):
            gens = {g: texts[pi] for g, texts in gain_outputs.items()}
            sweeps.append((prompt, gens))

        artifact_path = OUT_DIR / f"feature_{feat_idx:04d}.txt"
        write_artifact(artifact_path, feat_idx, stat, sweeps)

        elapsed = time.time() - t0
        rate = (fi + 1) / elapsed * 60
        eta = (len(remaining) - fi - 1) / max(rate / 60, 0.001)
        print(f"  [{fi+1}/{len(remaining)}] Feature {feat_idx} - {rate:.1f}/min, ETA {eta:.0f}s")

    print(f"Done: {len(remaining)} features in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
