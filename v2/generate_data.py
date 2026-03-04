"""Run the model at various knob settings and save all generations + probes to JSON.

This is the slow step (loads model, runs inference). Run once, then use
generate_report.py to iterate on the HTML output instantly.
"""

import json
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from env import BaseModelEnv

PROMPTS = [
    "The theory of relativity states that",
    "Once upon a time in a kingdom far away,",
    "The president announced today that",
    "def fibonacci(n):",
    "Dear diary, today I felt",
    "The most important thing about machine learning is",
]

STRENGTHS = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]


@torch.inference_mode()
def generate_and_probe(env, action, gen_inputs, max_new_tokens=100, temperature=0.7):
    """Apply weight deltas, compute probes + generate text."""
    deltas = env.lora_basis.compute_deltas(action)
    env._apply_deltas(deltas)
    try:
        # Probes
        env._hook_outputs.clear()
        out = env.model(**env.prompt_inputs)
        logits = out.logits
        mask = env.prompt_inputs["attention_mask"]
        token_counts = mask.sum(-1).clamp(min=1)

        log_p_new = F.log_softmax(logits, dim=-1)
        log_p_old = F.log_softmax(env.baseline_logits, dim=-1)
        kl = F.kl_div(log_p_new, log_p_old, log_target=True, reduction="none").sum(-1) * mask
        kl_per_prompt = kl.sum(-1) / token_counts
        S_global = kl_per_prompt.mean()
        S_var = kl_per_prompt.var()

        probs_new = F.softmax(logits, dim=-1)
        ent = -(probs_new * (probs_new + 1e-10).log()).sum(-1)
        S_entropy = (ent * mask).sum() / token_counts.sum() - env.baseline_entropy

        early_h = env._hook_outputs["early"]
        late_h = env._hook_outputs["late"]
        cos_new = F.cosine_similarity(early_h.flatten(1), late_h.flatten(1), dim=-1).mean()
        cos_old = F.cosine_similarity(
            env.baseline_early_hidden.flatten(1), env.baseline_late_hidden.flatten(1), dim=-1
        ).mean()
        S_coherence = cos_new - cos_old

        probes = {
            "KL": S_global.item(),
            "KL_var": S_var.item(),
            "Ent": S_entropy.item(),
            "Coh": S_coherence.item(),
        }
        del out, logits

        # Generation
        gen_out = env.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=env.tokenizer.pad_token_id,
        )
    finally:
        env._remove_deltas(deltas)

    texts = []
    for i, output in enumerate(gen_out):
        input_len = gen_inputs["input_ids"][i].shape[0]
        texts.append(env.tokenizer.decode(output[input_len:], skip_special_tokens=True))
    return texts, probes


@torch.inference_mode()
def generate_text(env, action, gen_inputs, max_new_tokens=100, temperature=0.7):
    """Generate text only."""
    if action is not None:
        deltas = env.lora_basis.compute_deltas(action)
        env._apply_deltas(deltas)
    try:
        gen_out = env.model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=env.tokenizer.pad_token_id,
        )
    finally:
        if action is not None:
            env._remove_deltas(deltas)

    texts = []
    for i, output in enumerate(gen_out):
        input_len = gen_inputs["input_ids"][i].shape[0]
        texts.append(env.tokenizer.decode(output[input_len:], skip_special_tokens=True))
    return texts


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate sweep data")
    parser.add_argument("--checkpoint", type=str, default="v2/runs/qwen_v2_selectivity3/checkpoint.pt")
    parser.add_argument("--output", type=str, default="v2/runs/qwen_v2_selectivity3/report_data.json")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lora-scale", type=float, default=0.1)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K

    print(f"Loading model on {device}...", flush=True)
    env = BaseModelEnv(K=K, device=device, lora_scale=args.lora_scale)

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    env.lora_basis.load_state_dict(ckpt["lora_basis"])
    print("Loaded.", flush=True)

    # Tokenize all prompts for generation
    gen_inputs = env.tokenizer(
        PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=64
    ).to(device)

    data = {"prompts": PROMPTS, "K": K}

    # Baseline
    print("Generating baseline...", flush=True)
    data["baseline"] = generate_text(env, None, gen_inputs)

    # Sweep each axis
    K_coarse = K // 2
    all_axes = list(range(K))
    data["sweeps"] = {}
    for axis_idx in all_axes:
        kind = "coarse" if axis_idx < K_coarse else "fine"
        print(f"Sweeping axis {axis_idx} ({kind})...", flush=True)
        axis_results = []
        for strength in STRENGTHS:
            if strength == 0.0:
                axis_results.append({
                    "strength": 0.0,
                    "texts": data["baseline"],
                    "probes": None,
                })
            else:
                action = torch.zeros(K, device=device)
                action[axis_idx] = strength
                texts, probes = generate_and_probe(env, action, gen_inputs)
                axis_results.append({
                    "strength": strength,
                    "texts": texts,
                    "probes": probes,
                })
                if device == "cuda":
                    torch.cuda.empty_cache()
        data["sweeps"][str(axis_idx)] = {"kind": kind, "results": axis_results}

    # Hierarchy: coarse x fine interactions
    data["hierarchy"] = []
    for ci in range(K_coarse):
        fi = ci + K_coarse  # paired fine axis
        print(f"Hierarchy: coarse[{ci}] x fine[{fi}]...", flush=True)
        combos = []
        for cv in [-2.0, 0.0, 2.0]:
            for fv in [-2.0, 0.0, 2.0]:
                if cv == 0.0 and fv == 0.0:
                    texts = data["baseline"]
                else:
                    action = torch.zeros(K, device=device)
                    action[ci] = cv
                    action[fi] = fv
                    texts = generate_text(env, action, gen_inputs)
                combos.append({"coarse_val": cv, "fine_val": fv, "texts": texts})
                if device == "cuda":
                    torch.cuda.empty_cache()
        data["hierarchy"].append({
            "coarse_idx": ci,
            "fine_idx": fi,
            "combos": combos,
        })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
