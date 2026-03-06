"""Generate sweep data: test individual basis directions at ±3 on Qwen instruct.

Like v1/gen_examples.py but for Qwen2.5-0.5B-Instruct.
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from env import BaseModelEnv

PROMPTS = [
    "Write a sad poem about losing a friend.",
    "Tell me a scary campfire story.",
    "Write a love letter from Romeo to Juliet.",
    "Roast me like I'm at a comedy show.",
    "Write an angry rant about people who don't use turn signals.",
    "Describe a peaceful morning in a small village.",
    "Write a dramatic monologue from a villain explaining their plan.",
    "Tell me a story about a dog who waits for its owner to come home.",
]

STRENGTHS = [-3.0, 3.0]


@torch.inference_mode()
def generate_multi(env, action, prompts, max_new_tokens=100, temperature=0.3):
    """Generate text for multiple prompts with weight edit applied."""
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted.append(env.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    inputs = env.tokenizer(
        formatted, return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    ).to(env.device)

    if action is not None:
        deltas = env.lora_basis.compute_deltas(action)
        env._apply_deltas(deltas)
    try:
        outputs = env.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=env.tokenizer.pad_token_id,
        )
    finally:
        if action is not None:
            env._remove_deltas(deltas)

    texts = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        texts.append(env.tokenizer.decode(output[input_len:], skip_special_tokens=True))
    return texts


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="runs/qwen_sweep/report_data.json")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lora-scale", type=float, default=0.05)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    K = args.K

    print(f"Loading model on {device}...", flush=True)
    env = BaseModelEnv(K=K, device=device, lora_scale=args.lora_scale)

    if device == "cuda":
        torch.cuda.empty_cache()

    data = {"prompts": PROMPTS, "K": K, "lora_scale": args.lora_scale}

    # === Baseline ===
    print("Generating baseline...", flush=True)
    data["baseline"] = generate_multi(env, None, PROMPTS)

    # === Sweep each direction ===
    print("Sweeping individual directions...", flush=True)
    data["directions"] = {}
    for d in range(K):
        print(f"  direction {d}...", flush=True)
        results = {}
        for strength in STRENGTHS:
            action = torch.zeros(K, device=device)
            action[d] = strength
            probes = env.compute_probes(action)
            texts = generate_multi(env, action, PROMPTS)
            label = "high" if strength > 0 else "low"
            results[label] = {
                "strength": strength,
                "texts": texts,
                "probes": {
                    "KL": probes[0].item(),
                    "KL_var": probes[1].item(),
                    "Ent": probes[2].item(),
                    "Coh": probes[3].item(),
                },
            }
            if device == "cuda":
                torch.cuda.empty_cache()
        data["directions"][str(d)] = results

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
