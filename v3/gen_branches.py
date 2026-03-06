"""Generate deterministic branching examples for the creative writing post.

For each story prompt, show:
1. Base continuation (no perturbation)
2. Continuations with each labeled perturbation direction
All with fixed seeds for reproducibility.
"""

import json
import torch
from env import DreamEnv
from policy import PerturbationPolicy
from state import extract_state, STATE_DIM

# Direction labels from comprehensive sweep analysis
DIRECTIONS = {
    2:  ("Grandeur",      +0.8, "palaces, kingdoms, grand structures, surprises"),
    4:  ("Exotic",        +0.8, "deserts, cities, distant lands, mystery"),
    8:  ("Warmth",        -0.8, "sharing, friendship, emotional connection"),
    9:  ("Specificity",   -0.8, "named characters, ages, concrete plans"),
    12: ("Nature",        +0.8, "leaves, grass, rocks, storms, outdoors"),
    13: ("Vulnerability", +0.8, "small creatures, hurt/scared, dialogue"),
    3:  ("Social",        -0.8, "relationships, family, community bonds"),
    7:  ("Village",       +0.8, "village life, nature beauty, community"),
    15: ("Melancholy",    -0.8, "sadness, loneliness, bravery in hardship"),
    14: ("Domestic",      +0.8, "houses, yards, embarrassment, daily life"),
    0:  ("Playground",    +0.8, "named kids in parks, playful scenes"),
    1:  ("Adventure",     +0.8, "action, weapons, exploration, quests"),
}

STORY_PROMPTS = [
    "Once upon a time there was a little",
    "The brave knight walked into the dark",
    "A tiny bird sat on the",
    "Deep in the forest there lived a",
    "The old wizard opened his book and",
    "One sunny morning the cat decided to",
    "A young prince stood at the edge of",
    "The little girl found a strange key",
    "Under the bridge there was a small",
    "When the rain stopped the children ran",
    "The king looked out from his tower at",
    "A mouse crept quietly into the big",
]

SEEDS = [42]


def gen_with_seed(env, action, prompt, seed, max_tokens=60):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    texts = env.generate(action, prompts=[prompt], max_new_tokens=max_tokens)
    return texts[0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = DreamEnv(model_name="roneneldan/TinyStories-1M", K=16, device=device)

    # Load trained policy + LoRA
    ckpt = torch.load("runs/v3/policy.pt", map_location=device, weights_only=False)
    policy = PerturbationPolicy(state_dim=STATE_DIM, K=16).to(device)
    policy.load_state_dict(ckpt["policy"])
    env.lora.load_state_dict(ckpt["lora"])
    policy.eval()

    results = {}

    for prompt in STORY_PROMPTS:
        results[prompt] = {"base": [], "directions": {}}

        # Base (no perturbation)
        for seed in SEEDS:
            text = gen_with_seed(env, None, prompt, seed)
            results[prompt]["base"].append({"seed": seed, "text": text})

        # Each direction
        for dk, (label, strength, desc) in DIRECTIONS.items():
            action = torch.zeros(16, device=device)
            action[dk] = strength
            dir_results = []
            for seed in SEEDS:
                text = gen_with_seed(env, action, prompt, seed)
                dir_results.append({"seed": seed, "text": text})
            results[prompt]["directions"][f"d{dk:02d}"] = {
                "label": label,
                "strength": strength,
                "desc": desc,
                "texts": dir_results,
            }

        # Combined directions (learned policy action)
        with torch.inference_mode():
            inp = env.tokenizer(
                prompt, return_tensors="pt", max_length=32, truncation=True,
            ).to(device)
            out = env.model(**inp, output_hidden_states=True)
            state = extract_state(
                out.logits, out.hidden_states,
                inp["attention_mask"],
            )
        state = state.clone().detach()
        with torch.no_grad():
            action = policy.deterministic(state)
        combined = []
        for seed in SEEDS:
            text = gen_with_seed(env, action, prompt, seed)
            combined.append({"seed": seed, "text": text})
        results[prompt]["combined"] = combined

    # Print compact grid for the blog post
    # Header
    dir_keys = list(DIRECTIONS.keys())
    dir_labels = [DIRECTIONS[k][0] for k in dir_keys]

    for prompt in STORY_PROMPTS:
        print(f"\n{'='*70}")
        print(f"PROMPT: \"{prompt}\"")
        print(f"{'='*70}")

        base_text = results[prompt]['base'][0]['text'].strip()
        # Truncate to ~first sentence or 120 chars
        trunc = base_text[:120]
        print(f"\n  BASE: {trunc}")

        for dk in dir_keys:
            dk_key = f"d{dk:02d}"
            dinfo = results[prompt]["directions"][dk_key]
            t = dinfo['texts'][0]['text'].strip()[:120]
            print(f"  {dinfo['label']:13s}: {t}")

        ct = results[prompt]['combined'][0]['text'].strip()[:120]
        print(f"  {'COMBINED':13s}: {ct}")

    # Save full results
    with open("runs/v3/branches.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved to runs/v3/branches.json")


if __name__ == "__main__":
    main()
