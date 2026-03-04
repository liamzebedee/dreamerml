"""Generate 16 sweep examples per direction for blog post."""
import json
import torch
from env import BaseModelEnv

PROMPTS = [
    "Once upon a time, there was a",
    "The little dog ran to the",
    "She looked at the big red",
    "He was very happy because",
    "The children played in the",
    "One day, a small boy found a",
    "The old woman sat by the",
    "It was a dark and stormy",
    "The princess looked out the window and",
    "Every morning, Tom would wake up and",
    "The two friends walked through the",
    "She held the tiny",
    "When the sun went down,",
    "The boy said to his mother,",
    "There was something strange about the",
    "He had never seen anything so",
]

DIRECTIONS = {
    "abstraction": 0,
    "complexity": 1,
    "confidence": 3,
    "vocabulary": 5,
}

def main():
    torch.manual_seed(42)
    env = BaseModelEnv(K=16, device="cpu", lora_scale=0.05)

    results = {}
    for name, dir_idx in DIRECTIONS.items():
        results[name] = {}
        for strength, label in [(-3.0, "low"), (3.0, "high")]:
            action = torch.zeros(16)
            action[dir_idx] = strength
            gens = env.generate(action, prompts=PROMPTS, max_new_tokens=40, temperature=0.7)
            results[name][label] = [
                {"prompt": p, "completion": g.strip()}
                for p, g in zip(PROMPTS, gens)
            ]
            print(f"Done: {name} {label}")

    with open("examples.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved examples.json")

if __name__ == "__main__":
    main()
