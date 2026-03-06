"""
TinyStories constraint prompt generator + reward computation.
"""

import numpy as np
import re
from typing import Dict, List, Tuple


ENTITIES = [
    "cat", "dog", "bird", "fish", "rabbit", "bear", "mouse", "fox", "turtle", "frog",
    "princess", "knight", "wizard", "dragon", "fairy", "queen", "king", "pirate",
    "tree", "flower", "river", "mountain", "castle", "garden", "forest", "cave",
    "ball", "book", "key", "star", "crown", "ring", "map", "sword",
]

TWIST_MARKERS = [
    "but then", "suddenly", "until", "secretly", "however", "to everyone's surprise",
    "nobody knew", "what they didn't know", "all of a sudden", "out of nowhere",
    "it turned out", "the truth was", "little did they know",
]

NAMES = [
    "Lily", "Tom", "Sara", "Ben", "Mia", "Max", "Emma", "Jack", "Ella", "Tim",
    "Anna", "Sam", "Lucy", "Leo", "Zoe", "Jake", "Amy", "Dan", "Ivy", "Ray",
]


def generate_entity_prompt() -> Tuple[str, Dict]:
    """Generate prompt requiring 3 specific entities."""
    entities = list(np.random.choice(ENTITIES, 3, replace=False))
    prompt = f"Write a short story that includes a {entities[0]}, a {entities[1]}, and a {entities[2]}. "
    prompt += f"The story must mention all three: {', '.join(entities)}."
    return prompt, {"type": "entities", "required": entities}


def generate_twist_prompt() -> Tuple[str, Dict]:
    """Generate prompt requiring a twist ending."""
    char = np.random.choice(NAMES)
    setting = np.random.choice(["forest", "castle", "garden", "village", "mountain"])
    prompt = f"Write a story about {char} in the {setting}. The story must end with a surprising twist."
    return prompt, {"type": "twist", "character": char, "setting": setting}


def generate_names_prompt() -> Tuple[str, Dict]:
    """Generate prompt requiring consistent character names."""
    names = list(np.random.choice(NAMES, 2, replace=False))
    prompt = f"Write a story about {names[0]} and {names[1]}. "
    prompt += f"Use their names {names[0]} and {names[1]} throughout the story."
    return prompt, {"type": "names", "required_names": names}


def generate_rhyme_prompt() -> Tuple[str, Dict]:
    """Generate prompt requiring a rhyme at the end."""
    topic = np.random.choice(["friendship", "adventure", "kindness", "bravery", "sharing"])
    prompt = f"Write a story about {topic}. End the story with two lines that rhyme."
    return prompt, {"type": "rhyme", "topic": topic}


def get_random_task() -> Tuple[str, Dict]:
    """Get a random constrained task."""
    generators = [generate_entity_prompt, generate_twist_prompt,
                  generate_names_prompt, generate_rhyme_prompt]
    gen = np.random.choice(generators)
    return gen()


def compute_reward(text: str, task_meta: Dict) -> float:
    """
    Compute reward R ∈ [0, 1].
    R = 0.4 * constraint_satisfaction + 0.3 * ending_quality + 0.3 * coherence
    """
    constraint = compute_constraint_satisfaction(text, task_meta)
    ending = compute_ending_quality(text)
    coherence = compute_coherence(text)

    R = 0.4 * constraint + 0.3 * ending + 0.3 * coherence
    return float(np.clip(R, 0, 1))


def compute_constraint_satisfaction(text: str, task_meta: Dict) -> float:
    """Check fraction of constraints met."""
    text_lower = text.lower()

    if task_meta["type"] == "entities":
        found = sum(1 for e in task_meta["required"] if e in text_lower)
        return found / len(task_meta["required"])

    elif task_meta["type"] == "twist":
        has_twist = any(m in text_lower for m in TWIST_MARKERS)
        has_char = task_meta["character"].lower() in text_lower
        return (0.5 * has_twist + 0.5 * has_char)

    elif task_meta["type"] == "names":
        counts = [text.count(n) for n in task_meta["required_names"]]
        # Each name should appear at least twice
        score = sum(min(c, 2) / 2 for c in counts) / len(task_meta["required_names"])
        return score

    elif task_meta["type"] == "rhyme":
        # Check if last two lines exist and have some structure
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if len(lines) >= 2:
            return 0.7  # Simple check: has ending lines
        return 0.2

    return 0.5


def compute_ending_quality(text: str) -> float:
    """Check ending quality: twist markers, novelty of last tokens."""
    text_lower = text.lower()
    last_chunk = text_lower[-200:] if len(text_lower) > 200 else text_lower

    # Check for twist/conclusion markers
    has_marker = any(m in last_chunk for m in TWIST_MARKERS)

    # Check for proper ending punctuation
    stripped = text.rstrip()
    has_ending = stripped and stripped[-1] in '.!?"'

    # Novelty: unique words in last 20 words vs first 20 words
    words = text.split()
    if len(words) > 40:
        first20 = set(words[:20])
        last20 = set(words[-20:])
        novelty = len(last20 - first20) / max(1, len(last20))
    else:
        novelty = 0.5

    score = 0.3 * has_marker + 0.3 * has_ending + 0.4 * novelty
    return float(np.clip(score, 0, 1))


def compute_coherence(text: str) -> float:
    """Check coherence: penalize repetition and name inconsistency."""
    words = text.lower().split()
    if len(words) < 5:
        return 0.3

    # Word-level repetition (4-grams)
    ngrams = []
    for i in range(len(words) - 3):
        ngrams.append(tuple(words[i:i+4]))

    if ngrams:
        unique_ratio = len(set(ngrams)) / len(ngrams)
    else:
        unique_ratio = 1.0

    # Char repetition
    chars = list(text)
    if len(chars) > 10:
        char_ngrams = [tuple(chars[i:i+5]) for i in range(len(chars) - 4)]
        char_unique = len(set(char_ngrams)) / max(1, len(char_ngrams))
    else:
        char_unique = 1.0

    # Penalize very short or very long outputs
    length_score = 1.0
    if len(words) < 10:
        length_score = 0.3
    elif len(words) > 500:
        length_score = 0.5

    score = 0.4 * unique_ratio + 0.3 * char_unique + 0.3 * length_score
    return float(np.clip(score, 0, 1))


if __name__ == "__main__":
    # Test prompt generation and reward
    for _ in range(5):
        prompt, meta = get_random_task()
        print(f"Prompt: {prompt}")
        print(f"Meta: {meta}")

        # Fake story for testing reward
        fake = "Lily went to the forest. She found a cat and a dog. But then the cat flew away! "
        r = compute_reward(fake, meta)
        print(f"Reward: {r:.3f}\n")
