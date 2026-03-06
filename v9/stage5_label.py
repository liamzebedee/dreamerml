#!/usr/bin/env python3
"""Stage 5: Label features with Claude Haiku + build meta-index."""

import os, json, subprocess, sys
from config import *

LOG = OUT_DIR / "label_log.txt"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

def main():
    stats = json.loads(STATS_FILE.read_text())

    to_label = []
    already = 0
    for s in stats:
        p = OUT_DIR / f"feature_{s['feature_idx']:04d}.txt"
        if not p.exists():
            continue
        content = p.read_text()
        if content.startswith("LABEL:"):
            already += 1
        else:
            to_label.append((s["feature_idx"], p))

    log(f"Features: {already} labelled, {len(to_label)} remaining")
    if not to_label:
        log("All labelled, building meta-index...")
        build_meta_index(stats)
        return

    env = {k: v for k, v in os.environ.items() if "CLAUDE" not in k.upper()}
    env["PATH"] = os.environ["PATH"]
    env["HOME"] = os.environ["HOME"]

    for i, (feat_idx, artifact_path) in enumerate(to_label):
        content = artifact_path.read_text()

        prompt = f"""You are analyzing activation features from a sparse autoencoder trained on GPT-2 (124M params, layer 6 of 12).

This model was trained on diverse web text (WebText), so features may correspond to topics, writing styles, syntactic patterns, sentiment, reasoning patterns, entity types, or any other linguistic phenomenon.

Below is data for one feature. The gain sweeps show how generation changes when the feature is amplified (positive gain) or suppressed (negative gain). HIGH-ACTIVATION EXAMPLES show real text where this feature fires strongly.

{content[:5000]}

Provide:
1. SHORT NAME (2-5 words, lowercase, slash-separated category like "topic/science", "style/formal", "syntax/lists", "sentiment/negative", "entity/person-names", "reasoning/causal")
2. BRIEF DESCRIPTION (1-2 sentences explaining what this feature detects/controls)

If no clear pattern, use shortname "no-idea/1".

Output EXACTLY:
SHORTNAME: <name>
DESCRIPTION: <description>"""

        try:
            result = subprocess.run(
                ["claude", "--print", "--model", "claude-haiku-4-5-20251001", "-p", prompt],
                capture_output=True, text=True, timeout=60, env=env,
                stdin=subprocess.DEVNULL
            )
            response = result.stdout.strip()

            if result.returncode != 0:
                log(f"  claude error (rc={result.returncode}): {result.stderr[:200]}")

            shortname = "no-idea/1"
            description = "Failed to parse"

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("SHORTNAME:"):
                    shortname = line.split(":", 1)[1].strip()
                elif line.startswith("DESCRIPTION:"):
                    description = line.split(":", 1)[1].strip()

            with open(artifact_path, "r+") as f:
                old = f.read()
                f.seek(0)
                f.write(f"LABEL: {shortname}\nDESCRIPTION: {description}\n\n{old}")

            log(f"  [{i+1}/{len(to_label)}] Feature {feat_idx}: {shortname}")

        except Exception as e:
            log(f"  [{i+1}/{len(to_label)}] Feature {feat_idx}: ERROR - {e}")

    build_meta_index(stats)

def build_meta_index(stats):
    labels = []
    for s in stats:
        p = OUT_DIR / f"feature_{s['feature_idx']:04d}.txt"
        if not p.exists():
            continue
        content = p.read_text()
        shortname = "unlabelled"
        description = ""
        for line in content.split("\n")[:3]:
            if line.startswith("LABEL:"):
                shortname = line.split(":", 1)[1].strip()
            elif line.startswith("DESCRIPTION:"):
                description = line.split(":", 1)[1].strip()
        labels.append({"feature_idx": s["feature_idx"], "shortname": shortname,
                       "description": description, "frequency": s["frequency"]})

    lines = ["DREAMER FEATURE META-INDEX (GPT-2, layer 6)", "=" * 80, ""]
    lines.append(f"Total: {len(labels)}")
    interp = sum(1 for l in labels if not l["shortname"].startswith(("no-idea", "error", "unlabelled")))
    lines.append(f"Interpretable: {interp}")
    lines.append("")
    lines.append(f"{'IDX':>6}  {'FREQ':>6}  {'SHORTNAME':<35}  DESCRIPTION")
    lines.append("-" * 100)
    for l in sorted(labels, key=lambda x: x["feature_idx"]):
        lines.append(f"{l['feature_idx']:>6}  {l['frequency']:>6.3f}  "
                     f"{l['shortname']:<35}  {l['description'][:80]}")

    META_INDEX_FILE.write_text("\n".join(lines))
    log(f"Meta-index written: {META_INDEX_FILE} ({interp}/{len(labels)} interpretable)")

if __name__ == "__main__":
    main()
