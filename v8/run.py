#!/usr/bin/env python3
"""Run all stages sequentially. Each stage skips if its output already exists."""

import subprocess, sys

stages = [
    ("stage1_extract.py", "Extract activations"),
    ("stage2_sae.py",     "Train SAE"),
    ("stage3_stats.py",   "Compute feature stats"),
    ("stage4_sweep.py",   "Gain sweep generation"),
    ("stage5_label.py",   "Label features + meta-index"),
    ("stage6_dreamer.py", "Dreamer GRPO training"),
    ("stage7_explore.py", "Feature exploration"),
]

for script, desc in stages:
    print(f"\n{'='*60}")
    print(f"STAGE: {desc} ({script})")
    print('='*60)
    r = subprocess.run([sys.executable, script], cwd="/home/liam/Documents/projects/dreamerml/v8")
    if r.returncode != 0:
        print(f"FAILED: {script} (exit {r.returncode})")
        sys.exit(r.returncode)

print(f"\n{'='*60}")
print("ALL STAGES COMPLETE")
print('='*60)
