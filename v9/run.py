#!/usr/bin/env python3
"""Run all stages sequentially. Each stage skips if its output already exists."""

import subprocess, sys
from pathlib import Path

CWD = str(Path(__file__).parent)

stages = [
    ("stage1_extract.py", "Extract activations from GPT-2"),
    ("stage2_sae.py",     "Train SAE (4096 features)"),
    ("stage3_stats.py",   "Compute feature stats"),
    ("stage4_sweep.py",   "Gain sweep generation"),
    ("stage5_label.py",   "Label features + meta-index"),
    ("stage6_dreamer.py", "Dreamer GRPO training"),
]

for script, desc in stages:
    print(f"\n{'='*60}")
    print(f"STAGE: {desc} ({script})")
    print('='*60)
    r = subprocess.run([sys.executable, script], cwd=CWD)
    if r.returncode != 0:
        print(f"FAILED: {script} (exit {r.returncode})")
        sys.exit(r.returncode)

print(f"\n{'='*60}")
print("ALL STAGES COMPLETE")
print('='*60)
