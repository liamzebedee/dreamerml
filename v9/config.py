"""Shared config for all stages."""
import os, torch
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "gpt2"  # GPT-2 small (124M params, 12 layers, 768 hidden)
TARGET_LAYER = 6      # mid-network where features are most interesting

# SAE
SAE_DICT_SIZE = 4096
SAE_SPARSITY_COEFF = 3e-3
SAE_LR = 3e-4
SAE_EPOCHS = 8
SAE_BATCH = 2048

# Extraction
ACT_NUM_SAMPLES = 40000
ACT_SEQ_LEN = 128
ACT_BATCH = 64

# Gain sweep
SWEEP_GAINS = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]
SWEEP_TOP_FEATURES = 64
GEN_MAX_TOKENS = 100

# Paths
BASE = Path("/home/liam/Documents/projects/dreamerml/v9")
OUT_DIR = BASE / "artifacts"
ACT_CACHE_DIR = BASE / "act_cache"
ACT_CACHE_FILE = ACT_CACHE_DIR / "acts_layer6.bin"
SHAPE_FILE = ACT_CACHE_DIR / "shape.json"
TOKEN_MAP_FILE = ACT_CACHE_DIR / "token_counts.json"
SAE_FILE = OUT_DIR / "sae.pt"
STATS_FILE = OUT_DIR / "feature_stats.json"
META_INDEX_FILE = OUT_DIR / "meta_index.txt"
