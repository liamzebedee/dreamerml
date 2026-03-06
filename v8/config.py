"""Shared config for all stages."""
import os, torch
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "roneneldan/TinyStories-33M"
TARGET_LAYER = 2

# SAE
SAE_DICT_SIZE = 2048
SAE_SPARSITY_COEFF = 5e-3
SAE_LR = 3e-4
SAE_EPOCHS = 5
SAE_BATCH = 2048

# Extraction
ACT_NUM_SAMPLES = 20000
ACT_SEQ_LEN = 128
ACT_BATCH = 64

# Gain sweep
SWEEP_GAINS = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
SWEEP_TOP_FEATURES = 50
GEN_MAX_TOKENS = 80

# Paths
BASE = Path("/home/liam/Documents/projects/dreamerml/v8")
OUT_DIR = BASE / "artifacts"
ACT_CACHE_DIR = BASE / "act_cache"
ACT_CACHE_FILE = ACT_CACHE_DIR / "acts_layer2.bin"
SHAPE_FILE = ACT_CACHE_DIR / "shape.json"
TOKEN_MAP_FILE = ACT_CACHE_DIR / "token_counts.json"
SAE_FILE = OUT_DIR / "sae.pt"
STATS_FILE = OUT_DIR / "feature_stats.json"
META_INDEX_FILE = OUT_DIR / "meta_index.txt"
