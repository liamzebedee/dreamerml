#!/usr/bin/env python3
"""Stage 1: Extract residual activations to disk as fp16 binary."""

import json, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from config import *

def main():
    if ACT_CACHE_FILE.exists() and SHAPE_FILE.exists():
        meta = json.loads(SHAPE_FILE.read_text())
        print(f"SKIP: Activations already cached ({meta['num_tokens']} tokens)")
        return

    ACT_CACHE_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model.eval()
    print(f"Model: hidden={model.config.hidden_size}, layers={model.config.num_layers}")

    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    texts = [s["text"] for s in dataset.select(range(min(ACT_NUM_SAMPLES, len(dataset))))]

    captured = {}
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[0] = h.detach()

    hook = model.transformer.h[TARGET_LAYER].register_forward_hook(hook_fn)
    t0 = time.time()
    total_tokens = 0
    token_counts = []

    with open(ACT_CACHE_FILE, "wb") as f:
        for i in range(0, len(texts), ACT_BATCH):
            batch_texts = texts[i:i+ACT_BATCH]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=ACT_SEQ_LEN).to(DEVICE)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
                model(**enc)

            mask = enc["attention_mask"]
            h = captured[0].float()
            for b in range(h.shape[0]):
                valid = mask[b].sum().item()
                f.write(h[b, :valid].half().cpu().numpy().tobytes())
                total_tokens += valid
                token_counts.append(valid)

            if (i // ACT_BATCH) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {total_tokens} tokens, {total_tokens/max(elapsed,1):.0f} tok/s")

    hook.remove()

    SHAPE_FILE.write_text(json.dumps({
        "hidden_size": model.config.hidden_size,
        "num_tokens": total_tokens,
        "layer": TARGET_LAYER
    }))
    TOKEN_MAP_FILE.write_text(json.dumps(token_counts))

    elapsed = time.time() - t0
    print(f"Done: {total_tokens} tokens in {elapsed:.1f}s ({total_tokens/elapsed:.0f} tok/s)")
    print(f"Cache: {ACT_CACHE_FILE} ({ACT_CACHE_FILE.stat().st_size/1e6:.0f}MB)")

if __name__ == "__main__":
    main()
