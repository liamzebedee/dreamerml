#!/usr/bin/env python
"""
Critical ablation: Is best-of-N improvement from perturbation or just rejection sampling?

Compare:
  1. Single base generation (no selection)
  2. Best-of-N base generations (rejection sampling, NO perturbation)
  3. Best-of-N perturbed generations (CEM with LoRA)
  4. Best-of-N perturbed generations + different base each time

If (2) ≈ (3), then LoRA adds nothing — improvement is just selection.
If (3) >> (2), then LoRA genuinely helps by expanding the reachable output space.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json
from contextlib import contextmanager
from collections import Counter
torch.backends.cudnn.benchmark = True

DEV = "cuda"
DT = torch.float16
K = 4; RANK = 8; ALPHA = 0.5; L2_CLAMP = 2.0
N_DREAM = 50; T_BASE = 20; TEMP = 0.9; TOP_P = 0.95; BATCH = 64

# Load model
print("Loading model...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M", torch_dtype=DT).to(DEV)
model.eval()
for p in model.parameters(): p.requires_grad = False
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id
NL = model.config.num_layers; HID = model.config.hidden_size
print(f"Loaded.", flush=True)

# LoRA basis
def get_mod(name):
    m = model
    for p in name.split("."): m = getattr(m, p)
    return m

TARGETS = []
for l in range(NL):
    TARGETS.append(f"transformer.h.{l}.attn.attention.out_proj")
    TARGETS.append(f"transformer.h.{l}.mlp.c_proj")

DW = {}
for name in TARGETS:
    w = get_mod(name).weight; o, i = w.shape
    A = torch.randn(K, o, RANK, device=DEV, dtype=DT) * (2.0/(o*RANK))**0.5
    B = torch.randn(K, RANK, i, device=DEV, dtype=DT) * (2.0/(i*RANK))**0.5
    DW[name] = torch.bmm(A, B)

def clamp(a):
    a = a.clamp(-1, 1)
    if a.dim() == 1:
        n = a.norm(); return a * min(1, L2_CLAMP/n) if n > L2_CLAMP else a
    n = a.norm(dim=-1, keepdim=True); return torch.where(n > L2_CLAMP, a*L2_CLAMP/n, a)

@contextmanager
def perturb(actions):
    a = clamp(actions.to(DEV, DT)); hooks = []
    for name in TARGETS:
        dw = DW[name]; mod = get_mod(name)
        def make_hook(dw_ref):
            def fn(module, inp, out):
                x = inp[0]; b = x.shape[0]
                pdw = torch.einsum("bk,koi->boi", a[:b], dw_ref)
                return out + torch.einsum("boi,bsi->bso", pdw, x) * ALPHA
            return fn
        hooks.append(mod.register_forward_hook(make_hook(dw)))
    try: yield
    finally:
        for h in hooks: h.remove()

@torch.no_grad()
def gen(ids, n, temp=TEMP):
    mask = (ids != PAD).long()
    out = model.generate(ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n,
                         do_sample=True, temperature=temp, top_p=TOP_P, pad_token_id=PAD)
    return out[:, ids.shape[1]:]

ENTITIES = ["cat","dog","bird","fish","rabbit","bear","mouse","fox","turtle","frog",
            "princess","knight","wizard","dragon","fairy","queen","king","pirate",
            "tree","flower","river","mountain","castle","garden","forest","cave",
            "boat","lamp","key","crown","sword","shield","map","book"]

def make_task():
    ents = list(np.random.choice(ENTITIES, 3, replace=False))
    return f"Write a story that includes a {ents[0]}, a {ents[1]}, and a {ents[2]}.", ents

def task_reward(text, ents):
    tl = text.lower(); words = tl.split()
    if len(words) < 5: return 0.05
    found = sum(1 for e in ents if e in tl) / len(ents)
    ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    uniq = len(set(ng))/max(1,len(ng)) if ng else 0
    vdiv = len(set(words))/len(words); lbon = min(1.0, len(words)/50)
    rep = 0
    if ng:
        ngc = Counter(ng); mcf = ngc.most_common(1)[0][1]/len(ng)
        if mcf > 0.12: rep = 0.5
        if mcf > 0.25: rep = 1.0
    return float(np.clip(0.50*found + 0.20*uniq + 0.10*vdiv + 0.10*lbon + 0.10*(1-rep), 0, 1))

def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return e["input_ids"].to(DEV)

def rand_actions(B):
    a = torch.empty(B, K, device=DEV, dtype=DT)
    h = B//2
    a[:h] = torch.rand(h, K, device=DEV, dtype=DT)*2-1
    a[h:] = torch.tanh(torch.randn(B-h, K, device=DEV, dtype=DT)*0.5)
    return clamp(a)

# ============================================================
# ABLATION EXPERIMENT
# ============================================================
def run_ablation(n_tasks=100, N=128):
    """
    For each task, compare:
    1. Single sample (no selection)
    2. Best-of-N (same base, N continuations, NO perturbation)
    3. Best-of-N (same base, N continuations WITH perturbation)
    4. Best-of-N (fresh base each time, NO perturbation) — tests whether diversity comes from base variation
    """
    print(f"\n{'='*60}\nABLATION: Rejection sampling vs LoRA perturbation\n{n_tasks} tasks, N={N} candidates\n{'='*60}", flush=True)
    t0 = time.time()

    single_r = []          # 1 sample, no selection
    bon_base_r = []        # best of N, no perturbation, SAME base prefix
    bon_perturb_r = []     # best of N, WITH perturbation, same base prefix
    bon_freshbase_r = []   # best of N, no perturbation, FRESH base each time
    mean_base_r = []       # mean of N, no perturbation (expected value)
    mean_perturb_r = []    # mean of N, WITH perturbation

    for ti in range(n_tasks):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        # Generate a single base prefix (shared for conditions 1, 2, 3)
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # === Condition 1: Single sample ===
        single = gen(ctx, N_DREAM + 30)
        txt = tok.decode(torch.cat([base[0], single[0]]), skip_special_tokens=True)
        single_r.append(task_reward(txt, ents))

        # === Condition 2: Best-of-N, NO perturbation, same base ===
        cond2_rewards = []
        for start in range(0, N, BATCH):
            B = min(BATCH, N - start)
            ctx_exp = ctx.expand(B, -1).contiguous()
            cont = gen(ctx_exp, N_DREAM + 30)
            for b in range(B):
                txt = tok.decode(torch.cat([base[0], cont[b]]), skip_special_tokens=True)
                cond2_rewards.append(task_reward(txt, ents))
        cond2_rewards = cond2_rewards[:N]
        bon_base_r.append(max(cond2_rewards))
        mean_base_r.append(np.mean(cond2_rewards))

        # === Condition 3: Best-of-N, WITH perturbation, same base ===
        cond3_rewards = []
        for start in range(0, N, BATCH):
            B = min(BATCH, N - start)
            actions = rand_actions(B)
            ctx_exp = ctx.expand(B, -1).contiguous()
            with perturb(actions):
                dream = gen(ctx_exp, N_DREAM)
            dream_ctx = torch.cat([ctx_exp, dream], dim=-1)
            post = gen(dream_ctx, 30)
            for b in range(B):
                txt = tok.decode(torch.cat([base[0], dream[b], post[b]]), skip_special_tokens=True)
                cond3_rewards.append(task_reward(txt, ents))
        cond3_rewards = cond3_rewards[:N]
        bon_perturb_r.append(max(cond3_rewards))
        mean_perturb_r.append(np.mean(cond3_rewards))

        # === Condition 4: Best-of-N, NO perturbation, FRESH base each time ===
        cond4_rewards = []
        for start in range(0, N, BATCH):
            B = min(BATCH, N - start)
            pids_exp = pids.expand(B, -1).contiguous()
            full_cont = gen(pids_exp, T_BASE + N_DREAM + 30, temp=TEMP)
            for b in range(B):
                txt = tok.decode(full_cont[b], skip_special_tokens=True)
                cond4_rewards.append(task_reward(txt, ents))
        cond4_rewards = cond4_rewards[:N]
        bon_freshbase_r.append(max(cond4_rewards))

        if (ti+1) % 10 == 0:
            el = time.time()-t0
            print(f"  [{ti+1}/{n_tasks}] single={np.mean(single_r):.4f} "
                  f"BoN_base={np.mean(bon_base_r):.4f} BoN_perturb={np.mean(bon_perturb_r):.4f} "
                  f"BoN_fresh={np.mean(bon_freshbase_r):.4f} "
                  f"mean_base={np.mean(mean_base_r):.4f} mean_pert={np.mean(mean_perturb_r):.4f} "
                  f"time={el:.0f}s", flush=True)

    print(f"\n{'='*60}\nRESULTS\n{'='*60}", flush=True)
    print(f"  Single sample (no selection):      {np.mean(single_r):.4f} ± {np.std(single_r):.4f}", flush=True)
    print(f"  Mean-of-{N} (no perturbation):     {np.mean(mean_base_r):.4f} ± {np.std(mean_base_r):.4f}", flush=True)
    print(f"  Mean-of-{N} (with perturbation):   {np.mean(mean_perturb_r):.4f} ± {np.std(mean_perturb_r):.4f}", flush=True)
    print(f"  Best-of-{N} same base, no pert:    {np.mean(bon_base_r):.4f} ± {np.std(bon_base_r):.4f}", flush=True)
    print(f"  Best-of-{N} same base, WITH pert:  {np.mean(bon_perturb_r):.4f} ± {np.std(bon_perturb_r):.4f}", flush=True)
    print(f"  Best-of-{N} fresh base, no pert:   {np.mean(bon_freshbase_r):.4f} ± {np.std(bon_freshbase_r):.4f}", flush=True)
    print(f"", flush=True)

    # Key comparison
    delta_pert = np.mean(bon_perturb_r) - np.mean(bon_base_r)
    print(f"  Δ(LoRA perturbation effect):  {delta_pert:+.4f}", flush=True)
    print(f"  Δ(Selection effect):          {np.mean(bon_base_r) - np.mean(single_r):+.4f}", flush=True)
    print(f"  Δ(Mean perturbation effect):  {np.mean(mean_perturb_r) - np.mean(mean_base_r):+.4f}", flush=True)

    from scipy import stats
    t, p = stats.ttest_rel(bon_perturb_r, bon_base_r)
    print(f"  Paired t-test BoN_perturb vs BoN_base: t={t:.3f}, p={p:.4f}", flush=True)
    t2, p2 = stats.ttest_rel(mean_perturb_r, mean_base_r)
    print(f"  Paired t-test mean_perturb vs mean_base: t={t2:.3f}, p={p2:.4f}", flush=True)

    print(f"\n  INTERPRETATION:", flush=True)
    if delta_pert > 0.02 and p < 0.05:
        print(f"  *** LoRA perturbation provides SIGNIFICANT benefit beyond rejection sampling ***", flush=True)
    elif delta_pert > 0.01:
        print(f"  LoRA perturbation provides marginal benefit (p={p:.4f})", flush=True)
    else:
        print(f"  LoRA perturbation provides NO benefit — improvement was just rejection sampling", flush=True)

    print(f"\n  Time: {time.time()-t0:.0f}s", flush=True)

    return {
        "single": np.mean(single_r), "mean_base": np.mean(mean_base_r),
        "mean_perturb": np.mean(mean_perturb_r),
        "bon_base": np.mean(bon_base_r), "bon_perturb": np.mean(bon_perturb_r),
        "bon_fresh": np.mean(bon_freshbase_r),
        "delta_pert": delta_pert, "p_value": p,
    }

if __name__ == "__main__":
    results = run_ablation(n_tasks=100, N=128)
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
