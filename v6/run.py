#!/usr/bin/env python
"""
DreamerML v6 - Full pipeline: collect transitions, train F/I, evaluate.
Maximally optimized for speed on RTX 3090.
"""

import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
import time
import os
import json
from contextlib import contextmanager
from torch.utils.data import DataLoader, TensorDataset

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

DEV = "cuda"
DTYPE = torch.float16

# === HYPERPARAMS ===
K = 32
ALPHA = 0.06
L2_CLAMP = 2.0
N_DREAM = 40
T_BASE = 24
TEMP = 0.9
TOP_P = 0.95
STATE_DIM = 64
CTX_DIM = 64
BATCH = 128
N_TRANSITIONS = 20000
M_PROPOSALS = 64

# === MODEL LOADING ===
print("Loading model...", flush=True)
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M", torch_dtype=DTYPE).to(DEV)
model.eval()
for p in model.parameters():
    p.requires_grad = False
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id
NUM_LAYERS = model.config.num_layers
HIDDEN = model.config.hidden_size
print(f"Loaded in {time.time()-t0:.1f}s: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, "
      f"{NUM_LAYERS} layers, hidden={HIDDEN}", flush=True)

# === LORA BASIS ===
def get_targets():
    s, e = int(0.3*NUM_LAYERS), int(0.7*NUM_LAYERS)
    t = []
    for l in range(s, e+1):
        t.append(f"transformer.h.{l}.attn.attention.out_proj")
        t.append(f"transformer.h.{l}.mlp.c_proj")
    return t

def get_mod(name):
    m = model
    for p in name.split("."):
        m = getattr(m, p)
    return m

TARGETS = get_targets()
# Precompute ΔW_k = A_k B_k for each target
DW = {}  # {name: (K, out, in)}
for name in TARGETS:
    w = get_mod(name).weight
    o, i = w.shape
    A = torch.randn(K, o, 1, device=DEV, dtype=DTYPE) * 0.02
    B = torch.randn(K, 1, i, device=DEV, dtype=DTYPE) * 0.02
    DW[name] = (A @ B).view(K, o, i)

def clamp(a):
    a = a.clamp(-1, 1)
    if a.dim() == 1:
        n = a.norm()
        return a * min(1, L2_CLAMP/n) if n > L2_CLAMP else a
    n = a.norm(dim=-1, keepdim=True)
    return torch.where(n > L2_CLAMP, a * L2_CLAMP / n, a)

# === PER-SAMPLE PERTURBATION HOOKS ===
@contextmanager
def perturb(actions):
    """Install hooks for per-sample LoRA perturbation. actions: (B, K)"""
    a = clamp(actions.to(DEV, DTYPE))
    hooks = []
    for name in TARGETS:
        dw = DW[name]
        mod = get_mod(name)
        def make_hook(dw_ref):
            def fn(module, inp, out):
                x = inp[0]
                b = x.shape[0]
                pdw = torch.einsum("bk,koi->boi", a[:b], dw_ref)
                return out + torch.einsum("boi,bsi->bso", pdw, x) * ALPHA
            return fn
        hooks.append(mod.register_forward_hook(make_hook(dw)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()

# Single action (no hooks, direct weight mod)
_orig = {}
def apply_action(a):
    a = clamp(a.to(DEV, DTYPE))
    for name in TARGETS:
        mod = get_mod(name)
        _orig[name] = mod.weight.data.clone()
        mod.weight.data += ALPHA * torch.einsum("k,koi->oi", a, DW[name])

def revert():
    for name, w in _orig.items():
        get_mod(name).weight.data.copy_(w)
    _orig.clear()

# === GENERATION ===
@torch.no_grad()
def gen(ids, n, temp=TEMP):
    """Batched generation with KV cache via HF generate."""
    mask = (ids != PAD).long()
    out = model.generate(ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n,
                         do_sample=True, temperature=temp, top_p=TOP_P, pad_token_id=PAD)
    return out[:, ids.shape[1]:]

# === STATE EXTRACTION (batched) ===
PROBE = [int(0.25*NUM_LAYERS), int(0.5*NUM_LAYERS), int(0.75*NUM_LAYERS)]

# Running normalizer
_nm = torch.zeros(STATE_DIM, device=DEV)
_nv = torch.ones(STATE_DIM, device=DEV)
_nc = 0
_nfrozen = False

def update_norm(x):
    global _nm, _nv, _nc
    if _nfrozen: return
    m = 0.001
    bm, bv = x.mean(0), x.var(0, unbiased=False) + 1e-8
    if _nc == 0:
        _nm, _nv = bm, bv
    else:
        _nm = (1-m)*_nm + m*bm
        _nv = (1-m)*_nv + m*bv
    _nc += 1

def normalize(x):
    return (x - _nm) / (_nv.sqrt() + 1e-8)

@torch.no_grad()
def extract_states(ids):
    """(B, seq) -> (B, 64) state vectors."""
    B = ids.shape[0]
    mask = (ids != PAD).long()
    out = model(ids, attention_mask=mask, output_hidden_states=True, output_attentions=True)

    seq_lens = mask.sum(-1)
    last = (seq_lens - 1).clamp(min=0)
    bi = torch.arange(B, device=DEV)

    logits = out.logits[bi, last].float()  # (B, V)
    last_tok = ids[bi, last]

    probs = logits.softmax(-1)
    lp = logits.log_softmax(-1)
    ent = -(probs * lp).sum(-1)
    sp, _ = probs.sort(descending=True)
    sl, _ = logits.sort(descending=True)

    feats = [
        ent, sp[:,0], sp[:,:5].sum(-1), sp[:,:20].sum(-1),
        sl[:,0] - sl[:,1], logits.mean(-1), logits.std(-1),
        sp[:,:200].sum(-1), ent.exp(),
        probs[bi, last_tok],
    ]  # 10 dims

    # Hidden stats at probe layers (3 * 8 = 24 dims)
    for li in PROBE:
        h = out.hidden_states[li+1].float()  # (B, seq, H)
        hl = h[bi, last]  # (B, H)
        hm = h.mean(dim=(1,2))
        hv = h.var(dim=(1,2))
        ham = h.abs().mean(dim=(1,2))
        hav = h.abs().var(dim=(1,2))
        hlm = hl.mean(-1)
        hlv = hl.var(-1)
        # cosine sim last vs recent mean
        starts = (last - 15).clamp(min=0)
        # vectorized recent mean
        rm = torch.zeros_like(hl)
        for b in range(B):
            rm[b] = h[b, starts[b]:last[b]+1].mean(0)
        cs = TF.cosine_similarity(hl, rm, dim=-1)
        feats.extend([hm, hv, ham, hav, hlm, hlv, cs, hl.norm(dim=-1)])

    # Attention stats (3 * 6 = 18 dims)
    for li in PROBE:
        att = out.attentions[li].float()  # (B, H, S, S)
        la = att[bi, :, last]  # (B, heads, S)
        ae = -(la * (la + 1e-10).log()).sum(-1)  # (B, heads)
        feats.extend([ae.mean(-1), ae.std(-1)])
        mx = la.max(-1).values
        feats.extend([mx.mean(-1), mx.std(-1)])
        pos = torch.arange(att.shape[-1], device=DEV, dtype=torch.float32)
        ep = (la * pos).sum(-1)
        feats.extend([ep.mean(-1), ep.std(-1)])

    # Trajectory deltas: zero (12 dims)
    feats.append(torch.zeros(B, 12, device=DEV))

    # Stack
    parts = []
    for f in feats:
        if f.dim() == 1:
            parts.append(f.unsqueeze(-1))
        else:
            parts.append(f)
    s = torch.cat(parts, dim=-1)  # (B, 64)

    update_norm(s)
    s = normalize(s)
    return torch.where(torch.isfinite(s), s, torch.zeros_like(s))

@torch.no_grad()
def extract_ctx(ids):
    """(B, seq) -> (B, HIDDEN) context embeddings."""
    mask = (ids != PAD).long()
    out = model(ids, attention_mask=mask, output_hidden_states=True)
    B = ids.shape[0]
    last = (mask.sum(-1) - 1).clamp(min=0)
    return out.hidden_states[-1][torch.arange(B, device=DEV), last]

# Context projector
ctx_proj = nn.Linear(HIDDEN, CTX_DIM).to(DEV).half()

# === PROMPTS ===
PROMPTS = [
    "Once upon a time, there was a", "One day, a little girl named",
    "There was a big dog who", "The cat and the bird were",
    "A brave knight went to the", "In a small village, there lived",
    "The sun was shining and the", "A funny monkey found a",
    "The princess looked out the window", "Tom and his friend went to",
    "The little bunny hopped into the", "A magic fish swam in the",
    "The old tree had a secret", "One morning, the baby bird",
    "Sara loved to play with her", "The farmer had a big red",
    "In the garden, there was a", "The boy found a shiny",
    "Lily and her mom went to the", "A tiny mouse lived in a",
    "Write a scary story about a cat.", "Write a funny story about a dog.",
    "Write a happy story about a girl.", "Write a sad story about a boy.",
    "Write a story about a brave knight.", "Write a story about a princess.",
    "Write a story about a wizard.", "Write a story about a bird.",
    "Write a story about a mouse.", "Write a story about a bear.",
]

def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return e["input_ids"].to(DEV)

def rand_actions(B):
    a = torch.empty(B, K, device=DEV, dtype=DTYPE)
    h = B // 2
    a[:h] = torch.rand(h, K, device=DEV, dtype=DTYPE) * 2 - 1
    a[h:] = torch.tanh(torch.randn(B-h, K, device=DEV, dtype=DTYPE) * 0.5)
    return clamp(a)

def check_quality(tokens):
    """(B, N) -> (B,) quality flags."""
    B = tokens.shape[0]
    q = torch.ones(B)
    for b in range(B):
        t = tokens[b].tolist()
        if len(t) >= 8:
            ng = set()
            rep = 0
            for i in range(len(t)-3):
                g = tuple(t[i:i+4])
                if g in ng: rep += 1
                ng.add(g)
            if rep / max(1, len(t)-3) > 0.4:
                q[b] = 0
    return q

# ============================================================
# PHASE 1: COLLECT TRANSITIONS
# ============================================================
def collect(n=N_TRANSITIONS, bs=BATCH):
    print(f"\n{'='*60}\nPHASE 1: Collecting {n} transitions (batch={bs})\n{'='*60}", flush=True)
    all_s, all_a, all_ns, all_q, all_c = [], [], [], [], []
    t0 = time.time()

    for i in range(0, n, bs):
        B = min(bs, n - i)
        prompts = [np.random.choice(PROMPTS) for _ in range(B)]
        pids = encode_batch(prompts)

        # Context embeddings
        c = ctx_proj(extract_ctx(pids).half())  # (B, 64)

        # Base generation (unperturbed, KV-cached)
        base = gen(pids, T_BASE)
        full = torch.cat([pids, base], dim=-1)

        # Pre-dream states
        st = extract_states(full)

        # Random actions + perturbed dream generation
        actions = rand_actions(B)
        with perturb(actions):
            dream = gen(full, N_DREAM)

        # Post-dream states (unperturbed)
        dfull = torch.cat([full, dream], dim=-1)
        sn = extract_states(dfull)

        # Quality
        q = check_quality(dream)

        all_s.append(st.float().cpu())
        all_a.append(actions.float().cpu())
        all_ns.append(sn.float().cpu())
        all_q.append(q)
        all_c.append(c.float().detach().cpu())

        if (i + bs) % (bs * 4) < bs:
            elapsed = time.time() - t0
            done = i + B
            rate = done / elapsed
            print(f"  [{done}/{n}] {rate:.0f}/s quality={sum(qq.sum() for qq in all_q)/done:.0%} "
                  f"ETA={max(0,(n-done)/rate):.0f}s", flush=True)

    data = {k: torch.cat(v) for k, v in zip(
        ["states","actions","next_states","quality","contexts"],
        [all_s, all_a, all_ns, all_q, all_c]
    )}
    elapsed = time.time() - t0
    print(f"Done: {n} transitions in {elapsed:.1f}s ({n/elapsed:.0f}/s), "
          f"quality={data['quality'].mean():.0%}", flush=True)
    return data

# ============================================================
# PHASE 2: TRAIN FORWARD + INVERSE MODELS
# ============================================================
class FwdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + K + CTX_DIM, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, STATE_DIM))
    def forward(self, s, a, c):
        return s + self.net(torch.cat([s, a, c], -1))

class InvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM*2 + CTX_DIM, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, K), nn.Tanh())
    def forward(self, s, sn, c):
        return self.net(torch.cat([s, sn, c], -1))

def train_models(data, epochs=15, bs=4096, lr=3e-4):
    print(f"\n{'='*60}\nPHASE 2: Training dynamics models\n{'='*60}", flush=True)
    d = {k: v.to(DEV) for k, v in data.items()}
    N = len(d["states"])

    fwd = FwdModel().to(DEV)
    inv = InvModel().to(DEV)

    # Train forward
    opt_f = torch.optim.AdamW(fwd.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEV)
        total = 0; nb = 0
        for j in range(0, N, bs):
            idx = perm[j:j+bs]
            s, a, sn, q, c = d["states"][idx], d["actions"][idx], d["next_states"][idx], d["quality"][idx], d["contexts"][idx]
            pred = fwd(s, a, c)
            loss = (q.unsqueeze(-1) * (sn - pred)**2).mean()
            opt_f.zero_grad(); loss.backward(); opt_f.step()
            total += loss.item(); nb += 1
        print(f"  F epoch {ep+1}/{epochs}: loss={total/nb:.6f}", flush=True)
    print(f"  Forward model trained in {time.time()-t0:.1f}s", flush=True)

    # Train inverse (quality=1 only)
    mask = d["quality"] > 0.5
    ds, da, dns, dc = d["states"][mask], d["actions"][mask], d["next_states"][mask], d["contexts"][mask]
    Nq = len(ds)
    opt_i = torch.optim.AdamW(inv.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(Nq, device=DEV)
        total = 0; nb = 0
        for j in range(0, Nq, bs):
            idx = perm[j:j+bs]
            pred_a = inv(ds[idx], dns[idx], dc[idx])
            loss = ((da[idx] - pred_a)**2).mean()
            opt_i.zero_grad(); loss.backward(); opt_i.step()
            total += loss.item(); nb += 1
        print(f"  I epoch {ep+1}/{epochs}: loss={total/nb:.6f}", flush=True)
    print(f"  Inverse model trained in {time.time()-t0:.1f}s", flush=True)

    return fwd, inv

# ============================================================
# PHASE 3: PLANNING + EVALUATION
# ============================================================
def objective(s):
    """Planning objective J(s). s: (B, 64)"""
    if s.dim() == 1: s = s.unsqueeze(0)
    # novelty: high entropy, low top1, low rep_risk
    j_nov = s[:,0] + (-s[:,1]) + (-s[:,9])
    # coherence penalties
    pen_ent = ((-1.0 - s[:,0]).clamp(min=0)) * 2
    pen_rep = ((s[:,9] - 1.5).clamp(min=0)) * 2
    attn_std = (s[:,35].abs() + s[:,41].abs() + s[:,47].abs()) / 3
    pen_att = ((0.1 - attn_std).clamp(min=0)) * 2
    j_coh = -(pen_ent + pen_rep + pen_att)
    return (1.0 * j_nov + 1.5 * j_coh).squeeze()

@torch.no_grad()
def plan(fwd, inv, st, c):
    """Choose action via forward proposal + inverse solve."""
    st_exp = st.unsqueeze(0).expand(M_PROPOSALS, -1)
    c_exp = c.unsqueeze(0).expand(M_PROPOSALS, -1)

    ac = clamp(torch.tanh(torch.randn(M_PROPOSALS, K, device=DEV) * 0.35))
    sp = fwd(st_exp, ac, c_exp)
    scores = objective(sp)
    best = scores.argmax()
    s_star = sp[best:best+1]

    a_star = inv(st_exp[:1], s_star, c_exp[:1]).squeeze(0)

    # Refine
    sc = fwd(st_exp[:1], a_star.unsqueeze(0), c_exp[:1])
    if objective(sc) < scores[best]:
        ac2 = inv(st_exp[:1], sc + (s_star - sc), c_exp[:1]).squeeze(0)
        a_star = clamp(0.5 * a_star + 0.5 * ac2)

    return clamp(a_star)

def evaluate(fwd, inv, n=80):
    print(f"\n{'='*60}\nPHASE 3: Evaluation ({n} prompts)\n{'='*60}", flush=True)

    planned_r, random_r, nodream_r = [], [], []
    planned_degen, random_degen = 0, 0
    planned_texts, random_texts, nodream_texts = [], [], []

    t0 = time.time()

    for i in range(0, n, BATCH):
        B = min(BATCH, n - i)
        prompts = [np.random.choice(PROMPTS) for _ in range(B)]
        pids = encode_batch(prompts)

        c = ctx_proj(extract_ctx(pids).half())

        # Generate shared base
        base = gen(pids, 30)
        ctx = torch.cat([pids, base], dim=-1)
        st = extract_states(ctx)

        # --- No dream baseline ---
        nodream_gen = gen(ctx, 70)
        for b in range(B):
            txt = tok.decode(torch.cat([base[b], nodream_gen[b]]), skip_special_tokens=True)
            nodream_texts.append(txt)
            nodream_r.append(simple_reward(txt))

        # --- Planned dream ---
        plan_actions = torch.stack([plan(fwd, inv, st[b], c[b]) for b in range(B)])
        with perturb(plan_actions):
            pdream = gen(ctx, N_DREAM)
        pq = check_quality(pdream)
        planned_degen += (pq < 0.5).sum().item()

        pctx = torch.cat([ctx, pdream], dim=-1)
        ppost = gen(pctx, 30)
        for b in range(B):
            txt = tok.decode(torch.cat([base[b], pdream[b], ppost[b]]), skip_special_tokens=True)
            planned_texts.append(txt)
            planned_r.append(simple_reward(txt))

        # --- Random dream ---
        ractions = rand_actions(B)
        with perturb(ractions):
            rdream = gen(ctx, N_DREAM)
        rq = check_quality(rdream)
        random_degen += (rq < 0.5).sum().item()

        rctx = torch.cat([ctx, rdream], dim=-1)
        rpost = gen(rctx, 30)
        for b in range(B):
            txt = tok.decode(torch.cat([base[b], rdream[b], rpost[b]]), skip_special_tokens=True)
            random_texts.append(txt)
            random_r.append(simple_reward(txt))

    elapsed = time.time() - t0

    pr, rr, nr = np.mean(planned_r), np.mean(random_r), np.mean(nodream_r)

    results = {
        "planned_reward": float(pr),
        "random_reward": float(rr),
        "nodream_reward": float(nr),
        "planned_vs_nodream": float(pr - nr),
        "planned_vs_random": float(pr - rr),
        "planned_degen_rate": planned_degen / n,
        "random_degen_rate": random_degen / n,
        "planner_wins": bool(pr > rr),
        "dream_helps": bool(pr > nr),
        "eval_time": elapsed,
    }

    print(f"\n  No-dream reward:  {nr:.4f}", flush=True)
    print(f"  Random dream:     {rr:.4f} (degen={random_degen/n:.0%})", flush=True)
    print(f"  Planned dream:    {pr:.4f} (degen={planned_degen/n:.0%})", flush=True)
    print(f"  Plan vs none:     {pr-nr:+.4f}", flush=True)
    print(f"  Plan vs random:   {pr-rr:+.4f}", flush=True)
    print(f"  Eval time:        {elapsed:.1f}s", flush=True)

    # Print sample outputs
    print(f"\n--- Sample Outputs ---", flush=True)
    for j in range(min(5, n)):
        print(f"\n[Prompt {j+1}]", flush=True)
        print(f"  No dream:  {nodream_texts[j][:200]}", flush=True)
        print(f"  Planned:   {planned_texts[j][:200]}", flush=True)
        print(f"  Random:    {random_texts[j][:200]}", flush=True)

    return results

def simple_reward(text):
    """Quick reward: diversity + coherence."""
    words = text.lower().split()
    if len(words) < 5: return 0.1
    # unique 4-grams ratio
    ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    uniq = len(set(ng)) / max(1, len(ng)) if ng else 0
    # vocab diversity
    vdiv = len(set(words)) / len(words)
    # length bonus
    lbon = min(1.0, len(words) / 50)
    # ending quality
    end = 0.5 + 0.5 * (text.rstrip()[-1] in '.!?"' if text.rstrip() else False)
    return float(np.clip(0.3*uniq + 0.25*vdiv + 0.25*lbon + 0.2*end, 0, 1))

# === DREAM DEMO ===
def demo_dreams(fwd, inv, n=8):
    """Show planned dreams vs base generation side by side."""
    print(f"\n{'='*60}\nDREAM DEMOS\n{'='*60}", flush=True)

    demo_prompts = [
        "Once upon a time, there was a little cat who",
        "The brave knight rode into the dark forest and",
        "A tiny mouse found a magic key that could",
        "The princess was sad because she had lost her",
        "One day, a funny monkey decided to",
        "In a small village, there lived an old wizard who",
        "The sun was setting and the little bird",
        "Tom found a mysterious door behind the old tree and",
    ][:n]

    for prompt in demo_prompts:
        pids = encode_batch([prompt])
        c = ctx_proj(extract_ctx(pids).half())

        # Base generation (no dream)
        base_out = gen(pids, 100)
        base_text = tok.decode(base_out[0], skip_special_tokens=True)

        # With planned dream
        pre = gen(pids, 25)
        ctx = torch.cat([pids, pre], dim=-1)
        st = extract_states(ctx)
        a = plan(fwd, inv, st[0], c[0])

        apply_action(a)
        dream = gen(ctx, N_DREAM)
        revert()

        dream_text = tok.decode(dream[0], skip_special_tokens=True)

        post_ctx = torch.cat([ctx, dream], dim=-1)
        post = gen(post_ctx, 35)

        pre_text = tok.decode(pre[0], skip_special_tokens=True)
        post_text = tok.decode(post[0], skip_special_tokens=True)

        print(f"\nPrompt: {prompt}", flush=True)
        print(f"  BASE:  {base_text[:250]}", flush=True)
        print(f"  PRE:   {pre_text[:100]}", flush=True)
        print(f"  DREAM: >>> {dream_text[:150]} <<<", flush=True)
        print(f"  POST:  {post_text[:100]}", flush=True)
        print(f"  Action norm: {a.norm():.3f}, obj improvement: ", end="", flush=True)
        sp = fwd(st[0:1], a.unsqueeze(0), c[0:1])
        print(f"{objective(sp).item() - objective(st[0:1]).item():.3f}", flush=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    total_t0 = time.time()

    # Phase 1
    data = collect()

    # Freeze normalizer
    _nfrozen = True

    # Phase 2
    fwd, inv = train_models(data)

    # Phase 3
    results = evaluate(fwd, inv, n=80)

    # Demo
    demo_dreams(fwd, inv, n=8)

    # Save everything
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(fwd.state_dict(), "checkpoints/forward_model.pt")
    torch.save(inv.state_dict(), "checkpoints/inverse_model.pt")
    torch.save({"mean": _nm.cpu(), "var": _nv.cpu(), "count": _nc, "frozen": _nfrozen}, "checkpoints/normalizer.pt")
    torch.save(ctx_proj.state_dict(), "checkpoints/ctx_proj.pt")
    # Save basis
    torch.save({k: v.cpu() for k, v in DW.items()}, "checkpoints/lora_basis.pt")

    # Save results
    with open("checkpoints/results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = time.time() - total_t0
    print(f"\n{'='*60}\nTOTAL TIME: {total:.0f}s ({total/60:.1f}min)\n{'='*60}", flush=True)
