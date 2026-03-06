#!/usr/bin/env python
"""
DreamerML v6.1 - Iterated architecture.
Key changes from v6:
  - α=0.20 (was 0.06): perturbations actually matter
  - Rank-4 LoRA (was rank-1): more expressive
  - K=8 modes (was 32): fewer, more distinct directions
  - State = random projection of hidden states (was hand-crafted stats)
  - No inverse model: forward-model-guided search only
  - GRPO training of a dream policy head
  - Task-based evaluation with entity constraints
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, json
from contextlib import contextmanager
torch.backends.cudnn.benchmark = True

DEV = "cuda"
DT = torch.float16

# === CONFIG ===
K = 8           # dream modes
RANK = 4        # LoRA rank per mode
ALPHA = 0.20    # perturbation strength (was 0.06)
L2_CLAMP = 2.0
N_DREAM = 40
T_BASE = 24
TEMP = 0.9
TOP_P = 0.95
STATE_DIM = 128    # richer state from hidden projections
CTX_DIM = 64
BATCH = 96
N_TRANS = 20000
N_SEARCH = 128     # candidates for forward-model search

# === LOAD MODEL ===
print("Loading model...", flush=True)
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M", torch_dtype=DT).to(DEV)
model.eval()
for p in model.parameters(): p.requires_grad = False
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id
NL = model.config.num_layers
HID = model.config.hidden_size
print(f"Loaded {sum(p.numel() for p in model.parameters())/1e6:.0f}M params in {time.time()-t0:.1f}s", flush=True)

# === RANK-4 LORA BASIS ===
def get_targets():
    s, e = int(0.3*NL), int(0.7*NL)
    t = []
    for l in range(s, e+1):
        t.append(f"transformer.h.{l}.attn.attention.out_proj")
        t.append(f"transformer.h.{l}.mlp.c_proj")
    return t

def get_mod(name):
    m = model
    for p in name.split("."): m = getattr(m, p)
    return m

TARGETS = get_targets()
# For rank-4: A is (K, out, rank), B is (K, rank, in)
# ΔW_k = A_k @ B_k has shape (out, in) and rank 4
DW = {}  # precomputed {name: (K, out, in)}
DW_A = {}  # raw A matrices for reference
DW_B = {}
for name in TARGETS:
    w = get_mod(name).weight
    o, i = w.shape
    A = torch.randn(K, o, RANK, device=DEV, dtype=DT) * 0.01
    B = torch.randn(K, RANK, i, device=DEV, dtype=DT) * 0.01
    DW_A[name] = A
    DW_B[name] = B
    DW[name] = torch.bmm(A, B)  # (K, out, in)

print(f"LoRA basis: K={K}, rank={RANK}, α={ALPHA}, targets={TARGETS}", flush=True)

def clamp(a):
    a = a.clamp(-1, 1)
    if a.dim() == 1:
        n = a.norm()
        return a * min(1, L2_CLAMP/n) if n > L2_CLAMP else a
    n = a.norm(dim=-1, keepdim=True)
    return torch.where(n > L2_CLAMP, a * L2_CLAMP / n, a)

@contextmanager
def perturb(actions):
    """Per-sample LoRA hooks. actions: (B, K)"""
    a = clamp(actions.to(DEV, DT))
    hooks = []
    for name in TARGETS:
        dw = DW[name]
        mod = get_mod(name)
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

_orig = {}
def apply_single(a):
    a = clamp(a.to(DEV, DT))
    for name in TARGETS:
        mod = get_mod(name)
        _orig[name] = mod.weight.data.clone()
        mod.weight.data += ALPHA * torch.einsum("k,koi->oi", a, DW[name])

def revert_single():
    for name, w in _orig.items(): get_mod(name).weight.data.copy_(w)
    _orig.clear()

# === GENERATION ===
@torch.no_grad()
def gen(ids, n, temp=TEMP):
    mask = (ids != PAD).long()
    out = model.generate(ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n,
                         do_sample=True, temperature=temp, top_p=TOP_P, pad_token_id=PAD)
    return out[:, ids.shape[1]:]

# === STATE: RANDOM PROJECTION OF HIDDEN STATES ===
# Use last-token hidden from middle and last layers, random-projected
PROJ_LAYERS = [NL//2, NL-1]  # layers 2, 3 for 4-layer model
# Random projection matrices (frozen)
PROJ = {l: torch.randn(HID, STATE_DIM // len(PROJ_LAYERS), device=DEV, dtype=DT) * (2.0/HID)**0.5
        for l in PROJ_LAYERS}

# Running normalizer
_nm = torch.zeros(STATE_DIM, device=DEV)
_nv = torch.ones(STATE_DIM, device=DEV)
_nc = 0
_nfrozen = False
MOM = 0.002

def update_norm(x):
    global _nm, _nv, _nc
    if _nfrozen: return
    bm, bv = x.mean(0), x.var(0, unbiased=False) + 1e-8
    if _nc == 0: _nm, _nv = bm, bv
    else: _nm = (1-MOM)*_nm + MOM*bm; _nv = (1-MOM)*_nv + MOM*bv
    _nc += 1

def norm_state(x):
    return (x - _nm) / (_nv.sqrt() + 1e-8)

@torch.no_grad()
def extract_states(ids):
    """(B, seq) -> (B, STATE_DIM) via random projection of hidden states."""
    B = ids.shape[0]
    mask = (ids != PAD).long()
    out = model(ids, attention_mask=mask, output_hidden_states=True)
    last = (mask.sum(-1) - 1).clamp(min=0)
    bi = torch.arange(B, device=DEV)

    parts = []
    for l in PROJ_LAYERS:
        h = out.hidden_states[l+1]  # (B, seq, HID)
        hl = h[bi, last].float()     # (B, HID)
        proj = hl @ PROJ[l].float()  # (B, STATE_DIM//n_layers)
        parts.append(proj)

    s = torch.cat(parts, dim=-1)  # (B, STATE_DIM)
    update_norm(s)
    s = norm_state(s)
    s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    return s

@torch.no_grad()
def extract_ctx(ids):
    mask = (ids != PAD).long()
    out = model(ids, attention_mask=mask, output_hidden_states=True)
    B = ids.shape[0]
    last = (mask.sum(-1) - 1).clamp(min=0)
    return out.hidden_states[-1][torch.arange(B, device=DEV), last]

ctx_proj = nn.Linear(HID, CTX_DIM).to(DEV).half()

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
]

def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return e["input_ids"].to(DEV)

def rand_actions(B):
    a = torch.empty(B, K, device=DEV, dtype=DT)
    h = B // 2
    a[:h] = torch.rand(h, K, device=DEV, dtype=DT) * 2 - 1
    a[h:] = torch.tanh(torch.randn(B-h, K, device=DEV, dtype=DT) * 0.5)
    return clamp(a)

def check_quality(tokens):
    B = tokens.shape[0]
    q = torch.ones(B)
    for b in range(B):
        t = tokens[b].tolist()
        if len(t) >= 8:
            ng = set(); rep = 0
            for i in range(len(t)-3):
                g = tuple(t[i:i+4])
                if g in ng: rep += 1
                ng.add(g)
            if rep / max(1, len(t)-3) > 0.3: q[b] = 0
    return q

# === ENTITY TASKS ===
ENTITIES = ["cat","dog","bird","fish","rabbit","bear","mouse","fox","turtle","frog",
            "princess","knight","wizard","dragon","fairy","queen","king","pirate",
            "tree","flower","river","mountain","castle","garden","forest","cave"]

def make_task():
    """Generate entity-inclusion task. Returns (prompt, required_entities)."""
    ents = list(np.random.choice(ENTITIES, 3, replace=False))
    prompt = f"Write a story that includes a {ents[0]}, a {ents[1]}, and a {ents[2]}."
    return prompt, ents

def task_reward(text, required_ents):
    """Reward: entity coverage + diversity + coherence."""
    tl = text.lower()
    # Entity coverage (0-1)
    found = sum(1 for e in required_ents if e in tl) / len(required_ents)
    # Diversity
    words = tl.split()
    if len(words) < 5: return 0.1
    ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    uniq = len(set(ng)) / max(1, len(ng)) if ng else 0
    vdiv = len(set(words)) / len(words)
    # Length
    lbon = min(1.0, len(words) / 40)
    # Coherence penalty: repetition
    rep_penalty = 0
    if ng:
        from collections import Counter
        ngc = Counter(ng)
        most_common_frac = ngc.most_common(1)[0][1] / len(ng)
        if most_common_frac > 0.15: rep_penalty = 0.3
    return float(np.clip(0.40*found + 0.20*uniq + 0.15*vdiv + 0.15*lbon + 0.10*(1-rep_penalty), 0, 1))

# ============================================================
# PHASE 1: COLLECT TRANSITIONS
# ============================================================
def collect(n=N_TRANS, bs=BATCH):
    print(f"\n{'='*60}\nPHASE 1: Collecting {n} transitions (batch={bs}, α={ALPHA})\n{'='*60}", flush=True)
    all_s, all_a, all_ns, all_q, all_c = [], [], [], [], []
    t0 = time.time()
    for i in range(0, n, bs):
        B = min(bs, n-i)
        prompts = [np.random.choice(PROMPTS) for _ in range(B)]
        pids = encode_batch(prompts)
        c = ctx_proj(extract_ctx(pids).half())
        base = gen(pids, T_BASE)
        full = torch.cat([pids, base], dim=-1)
        st = extract_states(full)
        actions = rand_actions(B)
        with perturb(actions):
            dream = gen(full, N_DREAM)
        dfull = torch.cat([full, dream], dim=-1)
        sn = extract_states(dfull)
        q = check_quality(dream)
        all_s.append(st.float().cpu()); all_a.append(actions.float().cpu())
        all_ns.append(sn.float().cpu()); all_q.append(q); all_c.append(c.float().detach().cpu())
        done = i + B
        if done % (bs*4) < bs:
            el = time.time()-t0; rate = done/el
            qr = sum(qq.sum() for qq in all_q)/done
            print(f"  [{done}/{n}] {rate:.0f}/s quality={qr:.0%} ETA={max(0,(n-done)/rate):.0f}s", flush=True)
    data = {k: torch.cat(v) for k, v in zip(
        ["states","actions","next_states","quality","contexts"], [all_s, all_a, all_ns, all_q, all_c])}
    el = time.time()-t0
    print(f"Done: {n} in {el:.1f}s ({n/el:.0f}/s), quality={data['quality'].mean():.0%}", flush=True)
    return data

# ============================================================
# PHASE 2: TRAIN FORWARD MODEL (no inverse)
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

def train_fwd(data, epochs=20, bs=2048, lr=3e-4):
    print(f"\n{'='*60}\nPHASE 2: Training forward model\n{'='*60}", flush=True)
    d = {k: v.to(DEV) for k, v in data.items()}
    N = len(d["states"])
    fwd = FwdModel().to(DEV)
    opt = torch.optim.AdamW(fwd.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEV)
        total = 0; nb = 0
        for j in range(0, N, bs):
            idx = perm[j:j+bs]
            s, a, sn, q, c = d["states"][idx], d["actions"][idx], d["next_states"][idx], d["quality"][idx], d["contexts"][idx]
            pred = fwd(s, a, c)
            loss = (q.unsqueeze(-1) * (sn - pred)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        sched.step()
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={total/nb:.4f}", flush=True)

    # Diagnostics
    with torch.no_grad():
        perm = torch.randperm(N, device=DEV)[:5000]
        s, a, sn, c = d["states"][perm], d["actions"][perm], d["next_states"][perm], d["contexts"][perm]
        pred = fwd(s, a, c)
        dt = sn - s; dp = pred - s
        cos = F.cosine_similarity(dt, dp, dim=-1)
        r2 = 1 - ((sn - pred)**2).mean(0) / (sn.var(0) + 1e-8)
        print(f"  Delta cosine sim: {cos.mean():.4f} ± {cos.std():.4f}", flush=True)
        print(f"  R² > 0: {(r2>0).sum()}/{STATE_DIM}, R² > 0.1: {(r2>0.1).sum()}/{STATE_DIM}", flush=True)
        print(f"  Trained in {time.time()-t0:.1f}s", flush=True)
    return fwd

# ============================================================
# PHASE 3: PLANNING (forward-model search, no inverse)
# ============================================================
def plan_action(fwd, st, c, n_cand=N_SEARCH):
    """Search for best action using forward model predictions."""
    st_exp = st.unsqueeze(0).expand(n_cand, -1)
    c_exp = c.unsqueeze(0).expand(n_cand, -1)
    # Sample candidates
    ac = clamp(torch.tanh(torch.randn(n_cand, K, device=DEV) * 0.4))
    # Predict outcomes
    with torch.no_grad():
        sp = fwd(st_exp, ac, c_exp)
    # Score: distance from current state (we want maximal USEFUL change)
    # Use norm of predicted delta as proxy for "perturbation impact"
    delta = sp - st_exp
    delta_norm = delta.norm(dim=-1)
    # Pick action with largest predicted state change (most impactful dream)
    best = delta_norm.argmax()
    return clamp(ac[best])

# ============================================================
# PHASE 4: EVALUATE PLANNER VS RANDOM
# ============================================================
def eval_planner(fwd, n=200):
    print(f"\n{'='*60}\nPHASE 4: Evaluating planner vs random (n={n})\n{'='*60}", flush=True)
    planned_r, random_r, nodream_r = [], [], []
    planned_q, random_q = 0, 0
    t0 = time.time()

    for i in range(0, n, BATCH):
        B = min(BATCH, n-i)
        tasks = [make_task() for _ in range(B)]
        prompts = [t[0] for t in tasks]
        ents = [t[1] for t in tasks]
        pids = encode_batch(prompts)
        c = ctx_proj(extract_ctx(pids).half())

        # Shared base
        base = gen(pids, 30)
        ctx = torch.cat([pids, base], dim=-1)
        st = extract_states(ctx)

        # No dream
        nd = gen(ctx, 70)
        for b in range(B):
            txt = tok.decode(torch.cat([base[b], nd[b]]), skip_special_tokens=True)
            nodream_r.append(task_reward(txt, ents[b]))

        # Planned
        pa = torch.stack([plan_action(fwd, st[b], c[b]) for b in range(B)])
        with perturb(pa):
            pd = gen(ctx, N_DREAM)
        pq = check_quality(pd); planned_q += (pq < 0.5).sum().item()
        pctx = torch.cat([ctx, pd], dim=-1)
        pp = gen(pctx, 30)
        for b in range(B):
            txt = tok.decode(torch.cat([base[b], pd[b], pp[b]]), skip_special_tokens=True)
            planned_r.append(task_reward(txt, ents[b]))

        # Random
        ra = rand_actions(B)
        with perturb(ra):
            rd = gen(ctx, N_DREAM)
        rq = check_quality(rd); random_q += (rq < 0.5).sum().item()
        rctx = torch.cat([ctx, rd], dim=-1)
        rp = gen(rctx, 30)
        for b in range(B):
            txt = tok.decode(torch.cat([base[b], rd[b], rp[b]]), skip_special_tokens=True)
            random_r.append(task_reward(txt, ents[b]))

    pr, rr, nr = np.mean(planned_r), np.mean(random_r), np.mean(nodream_r)
    print(f"\n  No-dream:  {nr:.4f} ± {np.std(nodream_r):.4f}", flush=True)
    print(f"  Random:    {rr:.4f} ± {np.std(random_r):.4f} (degen={random_q/n:.1%})", flush=True)
    print(f"  Planned:   {pr:.4f} ± {np.std(planned_r):.4f} (degen={planned_q/n:.1%})", flush=True)
    print(f"  Plan-None: {pr-nr:+.4f}", flush=True)
    print(f"  Plan-Rand: {pr-rr:+.4f}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)
    return {"planned": pr, "random": rr, "nodream": nr,
            "plan_degen": planned_q/n, "rand_degen": random_q/n}

# ============================================================
# PHASE 5: GRPO - TRAIN DREAM POLICY
# ============================================================
class DreamPolicy(nn.Module):
    """Small MLP that decides: dream with mode k, or don't dream."""
    def __init__(self):
        super().__init__()
        # Input: projected hidden state
        self.net = nn.Sequential(
            nn.Linear(HID, 128), nn.ReLU(),
            nn.Linear(128, K + 1),  # K dream modes + 1 "no dream"
        )

    def forward(self, hidden):
        """hidden: (B, HID). Returns logits (B, K+1)."""
        return self.net(hidden)

def train_grpo(fwd, n_prompts=500, group_size=6):
    print(f"\n{'='*60}\nPHASE 5: GRPO training ({n_prompts} prompts, G={group_size})\n{'='*60}", flush=True)

    policy = DreamPolicy().to(DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    t0 = time.time()
    rewards_log = []
    dream_rate_log = []

    for pi in range(n_prompts):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        group_rewards = []
        group_log_probs = []
        group_dreamed = []

        for g in range(group_size):
            # Generate base text
            base = gen(pids, 30)
            ctx = torch.cat([pids, base], dim=-1)

            # Extract hidden state at dream decision point
            with torch.no_grad():
                out = model(ctx, output_hidden_states=True)
                h = out.hidden_states[-1][:, -1, :].float()  # (1, HID)

            # Policy decision
            logits = policy(h)  # (1, K+1)
            probs = F.softmax(logits, dim=-1)
            choice = torch.multinomial(probs, 1).squeeze()  # scalar
            log_prob = F.log_softmax(logits, dim=-1)[0, choice]

            dreamed = choice.item() < K

            if dreamed:
                # Construct action: one-hot-ish for the chosen mode
                action = torch.zeros(K, device=DEV, dtype=DT)
                action[choice.item()] = 0.7  # moderate strength
                # Add small noise
                action += torch.randn(K, device=DEV, dtype=DT) * 0.1
                action = clamp(action)

                apply_single(action)
                dream = gen(ctx, N_DREAM)
                revert_single()

                dream_ctx = torch.cat([ctx, dream], dim=-1)
                post = gen(dream_ctx, 30)
                full_gen = torch.cat([base, dream, post], dim=-1)
            else:
                full_gen = torch.cat([base, gen(ctx, 70)], dim=-1)

            text = tok.decode(full_gen[0], skip_special_tokens=True)
            r = task_reward(text, ents)
            group_rewards.append(r)
            group_log_probs.append(log_prob)
            group_dreamed.append(dreamed)

        # GRPO update
        rewards_t = torch.tensor(group_rewards, device=DEV)
        baseline = rewards_t.mean()
        advantages = rewards_t - baseline
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(group_log_probs)
        loss = -(advantages.detach() * log_probs_t).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        rewards_log.extend(group_rewards)
        dream_rate_log.append(sum(group_dreamed) / group_size)

        if (pi+1) % 50 == 0:
            recent_r = rewards_log[-300:]
            recent_dr = dream_rate_log[-50:]
            print(f"  [{pi+1}/{n_prompts}] reward={np.mean(recent_r):.4f} "
                  f"dream_rate={np.mean(recent_dr):.2f} loss={loss.item():.4f}", flush=True)

    print(f"  GRPO training done in {time.time()-t0:.1f}s", flush=True)
    return policy, rewards_log, dream_rate_log

# ============================================================
# PHASE 6: EVALUATE GRPO POLICY
# ============================================================
def eval_grpo(policy, n=200):
    print(f"\n{'='*60}\nPHASE 6: Evaluating GRPO policy (n={n})\n{'='*60}", flush=True)

    policy_r, base_r, always_dream_r = [], [], []
    dream_choices = []

    for i in range(n):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        # Base model (no adapter, no dream)
        base_gen = gen(pids, 100)
        base_txt = tok.decode(base_gen[0], skip_special_tokens=True)
        base_r.append(task_reward(base_txt, ents))

        # Policy-guided
        base = gen(pids, 30)
        ctx = torch.cat([pids, base], dim=-1)
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            logits = policy(h)
            choice = logits.argmax(dim=-1).item()

        dream_choices.append(choice)

        if choice < K:
            action = torch.zeros(K, device=DEV, dtype=DT)
            action[choice] = 0.7
            action = clamp(action)
            apply_single(action)
            dream = gen(ctx, N_DREAM)
            revert_single()
            post = gen(torch.cat([ctx, dream], dim=-1), 30)
            full = torch.cat([base, dream, post], dim=-1)
        else:
            full = torch.cat([base, gen(ctx, 70)], dim=-1)

        txt = tok.decode(full[0], skip_special_tokens=True)
        policy_r.append(task_reward(txt, ents))

        # Always-dream (random mode)
        action = torch.zeros(K, device=DEV, dtype=DT)
        action[np.random.randint(K)] = 0.7
        apply_single(clamp(action))
        dream = gen(ctx, N_DREAM)
        revert_single()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        full_ad = torch.cat([base, dream, post], dim=-1)
        txt_ad = tok.decode(full_ad[0], skip_special_tokens=True)
        always_dream_r.append(task_reward(txt_ad, ents))

    br, pr, ar = np.mean(base_r), np.mean(policy_r), np.mean(always_dream_r)
    from collections import Counter
    choice_dist = Counter(dream_choices)

    print(f"\n  Base model:     {br:.4f} ± {np.std(base_r):.4f}", flush=True)
    print(f"  Always dream:   {ar:.4f} ± {np.std(always_dream_r):.4f}", flush=True)
    print(f"  GRPO policy:    {pr:.4f} ± {np.std(policy_r):.4f}", flush=True)
    print(f"  Policy-Base:    {pr-br:+.4f}", flush=True)
    print(f"  Policy-Always:  {pr-ar:+.4f}", flush=True)
    print(f"  Dream rate:     {sum(1 for c in dream_choices if c < K)/len(dream_choices):.1%}", flush=True)
    print(f"  Mode distribution: {dict(choice_dist)}", flush=True)

    return {"base": br, "policy": pr, "always_dream": ar,
            "dream_rate": sum(1 for c in dream_choices if c < K)/len(dream_choices),
            "choice_dist": dict(choice_dist)}

# ============================================================
# DEMO
# ============================================================
def demo(policy, n=8):
    print(f"\n{'='*60}\nDREAM DEMOS\n{'='*60}", flush=True)
    demo_prompts = [
        "Once upon a time, there was a little cat who",
        "The brave knight rode into the dark forest and",
        "A tiny mouse found a magic key that could",
        "The princess was sad because she had lost her",
        "One day, a funny monkey decided to",
        "In a small village, there lived an old wizard who",
        "Tom found a mysterious door behind the old tree and",
        "Write a story that includes a dragon, a flower, and a cave.",
    ][:n]

    for prompt in demo_prompts:
        pids = encode_batch([prompt])

        # Base
        base_out = gen(pids, 100)
        base_text = tok.decode(base_out[0], skip_special_tokens=True)

        # Policy
        pre = gen(pids, 25)
        ctx = torch.cat([pids, pre], dim=-1)
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            logits = policy(h)
            choice = logits.argmax(dim=-1).item()
            probs = F.softmax(logits, dim=-1)[0]

        pre_text = tok.decode(pre[0], skip_special_tokens=True)

        if choice < K:
            action = torch.zeros(K, device=DEV, dtype=DT)
            action[choice] = 0.7
            apply_single(clamp(action))
            dream = gen(ctx, N_DREAM)
            revert_single()
            dream_text = tok.decode(dream[0], skip_special_tokens=True)
            post = gen(torch.cat([ctx, dream], dim=-1), 35)
            post_text = tok.decode(post[0], skip_special_tokens=True)
        else:
            dream_text = "(no dream)"
            post = gen(ctx, 75)
            post_text = tok.decode(post[0], skip_special_tokens=True)

        mode_probs = ", ".join(f"m{i}={probs[i]:.2f}" for i in range(K))
        print(f"\nPrompt: {prompt}", flush=True)
        print(f"  BASE:   {base_text[:200]}", flush=True)
        print(f"  PRE:    {pre_text[:100]}", flush=True)
        print(f"  CHOICE: mode={choice} ({mode_probs}, none={probs[K]:.2f})", flush=True)
        print(f"  DREAM:  >>> {dream_text[:150]} <<<", flush=True)
        print(f"  POST:   {post_text[:100]}", flush=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    T0 = time.time()

    # Phase 1: Collect
    data = collect()
    _nfrozen = True

    # Phase 2: Train forward model
    fwd = train_fwd(data)

    # Phase 3: Evaluate planner vs random
    plan_results = eval_planner(fwd, n=200)

    # Phase 4: GRPO
    policy, rewards_log, dream_rate_log = train_grpo(fwd, n_prompts=500, group_size=6)

    # Phase 5: Evaluate GRPO policy
    grpo_results = eval_grpo(policy, n=200)

    # Phase 6: Demo
    demo(policy)

    # Save
    os.makedirs("checkpoints2", exist_ok=True)
    torch.save(fwd.state_dict(), "checkpoints2/fwd.pt")
    torch.save(policy.state_dict(), "checkpoints2/policy.pt")
    results = {"planner": plan_results, "grpo": grpo_results}
    with open("checkpoints2/results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = time.time() - T0
    print(f"\n{'='*60}\nTOTAL: {total:.0f}s ({total/60:.1f}min)\n{'='*60}", flush=True)
