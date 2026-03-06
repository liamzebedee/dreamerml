#!/usr/bin/env python
"""
DreamerML v6.5 - Learned LoRA modes with proper mode selection.

v6.4 proved: fine-tuned modes DO steer content (+21% animals, +19% objects, etc.)
But GRPO failed because:
  1. Softmax over modes → each mode gets only 0.25 weight
  2. α=0.3 * 0.25 = 0.075 effective → too weak
  3. Policy can't differentiate → outputs uniform

Fixes:
  1. Independent per-mode activations (sigmoid, not softmax)
  2. Higher α=0.5, action=1.0 per mode → effective perturbation = 0.5
  3. Binary GRPO: for each mode, learn when to turn it on/off
  4. Oracle uses hardcoded entity→mode mapping at full strength
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, json
from contextlib import contextmanager
from collections import Counter
torch.backends.cudnn.benchmark = True

DEV = "cuda"; DT = torch.float16
K = 4; RANK = 8; ALPHA = 0.5; L2_CLAMP = 2.0
N_DREAM = 60; T_BASE = 15; TEMP = 0.9; TOP_P = 0.95; BATCH = 64

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
PAD = tok.pad_token_id; NL = model.config.num_layers; HID = model.config.hidden_size
print(f"Loaded {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)

# === ENTITY CATEGORIES ===
ENTITY_CATEGORIES = {
    0: {"name": "animals", "entities": ["cat","dog","bird","fish","rabbit","bear","mouse","fox","turtle","frog"]},
    1: {"name": "characters", "entities": ["princess","knight","wizard","dragon","fairy","queen","king","pirate"]},
    2: {"name": "places", "entities": ["river","mountain","castle","garden","forest","cave"]},
    3: {"name": "objects", "entities": ["tree","flower","boat","lamp","key","crown","sword","shield","map","book"]},
}
ALL_ENTITIES = []
ENTITY_TO_MODE = {}
for k, v in ENTITY_CATEGORIES.items():
    for e in v["entities"]: ALL_ENTITIES.append(e); ENTITY_TO_MODE[e] = k

def get_mod(name):
    m = model
    for p in name.split("."): m = getattr(m, p)
    return m

TARGETS = []
for l in range(NL):
    TARGETS.append(f"transformer.h.{l}.attn.attention.out_proj")
    TARGETS.append(f"transformer.h.{l}.mlp.c_proj")

class LoRAMode(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for name in TARGETS:
            w = get_mod(name).weight; o, i = w.shape
            safe = name.replace(".", "_")
            self.A[safe] = nn.Parameter(torch.randn(o, RANK, device=DEV) * (2.0/(o*RANK))**0.5)
            self.B[safe] = nn.Parameter(torch.randn(RANK, i, device=DEV) * (2.0/(i*RANK))**0.5)
    def get_dw(self, name):
        safe = name.replace(".", "_")
        return self.A[safe] @ self.B[safe]

modes = nn.ModuleList([LoRAMode() for _ in range(K)]).to(DEV)
print(f"LoRA: K={K}, rank={RANK}, α={ALPHA}, {sum(p.numel() for p in modes.parameters())/1e3:.0f}K params", flush=True)

_orig = {}
def apply_modes(activations):
    """Apply modes with independent activations. activations: (K,) values in [0,1]."""
    for name in TARGETS:
        mod = get_mod(name); _orig[name] = mod.weight.data.clone()
        combined = sum(activations[k].item() * modes[k].get_dw(name).detach().float()
                      for k in range(K) if abs(activations[k].item()) > 0.01)
        if isinstance(combined, (int, float)): continue
        mod.weight.data += (ALPHA * combined).to(DT)

def revert():
    for name, w in _orig.items(): get_mod(name).weight.data.copy_(w)
    _orig.clear()

@contextmanager
def apply_single_mode(k):
    """Apply a single mode for fine-tuning (differentiable)."""
    hooks = []
    for name in TARGETS:
        mod = get_mod(name); dw = modes[k].get_dw(name)
        def make_hook(dw_ref):
            def fn(module, inp, out):
                x = inp[0]; correction = F.linear(x.float(), dw_ref.float()) * ALPHA
                return out + correction.to(out.dtype)
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

def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return e["input_ids"].to(DEV)

def make_task():
    ents = list(np.random.choice(ALL_ENTITIES, 3, replace=False))
    return f"Write a story that includes a {ents[0]}, a {ents[1]}, and a {ents[2]}.", ents

def task_reward(text, required_ents):
    tl = text.lower(); words = tl.split()
    if len(words) < 5: return 0.05
    found = sum(1 for e in required_ents if e in tl) / len(required_ents)
    ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    uniq = len(set(ng))/max(1,len(ng)) if ng else 0
    vdiv = len(set(words))/len(words); lbon = min(1.0, len(words)/50)
    rep = 0
    if ng:
        mcf = Counter(ng).most_common(1)[0][1]/len(ng)
        if mcf > 0.12: rep = 0.5
        if mcf > 0.25: rep = 1.0
    return float(np.clip(0.50*found + 0.20*uniq + 0.10*vdiv + 0.10*lbon + 0.10*(1-rep), 0, 1))

def get_oracle_action(ents):
    """Hardcoded entity→mode mapping."""
    a = torch.zeros(K, device=DEV)
    for e in ents:
        if e in ENTITY_TO_MODE: a[ENTITY_TO_MODE[e]] = 1.0
    return a

STORY_PROMPTS = [
    "Once upon a time, there was a", "One day, a little girl named",
    "There was a big dog who", "The cat and the bird were",
    "A brave knight went to the", "In a small village, there lived",
    "The sun was shining and the", "A funny monkey found a",
    "The princess looked out the window", "Tom and his friend went to",
    "The little bunny hopped into the", "A magic fish swam in the",
    "Write a story about a cat and a dog.",
    "Write a story about a princess in a castle.",
    "Write a story about a wizard and a dragon.",
    "Write a story about a knight in a forest.",
    "Write a story about a bear and a river.",
    "Write a story about a fairy with a magic key.",
    "Write a story about a fox and a flower.",
    "Write a story about a pirate on a boat.",
]

# ============================================================
# PHASE 1: GENERATE TRAINING DATA
# ============================================================
def generate_training_data(n_stories=2000, min_per_mode=200):
    print(f"\n{'='*60}\nPHASE 1: Training data ({n_stories} stories)\n{'='*60}", flush=True)
    t0 = time.time()
    mode_data = {k: [] for k in range(K)}
    total = 0
    while min(len(v) for v in mode_data.values()) < min_per_mode or total < n_stories:
        if total > n_stories * 3: break
        prompts = [np.random.choice(STORY_PROMPTS) for _ in range(BATCH)]
        pids = encode_batch(prompts)
        with torch.no_grad(): cont = gen(pids, 100)
        for b in range(len(prompts)):
            full_ids = torch.cat([pids[b], cont[b]])
            text = tok.decode(full_ids, skip_special_tokens=True).lower()
            for mk in range(K):
                if any(e in text for e in ENTITY_CATEGORIES[mk]["entities"]):
                    mode_data[mk].append(full_ids.clone())
        total += len(prompts)
        if total % (BATCH*5) < BATCH:
            print(f"  {total} stories. Modes: {[len(v) for v in mode_data.values()]}", flush=True)
    print(f"  Done in {time.time()-t0:.0f}s. Modes: {[len(v) for v in mode_data.values()]}", flush=True)
    return mode_data

# ============================================================
# PHASE 2: FINE-TUNE MODES
# ============================================================
def finetune_modes(mode_data, epochs=40, lr=5e-4, bs=16):
    print(f"\n{'='*60}\nPHASE 2: Fine-tuning LoRA modes ({epochs} epochs)\n{'='*60}", flush=True)
    t0 = time.time()
    model.float()
    for k in range(K):
        data = mode_data[k]
        if len(data) < 10: continue
        max_len = max(d.shape[0] for d in data)
        padded = torch.full((len(data), max_len), PAD, dtype=torch.long, device=DEV)
        for i, d in enumerate(data):
            padded[i, max_len-len(d):] = d
        opt = torch.optim.AdamW(modes[k].parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for ep in range(epochs):
            perm = torch.randperm(len(data)); tl = 0; nb = 0
            for j in range(0, len(data), bs):
                batch = padded[perm[j:j+bs]]; mask = (batch != PAD).long()
                labels = batch.clone(); labels[labels == PAD] = -100
                with apply_single_mode(k):
                    loss = model(batch, attention_mask=mask, labels=labels).loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(modes[k].parameters(), 1.0)
                opt.step(); tl += loss.item(); nb += 1
            sched.step()
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"  Mode {k} ({ENTITY_CATEGORIES[k]['name']}) ep {ep+1}: loss={tl/nb:.4f}", flush=True)
    model.half()
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# PHASE 2B: VERIFY
# ============================================================
def verify_modes(n=200):
    print(f"\n{'='*60}\nPHASE 2B: Verify specialization (n={n})\n{'='*60}", flush=True)
    t0 = time.time()
    results = {}
    for k in range(-1, K):
        counts = {kk: 0 for kk in range(K)}
        for start in range(0, n, BATCH):
            B = min(BATCH, n-start)
            pids = encode_batch(["Once upon a time,"] * B)
            if k >= 0:
                a = torch.zeros(K, device=DEV); a[k] = 1.0
                apply_modes(a)
            out = gen(pids, 100)
            if k >= 0: revert()
            for b in range(B):
                txt = tok.decode(out[b], skip_special_tokens=True).lower()
                for kk in range(K):
                    if any(e in txt for e in ENTITY_CATEGORIES[kk]["entities"]):
                        counts[kk] += 1
        results[k] = counts
        label = "BASE" if k == -1 else f"Mode {k} ({ENTITY_CATEGORIES[k]['name']})"
        s = " ".join(f"{ENTITY_CATEGORIES[kk]['name'][:4]}={counts[kk]/n:.0%}" for kk in range(K))
        print(f"  {label}: {s}", flush=True)

    # Delta analysis
    print(f"\n  Specialization deltas (mode activation - base):", flush=True)
    for k in range(K):
        base_rate = results[-1][k] / n
        mode_rate = results[k][k] / n
        print(f"    Mode {k} ({ENTITY_CATEGORIES[k]['name']}): {base_rate:.0%} → {mode_rate:.0%} ({mode_rate-base_rate:+.0%})", flush=True)

    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    return results

# ============================================================
# PHASE 3: GRPO WITH INDEPENDENT MODE ACTIVATIONS
# ============================================================
class ModePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HID, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, K),  # one logit per mode, independent sigmoid
        )
    def forward(self, h):
        return torch.sigmoid(self.net(h))  # (B, K) each in [0,1]

def train_grpo(n_prompts=800, group_size=6):
    print(f"\n{'='*60}\nPHASE 3: GRPO ({n_prompts} prompts, G={group_size})\n{'='*60}", flush=True)
    policy = ModePolicy().to(DEV)
    # Bias toward activating all modes initially
    with torch.no_grad():
        policy.net[-1].bias.fill_(1.0)  # sigmoid(1) ≈ 0.73
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    t0 = time.time()
    rewards_log, activations_log = [], []

    for pi in range(n_prompts):
        prompt, ents = make_task()
        pids = encode_batch([prompt])
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()

        probs = policy(h)  # (1, K) each in [0,1]

        group_rewards, group_log_probs = [], []

        for g in range(group_size):
            # Sample binary activations for each mode
            sampled = torch.bernoulli(probs[0])  # (K,)
            log_prob = (sampled * torch.log(probs[0] + 1e-8) +
                       (1-sampled) * torch.log(1-probs[0] + 1e-8)).sum()

            if sampled.sum() > 0:
                apply_modes(sampled)
                dream = gen(ctx, N_DREAM)
                revert()
                post = gen(torch.cat([ctx, dream], dim=-1), 30)
                full = torch.cat([base, dream, post], dim=-1)
            else:
                full = torch.cat([base, gen(ctx, N_DREAM + 30)], dim=-1)

            txt = tok.decode(full[0], skip_special_tokens=True)
            r = task_reward(txt, ents)
            group_rewards.append(r)
            group_log_probs.append(log_prob)

        rewards_t = torch.tensor(group_rewards, device=DEV)
        advantages = rewards_t - rewards_t.mean()
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(group_log_probs)
        pg_loss = -(advantages.detach() * log_probs_t).mean()

        # Entropy bonus on each mode's activation probability
        ent = -(probs * torch.log(probs + 1e-8) + (1-probs) * torch.log(1-probs + 1e-8)).mean()
        ent_coef = max(0.005, 0.05 * (1 - pi/n_prompts))
        loss = pg_loss - ent_coef * ent

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        rewards_log.extend(group_rewards)
        activations_log.append(probs[0].detach().cpu().numpy())

        if (pi+1) % 100 == 0:
            recent_r = rewards_log[-600:]
            recent_a = np.mean(activations_log[-100:], axis=0)
            a_str = " ".join(f"m{k}={recent_a[k]:.2f}" for k in range(K))
            print(f"  [{pi+1}/{n_prompts}] R={np.mean(recent_r):.4f} activations=[{a_str}]", flush=True)

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return policy

# ============================================================
# PHASE 4: EVALUATION
# ============================================================
def evaluate(policy, n=400):
    print(f"\n{'='*60}\nPHASE 4: Evaluation (n={n})\n{'='*60}", flush=True)
    t0 = time.time()
    base_r, oracle_r, policy_r, random_r, all_modes_r = [], [], [], [], []

    for i in range(n):
        prompt, ents = make_task()
        pids = encode_batch([prompt])
        base_tok = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base_tok], dim=-1)

        # === BASE (no mode) ===
        nd = gen(ctx, N_DREAM + 30)
        base_r.append(task_reward(tok.decode(torch.cat([base_tok[0], nd[0]]), skip_special_tokens=True), ents))

        # === ORACLE ===
        oa = get_oracle_action(ents)
        apply_modes(oa)
        dream = gen(ctx, N_DREAM)
        revert()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        oracle_r.append(task_reward(tok.decode(torch.cat([base_tok[0], dream[0], post[0]]), skip_special_tokens=True), ents))

        # === ALL MODES ===
        aa = torch.ones(K, device=DEV)
        apply_modes(aa)
        dream = gen(ctx, N_DREAM)
        revert()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        all_modes_r.append(task_reward(tok.decode(torch.cat([base_tok[0], dream[0], post[0]]), skip_special_tokens=True), ents))

        # === POLICY ===
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            probs = policy(h)[0]
            pa = (probs > 0.5).float()  # threshold at 0.5

        if pa.sum() > 0:
            apply_modes(pa)
            dream = gen(ctx, N_DREAM)
            revert()
            post = gen(torch.cat([ctx, dream], dim=-1), 30)
            full_p = torch.cat([base_tok, dream, post], dim=-1)
        else:
            full_p = torch.cat([base_tok, gen(ctx, N_DREAM+30)], dim=-1)
        policy_r.append(task_reward(tok.decode(full_p[0], skip_special_tokens=True), ents))

        # === RANDOM (single random mode) ===
        ra = torch.zeros(K, device=DEV); ra[np.random.randint(K)] = 1.0
        apply_modes(ra)
        dream = gen(ctx, N_DREAM)
        revert()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        random_r.append(task_reward(tok.decode(torch.cat([base_tok[0], dream[0], post[0]]), skip_special_tokens=True), ents))

    br, orr, pr, rr, ar = np.mean(base_r), np.mean(oracle_r), np.mean(policy_r), np.mean(random_r), np.mean(all_modes_r)

    print(f"\n  No dream (base):       {br:.4f} ± {np.std(base_r):.4f}", flush=True)
    print(f"  Random mode:           {rr:.4f} ± {np.std(random_r):.4f}", flush=True)
    print(f"  All modes on:          {ar:.4f} ± {np.std(all_modes_r):.4f}", flush=True)
    print(f"  Oracle (entity→mode):  {orr:.4f} ± {np.std(oracle_r):.4f}", flush=True)
    print(f"  GRPO policy:           {pr:.4f} ± {np.std(policy_r):.4f}", flush=True)
    print(f"\n  Oracle - Base:   {orr-br:+.4f}", flush=True)
    print(f"  Policy - Base:   {pr-br:+.4f}", flush=True)
    print(f"  All    - Base:   {ar-br:+.4f}", flush=True)
    print(f"  Random - Base:   {rr-br:+.4f}", flush=True)

    from scipy import stats
    results = {}
    for label, vals in [("oracle", oracle_r), ("policy", policy_r), ("all_modes", all_modes_r), ("random", random_r)]:
        t, p = stats.ttest_rel(vals, base_r)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {label} vs base: t={t:.3f}, p={p:.4f} {sig}", flush=True)
        results[f"p_{label}"] = p

    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    results.update({"base": br, "oracle": orr, "policy": pr, "random": rr, "all_modes": ar})
    return results

# ============================================================
# DEMO
# ============================================================
def demo(policy, n=8):
    print(f"\n{'='*60}\nDEMO\n{'='*60}", flush=True)
    tasks = [
        ("Write a story that includes a cat, a castle, and a river.", ["cat", "castle", "river"]),
        ("Write a story that includes a dragon, a flower, and a cave.", ["dragon", "flower", "cave"]),
        ("Write a story that includes a knight, a mouse, and a forest.", ["knight", "mouse", "forest"]),
        ("Write a story that includes a princess, a key, and a garden.", ["princess", "key", "garden"]),
        ("Write a story that includes a bear, a boat, and a mountain.", ["bear", "boat", "mountain"]),
        ("Write a story that includes a wizard, a bird, and a lamp.", ["wizard", "bird", "lamp"]),
        ("Write a story that includes a fox, a crown, and a tree.", ["fox", "crown", "tree"]),
        ("Write a story that includes a fairy, a sword, and a frog.", ["fairy", "sword", "frog"]),
    ][:n]

    for prompt, ents in tasks:
        pids = encode_batch([prompt])
        pre = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, pre], dim=-1)

        # Base
        nd = gen(ctx, N_DREAM + 30)
        base_text = tok.decode(torch.cat([pre[0], nd[0]]), skip_special_tokens=True)

        # Oracle
        oa = get_oracle_action(ents)
        apply_modes(oa)
        dream_o = gen(ctx, N_DREAM)
        revert()
        post_o = gen(torch.cat([ctx, dream_o], dim=-1), 30)
        oracle_text = tok.decode(torch.cat([pre[0], dream_o[0], post_o[0]]), skip_special_tokens=True)

        # Policy
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            probs = policy(h)[0]
            pa = (probs > 0.5).float()
        if pa.sum() > 0:
            apply_modes(pa)
            dream_p = gen(ctx, N_DREAM)
            revert()
            post_p = gen(torch.cat([ctx, dream_p], dim=-1), 30)
            policy_text = tok.decode(torch.cat([pre[0], dream_p[0], post_p[0]]), skip_special_tokens=True)
        else:
            nd_p = gen(ctx, N_DREAM + 30)
            policy_text = tok.decode(torch.cat([pre[0], nd_p[0]]), skip_special_tokens=True)

        bf = [e for e in ents if e in base_text.lower()]
        of = [e for e in ents if e in oracle_text.lower()]
        pf = [e for e in ents if e in policy_text.lower()]
        modes_on = [k for k in range(K) if oa[k] > 0]
        policy_on = [k for k in range(K) if pa[k] > 0]

        print(f"\nTask: [{', '.join(ents)}]  oracle_modes={modes_on}  policy_modes={policy_on}", flush=True)
        print(f"  BASE   ({task_reward(base_text,ents):.3f}, {bf}): {base_text[:180]}", flush=True)
        print(f"  ORACLE ({task_reward(oracle_text,ents):.3f}, {of}): {oracle_text[:180]}", flush=True)
        print(f"  POLICY ({task_reward(policy_text,ents):.3f}, {pf}): {policy_text[:180]}", flush=True)

# ============================================================
if __name__ == "__main__":
    T0 = time.time()
    mode_data = generate_training_data(n_stories=2000, min_per_mode=200)
    finetune_modes(mode_data, epochs=40, lr=5e-4)
    verify_modes(n=200)
    policy = train_grpo(n_prompts=800, group_size=6)
    eval_results = evaluate(policy, n=400)
    demo(policy)
    os.makedirs("checkpoints6", exist_ok=True)
    torch.save(modes.state_dict(), "checkpoints6/modes.pt")
    torch.save(policy.state_dict(), "checkpoints6/policy.pt")
    with open("checkpoints6/results.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=float)
    total = time.time() - T0
    print(f"\n{'='*60}\nTOTAL: {total:.0f}s ({total/60:.1f}min)\n{'='*60}", flush=True)
