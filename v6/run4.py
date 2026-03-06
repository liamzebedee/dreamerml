#!/usr/bin/env python
"""
DreamerML v6.3 - Fixed GRPO with warm-start + test-time CEM.

v6.2 proved: best-of-256 achieves 0.90 vs baseline 0.57 (+0.33, 100% tasks).
But GRPO collapsed to never dreaming because random exploration noise is useless.

Fixes:
  1. More oracle data (150 tasks, CEM-refined)
  2. Robustness check: verify oracle actions work across random seeds
  3. GRPO warm-start from predictor, tiny noise (std=0.1), entropy bonus
  4. Test-time CEM as practical approach
  5. Better predictor: use reward-weighted training, not just MSE on best action
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, json
from contextlib import contextmanager
from collections import Counter
torch.backends.cudnn.benchmark = True

DEV = "cuda"
DT = torch.float16

# === CONFIG ===
K = 4
RANK = 8
ALPHA = 0.5
L2_CLAMP = 2.0
N_DREAM = 50
T_BASE = 20
TEMP = 0.9
TOP_P = 0.95
BATCH = 64

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

# === LORA BASIS ===
def get_targets():
    t = []
    for l in range(NL):
        t.append(f"transformer.h.{l}.attn.attention.out_proj")
        t.append(f"transformer.h.{l}.mlp.c_proj")
    return t

def get_mod(name):
    m = model
    for p in name.split("."): m = getattr(m, p)
    return m

TARGETS = get_targets()
DW = {}
for name in TARGETS:
    w = get_mod(name).weight
    o, i = w.shape
    A = torch.randn(K, o, RANK, device=DEV, dtype=DT) * (2.0 / (o * RANK))**0.5
    B = torch.randn(K, RANK, i, device=DEV, dtype=DT) * (2.0 / (i * RANK))**0.5
    DW[name] = torch.bmm(A, B)

print(f"LoRA: K={K}, rank={RANK}, α={ALPHA}, {len(TARGETS)} targets", flush=True)

def clamp(a):
    a = a.clamp(-1, 1)
    if a.dim() == 1:
        n = a.norm()
        return a * min(1, L2_CLAMP/n) if n > L2_CLAMP else a
    n = a.norm(dim=-1, keepdim=True)
    return torch.where(n > L2_CLAMP, a * L2_CLAMP / n, a)

@contextmanager
def perturb(actions):
    a = clamp(actions.to(DEV, DT))
    hooks = []
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

_orig = {}
def apply_single(a):
    a = clamp(a.to(DEV, DT))
    for name in TARGETS:
        mod = get_mod(name); _orig[name] = mod.weight.data.clone()
        mod.weight.data += ALPHA * torch.einsum("k,koi->oi", a, DW[name])

def revert_single():
    for name, w in _orig.items(): get_mod(name).weight.data.copy_(w)
    _orig.clear()

@torch.no_grad()
def gen(ids, n, temp=TEMP):
    mask = (ids != PAD).long()
    out = model.generate(ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n,
                         do_sample=True, temperature=temp, top_p=TOP_P, pad_token_id=PAD)
    return out[:, ids.shape[1]:]

# === TASKS ===
ENTITIES = ["cat","dog","bird","fish","rabbit","bear","mouse","fox","turtle","frog",
            "princess","knight","wizard","dragon","fairy","queen","king","pirate",
            "tree","flower","river","mountain","castle","garden","forest","cave",
            "boat","lamp","key","crown","sword","shield","map","book"]

def make_task():
    ents = list(np.random.choice(ENTITIES, 3, replace=False))
    return f"Write a story that includes a {ents[0]}, a {ents[1]}, and a {ents[2]}.", ents

def task_reward(text, required_ents):
    tl = text.lower(); words = tl.split()
    if len(words) < 5: return 0.05
    found = sum(1 for e in required_ents if e in tl) / len(required_ents)
    ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    uniq = len(set(ng)) / max(1, len(ng)) if ng else 0
    vdiv = len(set(words)) / len(words)
    lbon = min(1.0, len(words) / 50)
    rep_penalty = 0
    if ng:
        ngc = Counter(ng)
        mcf = ngc.most_common(1)[0][1] / len(ng)
        if mcf > 0.12: rep_penalty = 0.5
        if mcf > 0.25: rep_penalty = 1.0
    return float(np.clip(0.50*found + 0.20*uniq + 0.10*vdiv + 0.10*lbon + 0.10*(1-rep_penalty), 0, 1))

def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return e["input_ids"].to(DEV)

def rand_actions(B):
    a = torch.empty(B, K, device=DEV, dtype=DT)
    h = B // 2
    a[:h] = torch.rand(h, K, device=DEV, dtype=DT) * 2 - 1
    a[h:] = torch.tanh(torch.randn(B-h, K, device=DEV, dtype=DT) * 0.5)
    return clamp(a)

def generate_with_action(ctx, base, action):
    """Generate dream+post text with a single action. Returns (text, reward_ready_tokens)."""
    apply_single(action)
    dream = gen(ctx, N_DREAM)
    revert_single()
    post = gen(torch.cat([ctx, dream], dim=-1), 30)
    return torch.cat([base, dream, post], dim=-1)

# ============================================================
# PHASE 1: CEM-REFINED ACTION SEARCH
# ============================================================
def cem_search(ctx, base, ents, n_cand=128, n_iters=3, elite_frac=0.1):
    """CEM to find best action for a task."""
    mu = torch.zeros(K, device=DEV, dtype=DT)
    sigma = torch.ones(K, device=DEV, dtype=DT) * 0.5
    best_r = -1; best_a = None

    for it in range(n_iters):
        # Sample candidates
        actions = mu + sigma * torch.randn(n_cand, K, device=DEV, dtype=DT)
        actions = clamp(actions)

        # Evaluate in batches
        rewards = []
        for start in range(0, n_cand, BATCH):
            B = min(BATCH, n_cand - start)
            batch_actions = actions[start:start+B]
            ctx_exp = ctx.expand(B, -1).contiguous()
            with perturb(batch_actions):
                dream = gen(ctx_exp, N_DREAM)
            dream_ctx = torch.cat([ctx_exp, dream], dim=-1)
            post = gen(dream_ctx, 30)
            for b in range(B):
                txt = tok.decode(torch.cat([base[0], dream[b], post[b]]), skip_special_tokens=True)
                rewards.append(task_reward(txt, ents))

        rewards = np.array(rewards[:n_cand])

        # Update best
        idx = rewards.argmax()
        if rewards[idx] > best_r:
            best_r = rewards[idx]; best_a = actions[idx].clone()

        # Fit elite
        n_elite = max(2, int(n_cand * elite_frac))
        elite_idx = rewards.argsort()[-n_elite:]
        elite = actions[elite_idx]
        mu = elite.mean(0); sigma = elite.std(0).clamp(min=0.05)

    return best_a, best_r, np.mean(rewards)

def search_oracle(n_tasks=150):
    """CEM search to find oracle actions for many tasks."""
    print(f"\n{'='*60}\nPHASE 1: CEM oracle search ({n_tasks} tasks)\n{'='*60}", flush=True)
    t0 = time.time()

    results = []
    baseline_r, oracle_r, random_r = [], [], []

    for ti in range(n_tasks):
        prompt, ents = make_task()
        pids = encode_batch([prompt])
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # Baseline (4 samples, mean)
        bl = []
        for _ in range(4):
            nd = gen(ctx, N_DREAM + 30)
            txt = tok.decode(torch.cat([base[0], nd[0]]), skip_special_tokens=True)
            bl.append(task_reward(txt, ents))
        bl_r = np.mean(bl)
        baseline_r.append(bl_r)

        # CEM search
        best_a, best_r, mean_r = cem_search(ctx, base, ents, n_cand=128, n_iters=3)
        oracle_r.append(best_r)
        random_r.append(mean_r)

        # Get prompt hidden
        with torch.no_grad():
            out = model(pids, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float().cpu()

        results.append({
            "hidden": h[0], "best_action": best_a.cpu(), "best_reward": best_r,
            "baseline_reward": bl_r, "prompt": prompt, "ents": ents,
        })

        if (ti+1) % 20 == 0:
            el = time.time()-t0
            print(f"  [{ti+1}/{n_tasks}] bl={np.mean(baseline_r):.4f} oracle={np.mean(oracle_r):.4f} "
                  f"rand={np.mean(random_r):.4f} improved={np.mean([r['best_reward']>r['baseline_reward'] for r in results]):.0%} "
                  f"time={el:.0f}s", flush=True)

    bl_m, or_m, rr_m = np.mean(baseline_r), np.mean(oracle_r), np.mean(random_r)
    deltas = [r["best_reward"]-r["baseline_reward"] for r in results]
    print(f"\n  Baseline:  {bl_m:.4f}", flush=True)
    print(f"  Random:    {rr_m:.4f}", flush=True)
    print(f"  Oracle:    {or_m:.4f}", flush=True)
    print(f"  Δ(oracle-bl): {np.mean(deltas):+.4f}", flush=True)
    print(f"  Improved:  {np.mean([d>0 for d in deltas]):.0%}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    return results

# ============================================================
# PHASE 1B: ROBUSTNESS CHECK
# ============================================================
def check_robustness(results, n_checks=20, n_resample=8):
    """Verify oracle actions work across random seeds."""
    print(f"\n{'='*60}\nPHASE 1B: Robustness check ({n_checks} tasks, {n_resample} resamples each)\n{'='*60}", flush=True)
    t0 = time.time()

    robust_deltas = []
    for i in range(min(n_checks, len(results))):
        r = results[i]
        pids = encode_batch([r["prompt"]])
        action = r["best_action"].to(DEV, DT)

        # Re-evaluate oracle action n_resample times with fresh random seeds
        oracle_rewards = []
        base_rewards = []
        for _ in range(n_resample):
            base = gen(pids, T_BASE, temp=0.7)
            ctx = torch.cat([pids, base], dim=-1)

            # Oracle action
            apply_single(action)
            dream = gen(ctx, N_DREAM)
            revert_single()
            post = gen(torch.cat([ctx, dream], dim=-1), 30)
            txt = tok.decode(torch.cat([base[0], dream[0], post[0]]), skip_special_tokens=True)
            oracle_rewards.append(task_reward(txt, r["ents"]))

            # Baseline
            nd = gen(ctx, N_DREAM + 30)
            txt_bl = tok.decode(torch.cat([base[0], nd[0]]), skip_special_tokens=True)
            base_rewards.append(task_reward(txt_bl, r["ents"]))

        delta = np.mean(oracle_rewards) - np.mean(base_rewards)
        robust_deltas.append(delta)

    mean_delta = np.mean(robust_deltas)
    pct_pos = np.mean([d > 0 for d in robust_deltas])
    print(f"  Robust Δ: {mean_delta:+.4f} ± {np.std(robust_deltas):.4f}", flush=True)
    print(f"  Robust improved: {pct_pos:.0%}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    return {"mean_delta": mean_delta, "pct_positive": pct_pos, "deltas": robust_deltas}

# ============================================================
# PHASE 2: TRAIN PREDICTOR (reward-weighted)
# ============================================================
class ActionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HID, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, K), nn.Tanh(),
        )
    def forward(self, h): return self.net(h)

def train_predictor(results, epochs=200, lr=1e-3):
    print(f"\n{'='*60}\nPHASE 2: Training action predictor\n{'='*60}", flush=True)

    # Weight by improvement over baseline (better examples get more weight)
    weights = torch.tensor([max(0.1, r["best_reward"] - r["baseline_reward"]) for r in results])
    weights = weights / weights.sum()

    hiddens = torch.stack([r["hidden"] for r in results]).float().to(DEV)
    actions = torch.stack([r["best_action"] for r in results]).float().to(DEV)
    weights = weights.float().to(DEV)

    predictor = ActionPredictor().to(DEV)
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        pred = predictor(hiddens)
        loss = (weights.unsqueeze(-1) * (pred - actions)**2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1) % 50 == 0 or ep == 0:
            # Check: how close are predictions to targets?
            with torch.no_grad():
                cos = F.cosine_similarity(pred, actions, dim=-1).mean()
            print(f"  Epoch {ep+1}/{epochs}: loss={loss.item():.4f} cos_sim={cos:.4f}", flush=True)

    return predictor

# ============================================================
# PHASE 3: GRPO WITH WARM START
# ============================================================
class DreamPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(HID, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.action_head = nn.Sequential(nn.Linear(128, K), nn.Tanh())
        self.gate_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(K) - 2.0)  # small initial std!

    def forward(self, h):
        feat = self.backbone(h)
        return self.action_head(feat), self.gate_head(feat), self.log_std.exp()

def train_grpo(predictor, n_prompts=500, group_size=8, warmup=50):
    print(f"\n{'='*60}\nPHASE 3: GRPO with warm start ({n_prompts} prompts, G={group_size})\n{'='*60}", flush=True)

    policy = DreamPolicy().to(DEV)
    # Initialize backbone + action head from predictor
    with torch.no_grad():
        policy.backbone[0].weight.copy_(predictor.net[0].weight)
        policy.backbone[0].bias.copy_(predictor.net[0].bias)
        policy.backbone[2].weight.copy_(predictor.net[3].weight)
        policy.backbone[2].bias.copy_(predictor.net[3].bias)
        policy.action_head[0].weight.copy_(predictor.net[6].weight)
        policy.action_head[0].bias.copy_(predictor.net[6].bias)
        # Initialize gate to bias toward dreaming (logit=2 → ~88% dream)
        policy.gate_head.bias.fill_(2.0)

    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    t0 = time.time()
    rewards_log, dream_rate_log = [], []

    for pi in range(n_prompts):
        prompt, ents = make_task()
        pids = encode_batch([prompt])
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()

        mu, gate_logit, std = policy(h)
        gate_prob = torch.sigmoid(gate_logit)

        # Entropy bonus coefficient (annealed)
        ent_coef = max(0.01, 0.1 * (1 - pi / n_prompts))

        group_rewards, group_log_probs, group_dreamed = [], [], []

        for g in range(group_size):
            dream_p = gate_prob.item()
            do_dream = np.random.random() < dream_p

            if do_dream:
                eps = torch.randn(K, device=DEV)
                action = clamp(mu[0] + std * eps)
                log_prob_a = -0.5 * ((action - mu[0]) / (std + 1e-8))**2 - std.log()
                log_prob = log_prob_a.sum() + torch.log(gate_prob + 1e-8).squeeze()

                apply_single(action)
                dream = gen(ctx, N_DREAM)
                revert_single()
                post = gen(torch.cat([ctx, dream], dim=-1), 30)
                full = torch.cat([base, dream, post], dim=-1)
            else:
                log_prob = torch.log(1 - gate_prob + 1e-8).squeeze()
                full = torch.cat([base, gen(ctx, N_DREAM + 30)], dim=-1)

            txt = tok.decode(full[0], skip_special_tokens=True)
            r = task_reward(txt, ents)
            group_rewards.append(r)
            group_log_probs.append(log_prob)
            group_dreamed.append(do_dream)

        rewards_t = torch.tensor(group_rewards, device=DEV)
        advantages = rewards_t - rewards_t.mean()
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(group_log_probs)

        # Policy gradient + entropy bonus
        pg_loss = -(advantages.detach() * log_probs_t).mean()
        # Gate entropy: -p*log(p) - (1-p)*log(1-p)
        gate_ent = -(gate_prob * torch.log(gate_prob + 1e-8) +
                     (1-gate_prob) * torch.log(1-gate_prob + 1e-8))
        loss = pg_loss - ent_coef * gate_ent.mean()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        rewards_log.extend(group_rewards)
        dream_rate_log.append(sum(group_dreamed) / group_size)

        if (pi+1) % 50 == 0:
            recent_r = rewards_log[-400:]
            recent_dr = dream_rate_log[-50:]
            gate_p = gate_prob.item()
            std_val = std.mean().item()
            print(f"  [{pi+1}/{n_prompts}] R={np.mean(recent_r):.4f} "
                  f"dream={np.mean(recent_dr):.2f} gate={gate_p:.2f} "
                  f"std={std_val:.3f} loss={loss.item():.4f}", flush=True)

    print(f"  GRPO done in {time.time()-t0:.1f}s", flush=True)
    return policy, rewards_log, dream_rate_log

# ============================================================
# PHASE 4: EVALUATION
# ============================================================
def eval_cem(n=50, n_cand=64, n_iters=2):
    """Test-time CEM evaluation."""
    print(f"\n{'='*60}\nPHASE 4a: Test-time CEM evaluation (n={n})\n{'='*60}", flush=True)
    t0 = time.time()
    cem_r, base_r = [], []

    for i in range(n):
        prompt, ents = make_task()
        pids = encode_batch([prompt])
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # Baseline
        nd = gen(ctx, N_DREAM + 30)
        txt = tok.decode(torch.cat([base[0], nd[0]]), skip_special_tokens=True)
        base_r.append(task_reward(txt, ents))

        # CEM
        best_a, best_r, _ = cem_search(ctx, base, ents, n_cand=n_cand, n_iters=n_iters)
        cem_r.append(best_r)

    print(f"  Base:  {np.mean(base_r):.4f} ± {np.std(base_r):.4f}", flush=True)
    print(f"  CEM:   {np.mean(cem_r):.4f} ± {np.std(cem_r):.4f}", flush=True)
    print(f"  Δ:     {np.mean(cem_r)-np.mean(base_r):+.4f}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    return {"base": np.mean(base_r), "cem": np.mean(cem_r)}

def eval_policy(policy, n=300):
    print(f"\n{'='*60}\nPHASE 4b: Policy evaluation (n={n})\n{'='*60}", flush=True)
    t0 = time.time()
    base_r, policy_r, random_r = [], [], []
    dream_choices = []

    for i in range(n):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        # Base
        base_gen = gen(pids, 100)
        base_r.append(task_reward(tok.decode(base_gen[0], skip_special_tokens=True), ents))

        # Setup
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # Policy
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            mu, gate_logit, std = policy(h)
            do_dream = torch.sigmoid(gate_logit).item() > 0.5
            action = clamp(mu[0])

        dream_choices.append(1 if do_dream else 0)
        if do_dream:
            apply_single(action)
            dream = gen(ctx, N_DREAM)
            revert_single()
            post = gen(torch.cat([ctx, dream], dim=-1), 30)
            full = torch.cat([base, dream, post], dim=-1)
        else:
            full = torch.cat([base, gen(ctx, N_DREAM + 30)], dim=-1)
        policy_r.append(task_reward(tok.decode(full[0], skip_special_tokens=True), ents))

        # Random
        ra = rand_actions(1)[0]
        apply_single(ra)
        dream = gen(ctx, N_DREAM)
        revert_single()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        full_r = torch.cat([base, dream, post], dim=-1)
        random_r.append(task_reward(tok.decode(full_r[0], skip_special_tokens=True), ents))

    br, pr, rr, dr = np.mean(base_r), np.mean(policy_r), np.mean(random_r), np.mean(dream_choices)
    print(f"\n  Base:     {br:.4f} ± {np.std(base_r):.4f}", flush=True)
    print(f"  Random:   {rr:.4f} ± {np.std(random_r):.4f}", flush=True)
    print(f"  Policy:   {pr:.4f} ± {np.std(policy_r):.4f}", flush=True)
    print(f"  P-Base:   {pr-br:+.4f}", flush=True)
    print(f"  P-Random: {pr-rr:+.4f}", flush=True)
    print(f"  Dream %:  {dr:.1%}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)

    from scipy import stats
    t1, p1 = stats.ttest_ind(policy_r, base_r)
    t2, p2 = stats.ttest_ind(policy_r, random_r)
    print(f"  t-test vs base: t={t1:.3f}, p={p1:.4f}", flush=True)
    print(f"  t-test vs random: t={t2:.3f}, p={p2:.4f}", flush=True)
    return {"base": br, "policy": pr, "random": rr, "dream_rate": dr, "p_vs_base": p1}

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

        # Base
        base_out = gen(pids, 100)
        base_text = tok.decode(base_out[0], skip_special_tokens=True)
        base_found = [e for e in ents if e in base_text.lower()]

        # Policy
        pre = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, pre], dim=-1)
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            mu, gate_logit, std = policy(h)
            do_dream = torch.sigmoid(gate_logit).item() > 0.5
            action = clamp(mu[0])

        if do_dream:
            apply_single(action)
            dream = gen(ctx, N_DREAM)
            revert_single()
            post = gen(torch.cat([ctx, dream], dim=-1), 30)
            policy_text = tok.decode(torch.cat([pre[0], dream[0], post[0]]), skip_special_tokens=True)
        else:
            nd = gen(ctx, N_DREAM + 30)
            policy_text = tok.decode(torch.cat([pre[0], nd[0]]), skip_special_tokens=True)
        policy_found = [e for e in ents if e in policy_text.lower()]

        # CEM
        best_a, best_r, _ = cem_search(ctx, pre, ents, n_cand=64, n_iters=2)
        apply_single(best_a)
        dream = gen(ctx, N_DREAM)
        revert_single()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        cem_text = tok.decode(torch.cat([pre[0], dream[0], post[0]]), skip_special_tokens=True)
        cem_found = [e for e in ents if e in cem_text.lower()]

        print(f"\nTask: [{', '.join(ents)}]", flush=True)
        print(f"  BASE   ({task_reward(base_text, ents):.3f}, {base_found}): {base_text[:180]}", flush=True)
        print(f"  POLICY ({task_reward(policy_text, ents):.3f}, {policy_found}, {'dream' if do_dream else 'no-dream'}): {policy_text[:180]}", flush=True)
        print(f"  CEM    ({task_reward(cem_text, ents):.3f}, {cem_found}): {cem_text[:180]}", flush=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    T0 = time.time()

    # Phase 1: Oracle search
    results = search_oracle(n_tasks=150)

    # Phase 1b: Robustness
    robust = check_robustness(results, n_checks=20, n_resample=8)

    # Phase 2: Train predictor
    predictor = train_predictor(results)

    # Phase 3: GRPO
    policy, rewards_log, dream_rate_log = train_grpo(predictor, n_prompts=500, group_size=8)

    # Phase 4a: CEM eval
    cem_results = eval_cem(n=50)

    # Phase 4b: Policy eval
    policy_results = eval_policy(policy, n=300)

    # Demo
    demo(policy)

    # Save
    os.makedirs("checkpoints4", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints4/policy.pt")
    all_results = {"robust": robust, "cem": cem_results, "policy": policy_results}
    with open("checkpoints4/results.json", "w") as f:
        json.dump({k: v for k, v in all_results.items() if not isinstance(v, dict) or 'deltas' not in v},
                  f, indent=2, default=float)

    total = time.time() - T0
    print(f"\n{'='*60}\nTOTAL: {total:.0f}s ({total/60:.1f}min)\n{'='*60}", flush=True)
