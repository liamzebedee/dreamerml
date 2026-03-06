#!/usr/bin/env python
"""
DreamerML v6.2 - Direct reward-driven approach.

Key insight from v6.1 failure: random LoRA bases are random noise.
No forward model or policy on top of random bases will help.

New approach:
  Phase 1: For many tasks, search over actions to find which perturbations
           actually improve reward (best-of-N search). This directly proves
           whether targeted perturbation CAN improve task performance.
  Phase 2: Train a policy to predict good actions from prompt context.
  Phase 3: GRPO refinement of the policy.
  Phase 4: Evaluate.

Changes from v6.1:
  - α=0.5 (was 0.2): much stronger perturbations
  - K=4 (was 8): fewer modes, larger per-mode effect
  - Rank-8 (was 4): each mode is more expressive
  - Skip forward model entirely - directly measure reward
  - Best-of-N action search as the oracle
  - GRPO trains on direct generation reward, not model predictions
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, json
from contextlib import contextmanager
from collections import Counter
torch.backends.cudnn.benchmark = True

DEV = "cuda"
DT = torch.float16

# === CONFIG ===
K = 4           # dream modes (fewer = more impact per mode)
RANK = 8        # LoRA rank per mode (more expressive)
ALPHA = 0.5     # perturbation strength (was 0.2)
L2_CLAMP = 2.0
N_DREAM = 50    # dream tokens (longer to show more effect)
T_BASE = 20     # base tokens before dream
TEMP = 0.9
TOP_P = 0.95
BATCH = 64      # smaller batch for memory with larger rank

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

# === RANK-8 LORA BASIS ===
def get_targets():
    """Target ALL layers for maximum perturbation impact."""
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
    DW[name] = torch.bmm(A, B)  # (K, out, in)

print(f"LoRA: K={K}, rank={RANK}, α={ALPHA}, {len(TARGETS)} targets across all {NL} layers", flush=True)

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

# === ENTITY TASKS ===
ENTITIES = ["cat","dog","bird","fish","rabbit","bear","mouse","fox","turtle","frog",
            "princess","knight","wizard","dragon","fairy","queen","king","pirate",
            "tree","flower","river","mountain","castle","garden","forest","cave",
            "boat","lamp","key","crown","sword","shield","map","book"]

def make_task():
    ents = list(np.random.choice(ENTITIES, 3, replace=False))
    prompt = f"Write a story that includes a {ents[0]}, a {ents[1]}, and a {ents[2]}."
    return prompt, ents

def task_reward(text, required_ents):
    tl = text.lower()
    words = tl.split()
    if len(words) < 5: return 0.05

    # Entity coverage (primary signal - 50%)
    found = sum(1 for e in required_ents if e in tl) / len(required_ents)

    # 4-gram diversity (20%)
    ng = [tuple(words[i:i+4]) for i in range(len(words)-3)]
    uniq = len(set(ng)) / max(1, len(ng)) if ng else 0

    # Vocab diversity (10%)
    vdiv = len(set(words)) / len(words)

    # Length bonus (10%)
    lbon = min(1.0, len(words) / 50)

    # Coherence: penalize repetition (10%)
    rep_penalty = 0
    if ng:
        ngc = Counter(ng)
        most_common_frac = ngc.most_common(1)[0][1] / len(ng)
        if most_common_frac > 0.12: rep_penalty = 0.5
        if most_common_frac > 0.25: rep_penalty = 1.0

    return float(np.clip(0.50*found + 0.20*uniq + 0.10*vdiv + 0.10*lbon + 0.10*(1-rep_penalty), 0, 1))

# === PROMPTS ===
def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return e["input_ids"].to(DEV)

def rand_actions(B):
    a = torch.empty(B, K, device=DEV, dtype=DT)
    h = B // 2
    a[:h] = torch.rand(h, K, device=DEV, dtype=DT) * 2 - 1
    a[h:] = torch.tanh(torch.randn(B-h, K, device=DEV, dtype=DT) * 0.5)
    return clamp(a)

# ============================================================
# PHASE 1: BEST-OF-N ACTION SEARCH
# ============================================================
def search_best_actions(n_tasks=80, n_candidates=256):
    """For each task, find the action that maximizes reward."""
    print(f"\n{'='*60}\nPHASE 1: Best-of-N action search ({n_tasks} tasks, N={n_candidates})\n{'='*60}", flush=True)
    t0 = time.time()

    results = []
    baseline_rewards = []
    best_rewards = []
    random_rewards = []

    for ti in range(n_tasks):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        # Generate base text (same for all candidates to make comparison fair)
        # Use deterministic base for fairness
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # === BASELINE: No perturbation ===
        # Generate 4 samples, take mean reward (reduce variance)
        bl_rewards = []
        for _ in range(4):
            nd = gen(ctx, N_DREAM + 30)
            txt = tok.decode(torch.cat([base[0], nd[0]]), skip_special_tokens=True)
            bl_rewards.append(task_reward(txt, ents))
        bl_r = np.mean(bl_rewards)
        baseline_rewards.append(bl_r)

        # === SEARCH: Try n_candidates random actions ===
        all_actions = []
        all_rewards_for_task = []
        for start in range(0, n_candidates, BATCH):
            B = min(BATCH, n_candidates - start)
            actions = rand_actions(B)
            all_actions.append(actions)

            # Expand context for batch
            ctx_exp = ctx.expand(B, -1).contiguous()

            # Dream with perturbation
            with perturb(actions):
                dream = gen(ctx_exp, N_DREAM)

            # Post-dream continuation (unperturbed)
            dream_ctx = torch.cat([ctx_exp, dream], dim=-1)
            post = gen(dream_ctx, 30)

            # Score each
            for b in range(B):
                txt = tok.decode(torch.cat([base[0], dream[b], post[b]]), skip_special_tokens=True)
                r = task_reward(txt, ents)
                all_rewards_for_task.append(r)

        all_actions = torch.cat(all_actions, dim=0)[:n_candidates]
        all_rewards_arr = np.array(all_rewards_for_task[:n_candidates])

        # Best action
        best_idx = all_rewards_arr.argmax()
        best_r = all_rewards_arr[best_idx]
        best_a = all_actions[best_idx]
        best_rewards.append(best_r)

        # Random action (mean of all)
        random_rewards.append(all_rewards_arr.mean())

        # Store
        # Get prompt hidden for policy training
        with torch.no_grad():
            out = model(pids, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float().cpu()

        results.append({
            "hidden": h[0],
            "best_action": best_a.cpu(),
            "best_reward": best_r,
            "baseline_reward": bl_r,
            "all_rewards": all_rewards_arr,
            "prompt": prompt,
            "ents": ents,
        })

        if (ti+1) % 10 == 0:
            el = time.time()-t0
            bl_m = np.mean(baseline_rewards)
            br_m = np.mean(best_rewards)
            rr_m = np.mean(random_rewards)
            pct_improved = np.mean([r["best_reward"] > r["baseline_reward"] for r in results])
            print(f"  [{ti+1}/{n_tasks}] baseline={bl_m:.4f} best={br_m:.4f} random={rr_m:.4f} "
                  f"improved={pct_improved:.0%} time={el:.0f}s", flush=True)

    bl_m = np.mean(baseline_rewards)
    br_m = np.mean(best_rewards)
    rr_m = np.mean(random_rewards)
    deltas = [r["best_reward"] - r["baseline_reward"] for r in results]
    pct_imp = np.mean([d > 0 for d in deltas])

    print(f"\n  SUMMARY:", flush=True)
    print(f"  Baseline (no dream):  {bl_m:.4f} ± {np.std(baseline_rewards):.4f}", flush=True)
    print(f"  Random action (mean): {rr_m:.4f} ± {np.std(random_rewards):.4f}", flush=True)
    print(f"  Best-of-{n_candidates}:        {br_m:.4f} ± {np.std(best_rewards):.4f}", flush=True)
    print(f"  Improvement (best-baseline): {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}", flush=True)
    print(f"  Tasks where best > baseline: {pct_imp:.0%}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    return results, {"baseline": bl_m, "random": rr_m, "best": br_m,
                     "improvement": np.mean(deltas), "pct_improved": pct_imp}

# ============================================================
# PHASE 2: TRAIN ACTION PREDICTOR
# ============================================================
class ActionPredictor(nn.Module):
    """Predict good action from prompt hidden state."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HID, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, K),
            nn.Tanh(),
        )
    def forward(self, h):
        return self.net(h)

def train_predictor(results, epochs=100, lr=1e-3):
    print(f"\n{'='*60}\nPHASE 2: Training action predictor\n{'='*60}", flush=True)

    # Only use tasks where best action improved over baseline
    good = [r for r in results if r["best_reward"] > r["baseline_reward"] + 0.01]
    print(f"  Using {len(good)}/{len(results)} tasks where dream helped", flush=True)
    if len(good) < 10:
        print("  WARNING: Too few tasks where dream helped. Proceeding anyway.", flush=True)
        good = results  # fall back to all

    hiddens = torch.stack([r["hidden"] for r in good]).float().to(DEV)
    actions = torch.stack([r["best_action"] for r in good]).float().to(DEV)

    predictor = ActionPredictor().to(DEV)
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        pred = predictor(hiddens)
        loss = F.mse_loss(pred, actions)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (ep+1) % 25 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/{epochs}: loss={loss.item():.4f}", flush=True)

    return predictor

# ============================================================
# PHASE 3: GRPO REFINEMENT
# ============================================================
class DreamPolicy(nn.Module):
    """Policy that outputs continuous action or 'no dream' decision."""
    def __init__(self):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(HID, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # Action head: mean of action distribution
        self.action_head = nn.Sequential(nn.Linear(128, K), nn.Tanh())
        # Dream gate: probability of dreaming
        self.gate_head = nn.Linear(128, 1)
        # Action log-std (learnable)
        self.log_std = nn.Parameter(torch.zeros(K) - 0.5)

    def forward(self, h):
        feat = self.backbone(h)
        mu = self.action_head(feat)
        gate_logit = self.gate_head(feat)
        return mu, gate_logit, self.log_std.exp()

def train_grpo(predictor, n_prompts=400, group_size=8):
    print(f"\n{'='*60}\nPHASE 3: GRPO refinement ({n_prompts} prompts, G={group_size})\n{'='*60}", flush=True)

    policy = DreamPolicy().to(DEV)
    # Initialize from predictor
    with torch.no_grad():
        policy.backbone[0].weight.copy_(predictor.net[0].weight)
        policy.backbone[0].bias.copy_(predictor.net[0].bias)

    opt = torch.optim.Adam(policy.parameters(), lr=2e-4)
    t0 = time.time()
    rewards_log = []
    dream_rate_log = []

    for pi in range(n_prompts):
        prompt, ents = make_task()
        pids = encode_batch([prompt])
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # Get hidden state
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()

        # Policy output
        mu, gate_logit, std = policy(h)
        gate_prob = torch.sigmoid(gate_logit)

        group_rewards = []
        group_log_probs = []
        group_dreamed = []

        for g in range(group_size):
            # Sample dream decision
            dream_p = gate_prob.item()
            do_dream = np.random.random() < dream_p

            if do_dream:
                # Sample action from Gaussian
                eps = torch.randn(K, device=DEV)
                action = clamp(mu[0] + std * eps)
                log_prob_action = -0.5 * ((action - mu[0]) / (std + 1e-8))**2 - std.log()
                log_prob_action = log_prob_action.sum()
                log_prob_gate = torch.log(gate_prob + 1e-8)
                log_prob = log_prob_action + log_prob_gate.squeeze()

                apply_single(action)
                dream = gen(ctx, N_DREAM)
                revert_single()

                dream_ctx = torch.cat([ctx, dream], dim=-1)
                post = gen(dream_ctx, 30)
                full = torch.cat([base, dream, post], dim=-1)
            else:
                log_prob = torch.log(1 - gate_prob + 1e-8).squeeze()
                full = torch.cat([base, gen(ctx, N_DREAM + 30)], dim=-1)

            txt = tok.decode(full[0], skip_special_tokens=True)
            r = task_reward(txt, ents)
            group_rewards.append(r)
            group_log_probs.append(log_prob)
            group_dreamed.append(do_dream)

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
            recent_r = rewards_log[-400:]
            recent_dr = dream_rate_log[-50:]
            print(f"  [{pi+1}/{n_prompts}] reward={np.mean(recent_r):.4f} "
                  f"dream_rate={np.mean(recent_dr):.2f} loss={loss.item():.4f}", flush=True)

    print(f"  GRPO done in {time.time()-t0:.1f}s", flush=True)
    return policy, rewards_log, dream_rate_log

# ============================================================
# PHASE 4: EVALUATION
# ============================================================
def evaluate(policy, n=300):
    print(f"\n{'='*60}\nPHASE 4: Evaluation (n={n})\n{'='*60}", flush=True)
    t0 = time.time()

    base_r, policy_r, random_r, always_dream_r = [], [], [], []
    dream_choices = []

    for i in range(n):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        # === BASE: no perturbation, no dream ===
        base_gen = gen(pids, 100)
        base_txt = tok.decode(base_gen[0], skip_special_tokens=True)
        base_r.append(task_reward(base_txt, ents))

        # === Shared setup for dream conditions ===
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # === POLICY: use trained policy ===
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

        txt = tok.decode(full[0], skip_special_tokens=True)
        policy_r.append(task_reward(txt, ents))

        # === RANDOM: random action, always dream ===
        ra = rand_actions(1)[0]
        apply_single(ra)
        dream = gen(ctx, N_DREAM)
        revert_single()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        full_r = torch.cat([base, dream, post], dim=-1)
        txt_r = tok.decode(full_r[0], skip_special_tokens=True)
        random_r.append(task_reward(txt_r, ents))

    br = np.mean(base_r)
    pr = np.mean(policy_r)
    rr = np.mean(random_r)
    dr = np.mean(dream_choices)

    print(f"\n  Base model:     {br:.4f} ± {np.std(base_r):.4f}", flush=True)
    print(f"  Random dream:   {rr:.4f} ± {np.std(random_r):.4f}", flush=True)
    print(f"  GRPO policy:    {pr:.4f} ± {np.std(policy_r):.4f}", flush=True)
    print(f"  Policy-Base:    {pr-br:+.4f}", flush=True)
    print(f"  Policy-Random:  {pr-rr:+.4f}", flush=True)
    print(f"  Dream rate:     {dr:.1%}", flush=True)
    print(f"  Time: {time.time()-t0:.1f}s", flush=True)

    # Statistical significance
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(policy_r, base_r)
    print(f"  t-test policy vs base: t={t_stat:.3f}, p={p_val:.4f}", flush=True)
    t_stat2, p_val2 = stats.ttest_ind(policy_r, random_r)
    print(f"  t-test policy vs random: t={t_stat2:.3f}, p={p_val2:.4f}", flush=True)

    return {"base": br, "policy": pr, "random": rr, "dream_rate": dr,
            "p_vs_base": p_val, "p_vs_random": p_val2}

# ============================================================
# DEMO
# ============================================================
def demo(policy, n=8):
    print(f"\n{'='*60}\nDEMO\n{'='*60}", flush=True)
    demo_tasks = [
        ("Write a story that includes a cat, a castle, and a river.", ["cat", "castle", "river"]),
        ("Write a story that includes a dragon, a flower, and a cave.", ["dragon", "flower", "cave"]),
        ("Write a story that includes a knight, a mouse, and a forest.", ["knight", "mouse", "forest"]),
        ("Write a story that includes a princess, a key, and a garden.", ["princess", "key", "garden"]),
        ("Write a story that includes a bear, a boat, and a mountain.", ["bear", "boat", "mountain"]),
        ("Write a story that includes a wizard, a bird, and a lamp.", ["wizard", "bird", "lamp"]),
        ("Write a story that includes a fox, a crown, and a tree.", ["fox", "crown", "tree"]),
        ("Write a story that includes a fairy, a sword, and a frog.", ["fairy", "sword", "frog"]),
    ][:n]

    for prompt, ents in demo_tasks:
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

        base_rew = task_reward(base_text, ents)
        policy_rew = task_reward(policy_text, ents)

        print(f"\nTask: include [{', '.join(ents)}]", flush=True)
        print(f"  BASE ({base_rew:.3f}, found {base_found}): {base_text[:200]}", flush=True)
        print(f"  POLICY ({policy_rew:.3f}, found {policy_found}, dream={'Y' if do_dream else 'N'}): {policy_text[:200]}", flush=True)
        print(f"  Action: [{', '.join(f'{a:.2f}' for a in action.tolist())}]", flush=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    T0 = time.time()

    # Phase 1: Search for best actions
    results, search_stats = search_best_actions(n_tasks=80, n_candidates=256)

    # Phase 2: Train action predictor
    predictor = train_predictor(results)

    # Phase 3: GRPO
    policy, rewards_log, dream_rate_log = train_grpo(predictor, n_prompts=400, group_size=8)

    # Phase 4: Evaluate
    eval_results = evaluate(policy, n=300)

    # Demo
    demo(policy)

    # Save
    os.makedirs("checkpoints3", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints3/policy.pt")
    all_results = {"search": search_stats, "eval": eval_results}
    with open("checkpoints3/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    total = time.time() - T0
    print(f"\n{'='*60}\nTOTAL: {total:.0f}s ({total/60:.1f}min)\n{'='*60}", flush=True)
