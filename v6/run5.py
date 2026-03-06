#!/usr/bin/env python
"""
DreamerML v6.4 - Learned LoRA modes via entity-specific fine-tuning.

Key insight from ablation: random LoRA perturbations provide ZERO benefit (p=0.87).
All improvement was rejection sampling. Random perturbations don't shift the
output distribution in any meaningful direction.

Fix: LEARN the LoRA bases so each mode steers toward specific content types.
  - Mode 0: Animals (cat, dog, bird, ...)
  - Mode 1: Characters (princess, knight, wizard, ...)
  - Mode 2: Places (river, mountain, castle, ...)
  - Mode 3: Objects (tree, flower, key, crown, ...)

Phase 1: Generate training data (stories from base model, filtered by entity type)
Phase 2: Fine-tune K LoRA modes on entity-specific data (standard LM loss)
Phase 3: GRPO to learn task-conditional mode selection
Phase 4: Evaluate against unperturbed baseline + ablations
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, os, json
from contextlib import contextmanager
from collections import Counter
torch.backends.cudnn.benchmark = True

DEV = "cuda"
DT = torch.float16

K = 4; RANK = 8; ALPHA = 0.3; L2_CLAMP = 2.0
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
print(f"Loaded {sum(p.numel() for p in model.parameters())/1e6:.0f}M params in {time.time()-t0:.1f}s", flush=True)

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
    for e in v["entities"]:
        ALL_ENTITIES.append(e)
        ENTITY_TO_MODE[e] = k

# === LORA MODULES ===
def get_mod(name):
    m = model
    for p in name.split("."): m = getattr(m, p)
    return m

TARGETS = []
for l in range(NL):
    TARGETS.append(f"transformer.h.{l}.attn.attention.out_proj")
    TARGETS.append(f"transformer.h.{l}.mlp.c_proj")

class LoRAMode(nn.Module):
    """Single LoRA mode with learnable A, B matrices."""
    def __init__(self):
        super().__init__()
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for name in TARGETS:
            w = get_mod(name).weight
            o, i = w.shape
            safe = name.replace(".", "_")
            self.A[safe] = nn.Parameter(torch.randn(o, RANK, device=DEV) * (2.0/(o*RANK))**0.5)
            self.B[safe] = nn.Parameter(torch.randn(RANK, i, device=DEV) * (2.0/(i*RANK))**0.5)

    def get_dw(self, name):
        safe = name.replace(".", "_")
        return self.A[safe] @ self.B[safe]  # (out, in)

modes = nn.ModuleList([LoRAMode() for _ in range(K)]).to(DEV)
n_lora_params = sum(p.numel() for p in modes.parameters())
print(f"LoRA modes: K={K}, rank={RANK}, α={ALPHA}, {n_lora_params/1e3:.0f}K learnable params", flush=True)

def clamp(a):
    a = a.clamp(-1, 1)
    if a.dim() == 1:
        n = a.norm(); return a * min(1, L2_CLAMP/n) if n > L2_CLAMP else a
    n = a.norm(dim=-1, keepdim=True); return torch.where(n > L2_CLAMP, a*L2_CLAMP/n, a)

@contextmanager
def apply_mode_mix(action):
    """Apply a linear combination of learned LoRA modes. action: (K,) or (B,K)."""
    a = clamp(action.to(DEV)).float()
    hooks = []
    for name in TARGETS:
        mod = get_mod(name)
        # Combine mode DWs weighted by action
        dws = torch.stack([modes[k].get_dw(name).float() for k in range(K)])  # (K, out, in)
        def make_hook(dws_ref):
            def fn(module, inp, out):
                orig_dtype = out.dtype
                x = inp[0].float()
                if a.dim() == 1:
                    combined_dw = torch.einsum("k,koi->oi", a, dws_ref)
                    correction = F.linear(x, combined_dw) * ALPHA
                else:
                    combined_dw = torch.einsum("bk,koi->boi", a[:x.shape[0]], dws_ref)
                    correction = torch.einsum("boi,bsi->bso", combined_dw, x) * ALPHA
                return (out.float() + correction).to(orig_dtype)
            return fn
        hooks.append(mod.register_forward_hook(make_hook(dws)))
    try: yield
    finally:
        for h in hooks: h.remove()

@contextmanager
def apply_single_mode(k):
    """Apply a single mode's LoRA for fine-tuning."""
    hooks = []
    for name in TARGETS:
        mod = get_mod(name)
        dw = modes[k].get_dw(name)  # differentiable!
        def make_hook(dw_ref):
            def fn(module, inp, out):
                x = inp[0]
                correction = F.linear(x.float(), dw_ref.float()) * ALPHA
                return out + correction.to(out.dtype)
            return fn
        hooks.append(mod.register_forward_hook(make_hook(dw)))
    try: yield
    finally:
        for h in hooks: h.remove()

_orig = {}
def apply_action_weights(action):
    """Modify weights directly (for generation with single action)."""
    a = clamp(action.to(DEV)).float()
    for name in TARGETS:
        mod = get_mod(name)
        _orig[name] = mod.weight.data.clone()
        dws = torch.stack([modes[k].get_dw(name).float().detach() for k in range(K)])
        combined = torch.einsum("k,koi->oi", a, dws)
        mod.weight.data += (ALPHA * combined).to(DT)

def revert_weights():
    for name, w in _orig.items(): get_mod(name).weight.data.copy_(w)
    _orig.clear()

@torch.no_grad()
def gen(ids, n, temp=TEMP):
    mask = (ids != PAD).long()
    out = model.generate(ids, attention_mask=mask, max_new_tokens=n, min_new_tokens=n,
                         do_sample=True, temperature=temp, top_p=TOP_P, pad_token_id=PAD)
    return out[:, ids.shape[1]:]

def encode_batch(texts):
    e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    return e["input_ids"].to(DEV)

# === TASK + REWARD ===
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

# ============================================================
# PHASE 1: GENERATE TRAINING DATA
# ============================================================
STORY_PROMPTS = [
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
    "Write a story about a cat and a dog.",
    "Write a story about a princess in a castle.",
    "Write a story about a wizard and a dragon.",
    "Write a story about a knight in a forest.",
    "Write a story about a bear and a river.",
    "Write a story about a fairy with a magic key.",
    "Write a story about a fox and a flower.",
    "Write a story about a pirate on a boat.",
    "Write a story about a bird on a mountain.",
    "Write a story about a frog in a garden.",
]

def generate_training_data(n_stories=2000, min_per_mode=200):
    """Generate stories and categorize by which entities they contain."""
    print(f"\n{'='*60}\nPHASE 1: Generating training data ({n_stories} stories)\n{'='*60}", flush=True)
    t0 = time.time()

    mode_data = {k: [] for k in range(K)}  # mode -> list of token_ids
    total = 0

    while min(len(v) for v in mode_data.values()) < min_per_mode or total < n_stories:
        if total > n_stories * 3:
            print(f"  Warning: exceeded 3x budget, stopping", flush=True)
            break

        prompts = [np.random.choice(STORY_PROMPTS) for _ in range(BATCH)]
        pids = encode_batch(prompts)

        with torch.no_grad():
            cont = gen(pids, 100)

        for b in range(len(prompts)):
            full_ids = torch.cat([pids[b], cont[b]])
            text = tok.decode(full_ids, skip_special_tokens=True).lower()

            # Check which entity categories are present
            for mode_k in range(K):
                cat_ents = ENTITY_CATEGORIES[mode_k]["entities"]
                if any(e in text for e in cat_ents):
                    mode_data[mode_k].append(full_ids.clone())

        total += len(prompts)

        if total % (BATCH * 5) < BATCH:
            counts = {k: len(v) for k, v in mode_data.items()}
            print(f"  Generated {total} stories. Per-mode: {counts}", flush=True)

    for k in range(K):
        mode_data[k] = mode_data[k][:max(min_per_mode, len(mode_data[k]))]

    counts = {ENTITY_CATEGORIES[k]["name"]: len(v) for k, v in mode_data.items()}
    print(f"  Final dataset: {counts}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    return mode_data

# ============================================================
# PHASE 2: FINE-TUNE LORA MODES
# ============================================================
def finetune_modes(mode_data, epochs=30, lr=3e-4, bs=16):
    """Fine-tune each LoRA mode on its entity-specific training data."""
    print(f"\n{'='*60}\nPHASE 2: Fine-tuning LoRA modes ({epochs} epochs, lr={lr})\n{'='*60}", flush=True)
    t0 = time.time()

    # Cast model to float32 for training
    model.float()

    for k in range(K):
        cat_name = ENTITY_CATEGORIES[k]["name"]
        data = mode_data[k]
        if len(data) < 10:
            print(f"  Mode {k} ({cat_name}): skipping, only {len(data)} examples", flush=True)
            continue

        # Pad sequences to same length
        max_len = max(d.shape[0] for d in data)
        padded = torch.full((len(data), max_len), PAD, dtype=torch.long, device=DEV)
        for i, d in enumerate(data):
            padded[i, max_len-len(d):] = d  # left-pad

        opt = torch.optim.AdamW(modes[k].parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

        for ep in range(epochs):
            perm = torch.randperm(len(data))
            total_loss = 0; n_batches = 0

            for j in range(0, len(data), bs):
                idx = perm[j:j+bs]
                batch = padded[idx]
                mask = (batch != PAD).long()

                labels = batch.clone()
                labels[labels == PAD] = -100

                with apply_single_mode(k):
                    outputs = model(batch, attention_mask=mask, labels=labels)
                    loss = outputs.loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modes[k].parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                n_batches += 1

            sched.step()

            if (ep+1) % 10 == 0 or ep == 0:
                print(f"  Mode {k} ({cat_name}) epoch {ep+1}/{epochs}: loss={total_loss/n_batches:.4f}", flush=True)

    # Cast back to fp16 for inference
    model.half()
    print(f"  Fine-tuning done in {time.time()-t0:.0f}s", flush=True)

# ============================================================
# PHASE 2B: VERIFY MODES WORK
# ============================================================
def verify_modes(n=100):
    """Check that each mode increases generation of its target entities."""
    print(f"\n{'='*60}\nPHASE 2B: Verifying mode specialization (n={n})\n{'='*60}", flush=True)
    t0 = time.time()

    prompts = ["Once upon a time,"] * n
    pids = encode_batch(prompts[:BATCH])

    results = {}
    for k in range(-1, K):  # -1 = base model
        entity_counts = {kk: 0 for kk in range(K)}

        for start in range(0, n, BATCH):
            B = min(BATCH, n - start)
            pids_b = encode_batch(["Once upon a time,"] * B)

            if k == -1:
                out = gen(pids_b, 100)
            else:
                action = torch.zeros(K, device=DEV)
                action[k] = 1.0
                apply_action_weights(action)
                out = gen(pids_b, 100)
                revert_weights()

            for b in range(B):
                txt = tok.decode(out[b], skip_special_tokens=True).lower()
                for kk in range(K):
                    if any(e in txt for e in ENTITY_CATEGORIES[kk]["entities"]):
                        entity_counts[kk] += 1

        label = "BASE" if k == -1 else f"Mode {k} ({ENTITY_CATEGORIES[k]['name']})"
        counts_str = " ".join(f"{ENTITY_CATEGORIES[kk]['name']}={entity_counts[kk]/n:.0%}" for kk in range(K))
        results[k] = entity_counts
        print(f"  {label}: {counts_str}", flush=True)

    # Check if each mode increases its target category
    base_counts = results[-1]
    for k in range(K):
        target = ENTITY_CATEGORIES[k]["name"]
        base_rate = base_counts[k] / n
        mode_rate = results[k][k] / n
        delta = mode_rate - base_rate
        status = "✓" if delta > 0.05 else "✗" if delta < -0.05 else "~"
        print(f"  Mode {k} ({target}): {base_rate:.0%} → {mode_rate:.0%} ({delta:+.0%}) {status}", flush=True)

    print(f"  Time: {time.time()-t0:.0f}s", flush=True)
    return results

# ============================================================
# PHASE 3: GRPO - LEARN MODE SELECTION POLICY
# ============================================================
class ModePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HID, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        # Output: K mode weights + 1 "strength" parameter
        self.mode_head = nn.Linear(128, K)
        self.strength_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.log_std = nn.Parameter(torch.zeros(K) - 1.5)

    def forward(self, h):
        feat = self.net(h)
        mode_logits = self.mode_head(feat)  # (B, K)
        strength = self.strength_head(feat)  # (B, 1)
        return mode_logits, strength, self.log_std.exp()

def train_grpo(n_prompts=600, group_size=8):
    print(f"\n{'='*60}\nPHASE 3: GRPO training ({n_prompts} prompts, G={group_size})\n{'='*60}", flush=True)

    policy = ModePolicy().to(DEV)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
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

        mode_logits, strength, std = policy(h)
        mode_probs = F.softmax(mode_logits, dim=-1)  # (1, K)
        strength_val = strength.item()

        ent_coef = max(0.01, 0.1 * (1 - pi / n_prompts))
        group_rewards, group_log_probs, group_dreamed = [], [], []

        for g in range(group_size):
            # Sample action: choose mode weights from softmax + noise
            eps = torch.randn(K, device=DEV)
            action_raw = mode_probs[0] + std * eps  # (K,)
            action = clamp(action_raw * strength_val * 2)  # scale by strength

            # Log probability
            log_prob_a = -0.5 * ((action_raw - mode_probs[0]) / (std + 1e-8))**2 - std.log()
            log_prob = log_prob_a.sum()

            # Generate with perturbation
            apply_action_weights(action)
            dream = gen(ctx, N_DREAM)
            revert_weights()
            post = gen(torch.cat([ctx, dream], dim=-1), 30)
            full = torch.cat([base, dream, post], dim=-1)

            txt = tok.decode(full[0], skip_special_tokens=True)
            r = task_reward(txt, ents)
            group_rewards.append(r)
            group_log_probs.append(log_prob)
            group_dreamed.append(True)

        # Also include a no-dream sample for comparison
        nd = gen(ctx, N_DREAM + 30)
        nd_txt = tok.decode(torch.cat([base[0], nd[0]]), skip_special_tokens=True)
        nd_r = task_reward(nd_txt, ents)

        rewards_t = torch.tensor(group_rewards, device=DEV)
        advantages = rewards_t - rewards_t.mean()
        if advantages.std() > 1e-6:
            advantages = advantages / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(group_log_probs)
        pg_loss = -(advantages.detach() * log_probs_t).mean()
        entropy = -(mode_probs * torch.log(mode_probs + 1e-8)).sum()
        loss = pg_loss - ent_coef * entropy

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        rewards_log.extend(group_rewards)
        rewards_log.append(nd_r)
        dream_rate_log.append(np.mean(group_rewards) - nd_r)

        if (pi+1) % 50 == 0:
            recent_r = rewards_log[-400:]
            recent_delta = dream_rate_log[-50:]
            print(f"  [{pi+1}/{n_prompts}] R={np.mean(recent_r):.4f} "
                  f"dream_advantage={np.mean(recent_delta):+.4f} "
                  f"strength={strength_val:.2f} loss={loss.item():.4f}", flush=True)

    print(f"  GRPO done in {time.time()-t0:.0f}s", flush=True)
    return policy

# ============================================================
# PHASE 4: EVALUATION
# ============================================================
def evaluate(policy, n=300):
    print(f"\n{'='*60}\nPHASE 4: Evaluation (n={n})\n{'='*60}", flush=True)
    t0 = time.time()
    base_r, policy_r, oracle_r, random_r = [], [], [], []

    for i in range(n):
        prompt, ents = make_task()
        pids = encode_batch([prompt])

        # Base model
        base_gen = gen(pids, 100)
        base_r.append(task_reward(tok.decode(base_gen[0], skip_special_tokens=True), ents))

        # Shared setup
        base = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, base], dim=-1)

        # Policy
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            mode_logits, strength, std = policy(h)
            mode_probs = F.softmax(mode_logits, dim=-1)[0]
            action = clamp(mode_probs * strength.item() * 2)

        apply_action_weights(action)
        dream = gen(ctx, N_DREAM)
        revert_weights()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        full_p = torch.cat([base, dream, post], dim=-1)
        policy_r.append(task_reward(tok.decode(full_p[0], skip_special_tokens=True), ents))

        # Oracle: use known entity-mode mapping
        oracle_action = torch.zeros(K, device=DEV)
        for e in ents:
            if e in ENTITY_TO_MODE:
                oracle_action[ENTITY_TO_MODE[e]] = 0.5
        oracle_action = clamp(oracle_action)
        apply_action_weights(oracle_action)
        dream_o = gen(ctx, N_DREAM)
        revert_weights()
        post_o = gen(torch.cat([ctx, dream_o], dim=-1), 30)
        full_o = torch.cat([base, dream_o, post_o], dim=-1)
        oracle_r.append(task_reward(tok.decode(full_o[0], skip_special_tokens=True), ents))

        # Random mode
        rand_action = torch.randn(K, device=DEV) * 0.3
        rand_action = clamp(rand_action)
        apply_action_weights(rand_action)
        dream_r = gen(ctx, N_DREAM)
        revert_weights()
        post_r = gen(torch.cat([ctx, dream_r], dim=-1), 30)
        full_rr = torch.cat([base, dream_r, post_r], dim=-1)
        random_r.append(task_reward(tok.decode(full_rr[0], skip_special_tokens=True), ents))

    br, pr, orr, rr = np.mean(base_r), np.mean(policy_r), np.mean(oracle_r), np.mean(random_r)
    print(f"\n  Base:        {br:.4f} ± {np.std(base_r):.4f}", flush=True)
    print(f"  Random mode: {rr:.4f} ± {np.std(random_r):.4f}", flush=True)
    print(f"  Oracle mode: {orr:.4f} ± {np.std(oracle_r):.4f}", flush=True)
    print(f"  GRPO policy: {pr:.4f} ± {np.std(policy_r):.4f}", flush=True)
    print(f"  Policy-Base:   {pr-br:+.4f}", flush=True)
    print(f"  Oracle-Base:   {orr-br:+.4f}", flush=True)
    print(f"  Time: {time.time()-t0:.0f}s", flush=True)

    from scipy import stats
    t1, p1 = stats.ttest_ind(policy_r, base_r)
    t2, p2 = stats.ttest_ind(oracle_r, base_r)
    print(f"  t-test policy vs base: t={t1:.3f}, p={p1:.4f}", flush=True)
    print(f"  t-test oracle vs base: t={t2:.3f}, p={p2:.4f}", flush=True)

    return {"base": br, "policy": pr, "oracle": orr, "random": rr,
            "p_policy_base": p1, "p_oracle_base": p2}

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
        base_out = gen(pids, 100)
        base_text = tok.decode(base_out[0], skip_special_tokens=True)

        pre = gen(pids, T_BASE, temp=0.7)
        ctx = torch.cat([pids, pre], dim=-1)

        # Policy
        with torch.no_grad():
            out = model(ctx, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :].float()
            mode_logits, strength, std = policy(h)
            mode_probs = F.softmax(mode_logits, dim=-1)[0]
            action = clamp(mode_probs * strength.item() * 2)

        apply_action_weights(action)
        dream = gen(ctx, N_DREAM)
        revert_weights()
        post = gen(torch.cat([ctx, dream], dim=-1), 30)
        policy_text = tok.decode(torch.cat([pre[0], dream[0], post[0]]), skip_special_tokens=True)

        # Oracle
        oracle_action = torch.zeros(K, device=DEV)
        for e in ents:
            if e in ENTITY_TO_MODE: oracle_action[ENTITY_TO_MODE[e]] = 0.5
        apply_action_weights(clamp(oracle_action))
        dream_o = gen(ctx, N_DREAM)
        revert_weights()
        post_o = gen(torch.cat([ctx, dream_o], dim=-1), 30)
        oracle_text = tok.decode(torch.cat([pre[0], dream_o[0], post_o[0]]), skip_special_tokens=True)

        base_found = [e for e in ents if e in base_text.lower()]
        policy_found = [e for e in ents if e in policy_text.lower()]
        oracle_found = [e for e in ents if e in oracle_text.lower()]

        probs_str = " ".join(f"{ENTITY_CATEGORIES[k]['name'][:3]}={mode_probs[k]:.2f}" for k in range(K))
        print(f"\nTask: [{', '.join(ents)}] | Policy modes: [{probs_str}] str={strength.item():.2f}", flush=True)
        print(f"  BASE   ({task_reward(base_text, ents):.3f}, {base_found}): {base_text[:180]}", flush=True)
        print(f"  POLICY ({task_reward(policy_text, ents):.3f}, {policy_found}): {policy_text[:180]}", flush=True)
        print(f"  ORACLE ({task_reward(oracle_text, ents):.3f}, {oracle_found}): {oracle_text[:180]}", flush=True)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    T0 = time.time()

    # Phase 1: Training data
    mode_data = generate_training_data(n_stories=2000, min_per_mode=200)

    # Phase 2: Fine-tune modes
    finetune_modes(mode_data, epochs=30, lr=3e-4)

    # Phase 2b: Verify
    verify_modes(n=100)

    # Phase 3: GRPO
    policy = train_grpo(n_prompts=600, group_size=8)

    # Phase 4: Evaluate
    eval_results = evaluate(policy, n=300)

    # Demo
    demo(policy)

    # Save
    os.makedirs("checkpoints5", exist_ok=True)
    torch.save(modes.state_dict(), "checkpoints5/modes.pt")
    torch.save(policy.state_dict(), "checkpoints5/policy.pt")
    with open("checkpoints5/results.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=float)

    total = time.time() - T0
    print(f"\n{'='*60}\nTOTAL: {total:.0f}s ({total/60:.1f}min)\n{'='*60}", flush=True)
