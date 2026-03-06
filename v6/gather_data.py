#!/usr/bin/env python
"""Gather extensive data for the blog post appendix."""
import sys, json
sys.path.insert(0, '.')

# Run pipeline first
exec(open('run.py').read().replace('if __name__ == "__main__":', 'if False:'))

# Run pipeline
total_t0 = time.time()
data = collect()
_nfrozen = True
fwd_m, inv_m = train_models(data)
results = evaluate(fwd_m, inv_m, n=200)

out = {"examples": [], "fwd_analysis": {}, "inv_analysis": {}, "state_stats": {}, "eval": results}

# === 30 DETAILED EXAMPLES ===
print("\n=== Generating 30 detailed examples ===", flush=True)
for i in range(30):
    prompt = np.random.choice(PROMPTS)
    pids = encode_batch([prompt])
    c = ctx_proj(extract_ctx(pids).half())
    base = gen(pids, T_BASE)
    full = torch.cat([pids, base], dim=-1)
    st = extract_states(full)

    # No dream
    nd = gen(full, 60)
    nd_text = tok.decode(nd[0], skip_special_tokens=True)

    # Planned
    a = plan(fwd_m, inv_m, st[0], c[0])
    apply_action(a)
    dream = gen(full, N_DREAM)
    revert()
    dream_text = tok.decode(dream[0], skip_special_tokens=True)
    post_ctx = torch.cat([full, dream], dim=-1)
    post = gen(post_ctx, 30)
    post_text = tok.decode(post[0], skip_special_tokens=True)

    # Random
    ra = rand_actions(1)[0]
    apply_action(ra)
    rd = gen(full, N_DREAM)
    revert()
    rd_text = tok.decode(rd[0], skip_special_tokens=True)
    rpost = gen(torch.cat([full, rd], dim=-1), 30)
    rpost_text = tok.decode(rpost[0], skip_special_tokens=True)

    base_text = tok.decode(base[0], skip_special_tokens=True)
    sp = fwd_m(st[0:1], a.unsqueeze(0), c[0:1])
    obj_b = objective(st[0:1]).item()
    obj_a = objective(sp).item()

    ex = {
        "prompt": prompt, "base": base_text, "nodream": nd_text,
        "dream_planned": dream_text, "post_planned": post_text,
        "dream_random": rd_text, "post_random": rpost_text,
        "action_norm": round(a.norm().item(), 3),
        "obj_before": round(obj_b, 3), "obj_after": round(obj_a, 3),
        "obj_delta": round(obj_a - obj_b, 3),
        "state_entropy": round(st[0,0].item(), 3),
        "state_top1": round(st[0,1].item(), 3),
        "state_rep_risk": round(st[0,9].item(), 3),
        "reward_nodream": round(simple_reward(nd_text), 4),
        "reward_planned": round(simple_reward(dream_text + " " + post_text), 4),
        "reward_random": round(simple_reward(rd_text + " " + rpost_text), 4),
    }
    out["examples"].append(ex)
    print(f"  Example {i+1}/30", flush=True)

# === FORWARD MODEL ANALYSIS ===
print("\n=== Forward model analysis ===", flush=True)
d = {k: v.to(DEV) for k, v in data.items()}
N = len(d['states'])
perm = torch.randperm(N)[:5000]
s, a, sn, q, c2 = d['states'][perm], d['actions'][perm], d['next_states'][perm], d['quality'][perm], d['contexts'][perm]
with torch.no_grad():
    pred = fwd_m(s, a, c2)
mse_per_dim = ((sn - pred)**2).mean(0)
delta_true = sn - s
delta_pred = pred - s
cos_per_sample = torch.nn.functional.cosine_similarity(delta_true, delta_pred, dim=-1)

out["fwd_analysis"] = {
    "mse_per_dim": [round(x, 4) for x in mse_per_dim.tolist()],
    "total_mse": round(mse_per_dim.sum().item(), 4),
    "mean_delta_norm_true": round(delta_true.norm(dim=-1).mean().item(), 4),
    "mean_delta_norm_pred": round(delta_pred.norm(dim=-1).mean().item(), 4),
    "cosine_sim_mean": round(cos_per_sample.mean().item(), 4),
    "cosine_sim_std": round(cos_per_sample.std().item(), 4),
    "cosine_sim_median": round(cos_per_sample.median().item(), 4),
    "r2_per_dim": [round(x, 4) for x in (1 - mse_per_dim / (sn.var(0) + 1e-8)).tolist()],
}

# === INVERSE MODEL ANALYSIS ===
print("=== Inverse model analysis ===", flush=True)
with torch.no_grad():
    pred_a = inv_m(s, sn, c2)
a_mse_per_dim = ((a - pred_a)**2).mean(0)
a_cos = torch.nn.functional.cosine_similarity(a, pred_a, dim=-1)
out["inv_analysis"] = {
    "mse_per_dim": [round(x, 4) for x in a_mse_per_dim.tolist()],
    "total_mse": round(a_mse_per_dim.sum().item(), 4),
    "cosine_sim_mean": round(a_cos.mean().item(), 4),
    "cosine_sim_std": round(a_cos.std().item(), 4),
}

# === STATE DISTRIBUTION ===
print("=== State distribution ===", flush=True)
all_s = d['states']
dim_stats = []
for dim_i in range(64):
    vals = all_s[:, dim_i]
    dim_stats.append({
        "dim": dim_i, "mean": round(vals.mean().item(), 3),
        "std": round(vals.std().item(), 3),
        "min": round(vals.min().item(), 3), "max": round(vals.max().item(), 3),
        "q25": round(vals.quantile(0.25).item(), 3), "q75": round(vals.quantile(0.75).item(), 3),
    })
out["state_stats"] = dim_stats

# === ACTION EFFECT ANALYSIS ===
print("=== Action effect analysis ===", flush=True)
# Test: do larger actions produce larger state changes?
action_norms = a.norm(dim=-1)
delta_norms = delta_true.norm(dim=-1)
correlation = torch.corrcoef(torch.stack([action_norms, delta_norms]))[0,1].item()
out["action_effect"] = {
    "action_norm_vs_delta_norm_corr": round(correlation, 4),
    "mean_action_norm": round(action_norms.mean().item(), 4),
    "std_action_norm": round(action_norms.std().item(), 4),
}

with open("appendix_data.json", "w") as f:
    json.dump(out, f, indent=2)

total = time.time() - total_t0
print(f"\nAll data gathered in {total:.0f}s. Saved to appendix_data.json", flush=True)
