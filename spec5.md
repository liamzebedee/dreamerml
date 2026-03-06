spec.md

Objective: learn a forward/inverse dynamics model over a TinyStories base LLM’s local “weight-perturbation manifold”, then use inverse dynamics to choose weightmods that reliably move the model into alternative cognitive regimes (“dream states”) that produce completions the unperturbed model rarely reaches. Emphasis: fastest possible inference, minimal components, fully decided design.

0. Base environment

Base LLM: TinyStories-trained decoder-only transformer (GPT-2 class), choose the smallest that still produces coherent stories; target ~110M parameters (the common “small” size). All base weights W are frozen forever.

Tokenization: whatever the base model uses; no changes.

Inference constraint: everything must run on a single GPU; dream rollouts dominate compute.

1. Latent weight action space (LoRA basis)

We do not learn edits directly in full weight space. We define a small continuous action vector a in R^K that parameterizes temporary weight perturbations through a fixed LoRA basis.

K = 32 directions.

Each direction k defines a rank-1 LoRA update for selected linear layers: ΔW_k = A_k B_k with A_k ∈ R^{out×1}, B_k ∈ R^{1×in}.

Targets (only these, nothing else):

* attention output projection: o_proj
* MLP down projection: down_proj

Layer subset: middle band only to avoid destabilizing early lexical and late logits. If L is number of transformer blocks, perturb layers l ∈ [⌊0.3L⌋, ⌊0.7L⌋].

Basis initialization:

* A_k entries ~ N(0, 0.02)
* B_k entries ~ N(0, 0.02)
* store basis as fixed parameters; never trained

Action application:

* clamp each component: a_k ∈ [-1, 1]
* global scale α = 0.06
* applied weight for each target layer: W’ = W + α Σ_k a_k ΔW_k

Hard safety clamp:

* if ||a||_2 > 2.0, rescale to norm 2.0

2. Dream operator (runtime mechanic)

The model can trigger a dream via a special tool-call token pattern. The base model itself does not need to emit floats. Instead it emits a discrete “DREAM” marker, and the controller computes a.

Text protocol: <dream></dream>

Execution when marker encountered during generation:

1. compute current state s_t from the unperturbed model on the current context (details below)
2. choose an action a_t using the inverse-dynamics planner (Sections 4–6)
3. apply W’ = W + ΔW(a_t)
4. generate N_dream = 40 tokens (temperature 0.9, top_p 0.95)
5. revert to W
6. insert the dream text into the context (verbatim) and continue generation unperturbed

Limits:

* max dream calls per sample: 2
* if dream text degenerates (repetition loop / entropy collapse), abort dream (insert nothing), record as failure transition

3. State representation s (what the dynamics models operate on)

We must learn dynamics in a compact “cognitive state” space, not raw text.

State extractor obs(context) produces s ∈ R^64. It is deterministic, cheap, and uses a single forward pass with cached activations.

Compute on the current context (prompt + generated prefix), using the unperturbed model W:

Features (all scalars, then concatenated and normalized):
A) Logit stats at current step (10 dims)

* entropy over vocab
* top1 prob
* top5 mass
* top20 mass
* logit gap (top1 - top2)
* mean logit
* std logit
* fraction of probability mass in top200
* perplexity proxy = exp(entropy)
* repetition risk = prob(token == last_token)

B) Hidden-state stats at 3 layers (each layer gives 8 dims → 24 dims)
Pick layers: l1=⌊0.25L⌋, l2=⌊0.5L⌋, l3=⌊0.75L⌋.
For each selected layer l:

* mean(h_l)
* var(h_l)
* mean(|h_l|)
* var(|h_l|)
* mean of last-token vector (same as mean, but computed on last token only; treat separately)
* var of last-token vector
* cosine similarity between last-token h_l and running mean of last 16 tokens (captures “stuckness”)
* norm of last-token h_l

C) Attention stats at same 3 layers (each gives 6 dims → 18 dims)
For each layer l:

* mean attention entropy across heads for last token
* std attention entropy across heads
* mean max-attention weight across heads
* std max-attention weight across heads
* mean attention distance (expected position index)
* std attention distance

D) “Recent trajectory” deltas (12 dims)
Compute the same logit stats (A subset: entropy, top1, top5, gap) at the last 3 generated steps (if available). Include first differences vs current step. If insufficient history, zero-pad.

Normalize s with running mean/std (EMA) computed during data collection. This normalization is frozen after Phase 1.

4. Data collection: transitions under random actions (fast manifold probing)

We collect a dataset of state transitions induced by weight actions.

A transition is (c, s_t, a_t, s_{t+1}, quality), where c is a fixed prompt embedding (optional; we include it) and quality flags stability.

Prompt distribution: TinyStories prompts only, because we need tons of rollouts fast. Use:

* random story starters (1–2 sentences)
* controlled starters with style tags (e.g. “Write a scary story about …”, “Write a funny story …”)
* random character+setting templates

Rollout protocol for one prompt:

1. start with prompt P, empty prefix
2. generate T_base = 24 tokens unperturbed (temperature 0.9, top_p 0.95) to get into a “story state”
3. at step t, compute state s_t on the current context
4. sample random action a_t:

* with prob 0.5: a_t ~ Uniform([-1,1]^K) then L2 clamp
* with prob 0.5: a_t ~ Normal(0, 0.5) then tanh, then L2 clamp

5. apply dream perturbation and generate N_dream = 40 tokens under W’
6. compute next state s_{t+1} on the new context, but using unperturbed W (important: state is always measured in the base coordinate system)
7. compute stability/quality metrics on the dream segment:

* repetition rate over 4-grams
* average token entropy under W’ during dream
* fraction of tokens that are newline/punctuation spam
  Mark quality=0 if degenerate; else quality=1

8. store transition

Collect 2,000,000 transitions (this is the point of TinyStories speed). Store in a flat dataset (parquet) with s_t, a_t, s_{t+1}, quality, plus prompt id.

5. Forward dynamics model F (predict effect of action)

Goal: learn s_{t+1} ≈ F(s_t, a_t, c_t). This gives predictive structure and supports planning / model diagnostics.

Context embedding c_t: use the base model’s last-layer hidden of the last prompt token before generation begins (computed once per prompt; 256 dims). Project down to 64 dims with a learned linear layer.

Model architecture (fixed):

* Input: [s_t (64), a_t (32), c (64)] → 160 dims
* MLP with residual:
  160 → 512 → 512 → 64
  ReLU activations
  Output is Δs; predict s_{t+1} = s_t + Δs

Loss:

* weighted MSE over dims, with quality gating:
  L_F = quality * ||s_{t+1} - Ŝ_{t+1}||_2^2
* plus small L2 regularization on weights

Train for 3 epochs over 2M transitions, batch 4096, AdamW lr 2e-4.

6. Inverse dynamics model I (infer action for desired transition)

Goal: learn a ≈ I(s_t, s_{t+1}, c). This is the key sample-efficiency unlock: it turns “search over actions” into “solve for action”.

Architecture (fixed):

* Input: [s_t (64), s_{t+1} (64), c (64)] → 192 dims
* MLP:
  192 → 512 → 512 → 32
  tanh output in [-1,1]
* enforce L2 clamp at inference time

Loss:

* only on quality=1 samples:
  L_I = quality * ||a_t - â_t||_2^2

Train jointly with F or sequentially; choose sequentially:

1. train F to convergence
2. train I to convergence

7) Planning: choose “where to go” in state space

Inverse dynamics needs a target next-state s*. We define s* by maximizing an automatically computable objective that corresponds to “useful alternate thought” while staying coherent. No human in the loop.

We define a scalar objective J(s_{t+1}) computed from the state vector (and optionally from a cheap classifier). It must correlate with “novel but coherent continuation”.

Define:

* novelty proxy: J_nov = + entropy(s_logits) + (1 - top1_prob) + (1 - repetition_risk)
* coherence proxy: J_coh = - penalty_low_entropy - penalty_high_repetition - penalty_attention_collapse
  Where:
  penalty_low_entropy triggers if entropy < threshold (set threshold by 5th percentile of entropy over quality=1 data)
  penalty_high_repetition triggers if repetition_risk > threshold
  penalty_attention_collapse triggers if attention entropy std is near 0

Total objective:
J = 1.0*J_nov + 1.5*J_coh

Target selection rule (fixed):
At a given s_t and c, generate M=64 candidate targets using the forward model as a proposal mechanism:

1. sample candidate actions a^(m) ~ Normal(0,0.35) then tanh then L2 clamp
2. predict s^(m) = F(s_t, a^(m), c)
3. score J(s^(m))
4. choose s* = argmax_m J(s^(m))

Then compute the action to realize it using inverse dynamics:
a* = I(s_t, s*, c)

Optionally refine once (fixed, one step):
Predict s_pred = F(s_t, a*, c). If J(s_pred) < J(s*), do one correction:
a_corr = I(s_t, s_pred + (s* - s_pred), c)
Set a* = 0.5 a* + 0.5 a_corr, then clamp.

This gives a directed, sample-efficient action chooser without RL.

8. Training the dream trigger and integration (meta-adapter)

Now we need the base model to learn “when to call <dream>” and “how to use the dream text”.

We train a small LoRA meta-adapter on the base model with GRPO. The adapter does not output actions; actions come from the planner (Section 7). The adapter only decides whether to dream and how to integrate.

Adapter:

* targets: q_proj and v_proj in all layers
* LoRA rank: 8
* alpha: 16
* dropout: 0.05

Behavioral format:
The model is trained to optionally emit <dream></dream> at most twice.

Task dataset for GRPO:
Use TinyStories style objectives that are automatically scoreable:

* “include these 3 required entities in the story”
* “end with a twist”
* “maintain consistent character names”
* “use a rhyme couplet at the end”
  Each prompt includes explicit constraints; reward can be computed with string checks + a small frozen classifier.

Reward (scalar in [0,1]):
R = 0.4*constraint_satisfaction + 0.3*ending_quality_proxy + 0.3*coherence_proxy
Where:
constraint_satisfaction: fraction of required entities/constraints met
ending_quality_proxy: must contain an explicit “twist marker” (e.g. “but then”, “until”, “secretly”) plus novelty stats of last 20 tokens
coherence_proxy: penalize repetition + name inconsistency (simple regex name tracker)

GRPO:

* group size G=8 completions per prompt
* update only adapter params
* baseline = group mean
* clip ratio = 0.2
* lr = 1e-4
* train 50k prompts

During training, when the model emits <dream>, execute planner-chosen dream (Sections 2,7). The dream segment becomes part of the context, so the model can “read” the alternative continuation and integrate.

9. What “working” means (validation)

You must show three effects, measured automatically, no subjective judgement:

A) Dream usage is selective:
P(dream | high difficulty) > P(dream | low difficulty)
Difficulty proxy: low top1_prob + high entropy + repetition_risk rising.

B) Dream improves reward:
E[R | dreams enabled] > E[R | dreams disabled] on held-out prompts, compared against:

* same adapter with dreams forcibly disabled
* same system but planner replaced with random a

C) Inverse dynamics matters:
planner (F+I) beats random actions at same dream budget, meaning:
E[R_planner] > E[R_random] and degeneration rate lower.

10. Fixed hyperparameters summary

Base: TinyStories GPT-2 class (~110M), frozen
K=32, rank-1 basis, α=0.06, L2 clamp=2.0
Targets: o_proj, down_proj, layers 0.3L..0.7L
Dream: 40 tokens, temp 0.9, top_p 0.95, max 2 dreams
State: 64 dims as defined
Data: 2,000,000 transitions
Forward model: MLP 160→512→512→64 residual, lr 2e-4, batch 4096
Inverse model: MLP 192→512→512→32 tanh, lr 2e-4, batch 4096
Planning: M=64 proposals, 1 refinement step
Meta-adapter: LoRA rank 8 on q_proj,v_proj all layers
GRPO: G=8, lr 1e-4, 50k prompts

11. Files (exact)

v0/
env.py                  (TinyStories wrapper, LoRA basis apply/revert, generation)
state.py                (obs extractor producing s ∈ R^64)
collect_transitions.py  (random action rollout, writes dataset)
forward_model.py        (F definition + train)
inverse_model.py        (I definition + train)
planner.py              (target proposal via F, action solve via I)
dream_exec.py           (hook: on <dream> call planner then generate dream)
tasks.py                (TinyStories constraint prompt generator + reward)
train_grpo.py           (adapter GRPO training with dream execution)
eval.py                 (A/B/C validation metrics)

No other components. No world model transformers. No diffusion. No action clustering. Sample efficiency comes from inverse dynamics + forward-proposal target selection, all trained on cheap random probing.
