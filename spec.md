Title
dreamirl: Model-Based Reinforcement Learning to Discover Hierarchical Weight Control Directions

Purpose
Learn a low-dimensional control space over a frozen language model’s weights by actively exploring weight edits with RL.
A second model (world model) learns the behavioral landscape so the agent can “dream” and improve exploration over time.

We are not modeling p(w).
We are learning the causal mapping:

a → behavioral consequences

where a parameterizes structured weight edits.

---

System Overview

There are three components:

1. Base Model (Environment)
2. Actor (Weight-Edit Policy)
3. World Model (Learned Landscape Predictor)

The system performs model-based RL over weight space.

---

1. Base Model (Environment)

Given frozen weights w.

Action:

a ∈ R^K

Weight edit:

Δw = Σ_k a_k b_k
w' = w + Δw

b_k are learned basis directions implemented as LoRA rank-1 edits on:

* attention output projection (wo)
* MLP output projection (fc2)

Rollout procedure:

For fixed prompt set X:

* Generate N continuations per prompt
* Compute probe vector p(a)

Environment returns:

p(a), reward R(a)

---

2. Probe Design (Unsupervised)

Probes measure structured changes in output distribution.

Global Divergence
S_global(a) = E_x D_KL(p_{w'}(·|x) || p_w(·|x))

Prompt Variance
S_var(a) = Var_x D_KL(p_{w'}(·|x) || p_w(·|x))

Entropy Shift
S_entropy(a) = change in average token entropy

Long-Range Coherence
S_coherence(a) = mutual information proxy between early and late token states

All probes are purely statistical and require no labels.

Probe vector:

p(a) = [S_global, S_var, S_entropy, S_coherence]

---

3. Reward Function

Reward encourages discovery of structured, scale-separated behavior.

Define:

R_coarse = S_global − λ S_var
R_fine   = S_var − μ S_global
R_struct = S_coherence
R_reg    = −η ||Δw||^2

Total reward:

R(a) = α R_coarse + β R_fine + γ R_struct + R_reg

This induces hierarchy:

* Coarse directions produce global, consistent shifts.
* Fine directions produce localized, prompt-dependent shifts.

---

4. Actor (Policy Over Weight Edits)

Policy:

π_φ(a | z)

Simple Gaussian with learned mean and diagonal covariance.

Inputs:

Optional latent exploration state z (can be empty for minimal version).

Outputs:

Mean vector μ ∈ R^K.

Policy objective:

Maximize expected reward:

max_φ E[R(a)]

Use PPO or simple policy gradient.

---

5. World Model (Landscape Model)

Neural network:

M_ψ(a) → predicted probe vector p̂

Trained with regression loss:

L_world = || M_ψ(a) − p(a) ||^2

World model learns the behavioral response surface.

This is the learned landscape.

---

6. Model-Based RL Loop (Dreaming)

Early phase:

* Actor samples a.
* Real environment computes p(a).
* Update world model.
* Update actor using real reward.

Later phase:

* Actor generates candidate actions.
* World model predicts p̂(a).
* Actor improves policy using predicted reward.
* Periodically validate against real environment.

Over time:

* Actor learns where nonlinear regime boundaries are.
* Actor discovers extreme but stable weight edits.
* World model approximates geometry of probe manifold.

This is “dreaming”: internal simulation of weight edits.

---

7. Hierarchical Control Mechanism

Partition action vector:

a = [a_coarse, a_fine]

Implement gating:

a_fine_effective = sigmoid(W a_coarse) ⊙ a_fine

This forces fine controls to depend on coarse regime.

Hierarchy is enforced structurally.

---

8. Training Procedure

Initialize:

* Random small basis directions b_k.
* Random actor.
* Random world model.

Repeat:

1. Sample action a from policy.
2. Apply weight edit.
3. Generate rollouts.
4. Compute probes p(a).
5. Update world model.
6. Compute reward R(a).
7. Update actor.
8. Update basis directions b_k via gradient through reward.

Continue until:

* Probe landscape becomes predictable.
* Actor reliably produces structured, disentangled behavior shifts.

---

9. Outputs / End Result

The learned artifacts are:

1. Basis directions b_k (weight control axes).
2. Actor capable of navigating control space.
3. World model approximating behavioral manifold.

UI Representation:

* Sliders for each discovered knob.
* Real-time generation preview.
* 2D map of probe space discovered by actor.
* Visualization of exploration trajectory over training.

Interesting behaviors include:

* Phase transitions in entropy/coherence.
* Global structural modulation.
* Conditional stylistic modulation.
* Smooth morphing between behavioral regimes.

---

Core Principle

The system learns a low-dimensional, hierarchical control space over a neural network by actively exploring weight edits and learning a predictive model of their behavioral consequences.

It is not density modeling.
It is causal landscape discovery via RL.
