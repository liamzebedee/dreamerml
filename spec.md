Title
RL-Based Discovery of Hierarchical Weight Directions via Unsupervised Behavioral Probes

Core Commitment
This system is explicitly reinforcement learning over weight edits.
The base model is an environment.
Actions are structured weight perturbations.
Rewards come from unsupervised behavioral probes computed on rollouts.

Goal
Discover a small set of reusable weight directions that produce coherent, scale-separated changes in output behavior. No semantic labels.

Environment

State
Frozen base model weights w.
We do not treat w as dynamic during an episode.

Action

Action vector a ∈ R^K.
Weight edit:

w' = w + Σ_k a_k b_k

b_k are learnable basis directions implemented as LoRA rank-1 updates on:

* attention output projection (wo)
* MLP output projection (fc2)

This ensures edits affect the residual stream directly and propagate globally.

K small (e.g. 4–8).

Episode

1. Sample action a from policy π_φ(a).
2. Construct edited weights w'.
3. For fixed prompt set X (e.g. 32 prompts), generate N short rollouts each (length L).
4. Compute probe statistics over all rollouts.
5. Compute reward R(a).
6. Update policy φ and basis directions b_k using policy gradient (REINFORCE or PPO-style).

Unsupervised Probes (Chosen for Maximum Structural Signal)

We choose probes that measure scale and structure of distributional change, not surface token counts.

Probe 1: Global Behavioral Divergence

S_global(a) = E_x D_KL(p_{w'}(·|x) || p_w(·|x))

Approximate KL using next-token distributions over rollout prefixes.

Interpretation: how much the overall predictive distribution changed.

Probe 2: Prompt-Variance Divergence

S_var(a) = Var_x D_KL(p_{w'}(·|x) || p_w(·|x))

Interpretation: does the edit shift behavior uniformly across prompts (low variance) or selectively (high variance)?

This naturally separates coarse vs fine effects.

Probe 3: Long-Range Coherence Proxy

For each rollout y, compute:

C(y) = average mutual information between first half and second half token embeddings (estimated via hidden-state similarity or predictive logprob conditioning gap).

Define:

S_coherence(a) = E_y C(y)

This captures global structural organization without semantic labels.

Probe 4: Entropy Shift

S_entropy(a) = E_x,t H(p_{w'}(·|prefix_{x,t})) − H(p_w(·|prefix_{x,t}))

This detects changes in confidence, mode collapse, or diversification.

Reward Design

We want basis directions to specialize.

Define reward components:

R_coarse(a) = S_global(a) − λ S_var(a)

Encourages large, uniform global shifts (coarse feature).

R_fine(a) = S_var(a) − μ S_global(a)

Encourages selective prompt-dependent shifts (fine feature).

R_structure(a) = S_coherence(a)

Encourages structural change.

Total reward:

R(a) = α R_coarse(a) + β R_fine(a) + γ R_structure(a) − η ||Δw||^2

The coefficients determine which behaviors dominate.

Hierarchy Emergence

We partition knobs:

First half of a reserved for coarse directions.
Second half reserved for fine directions.

We enforce specialization by:

* Reward masking: coarse knobs only receive gradient from R_coarse.
* Fine knobs only receive gradient from R_fine.

Additionally, implement gating:

a_fine_effective = sigmoid(W a_coarse) ⊙ a_fine

This makes fine effects conditional on coarse state, creating hierarchical control.

Policy

Policy π_φ(a) = Normal(μ_φ, Σ_φ)

Small MLP outputs mean vector.
Variance either learned or fixed.

We use policy gradient:

∇_φ E[R(a)] = E[ R(a) ∇_φ log π_φ(a) ]

Basis vectors b_k also updated through reward gradients by treating them as part of action-to-reward mapping.

Why These Probes

KL divergence is the cleanest unsupervised measure of behavioral change.
Prompt variance of KL naturally defines scale separation.
Coherence and entropy introduce structural axes beyond mere magnitude.

Together they produce a multi-dimensional behavioral manifold without labels.

Expected Dynamics

Early training: random directions cause chaotic divergence.
Policy learns to avoid destructive edits (penalty + KL).
Stable directions emerge that reliably shift distribution structure.
Coarse knobs become global behavior modulators.
Fine knobs become conditional style-like modifiers.

Core Insight

We are not modeling p(w).
We are learning a control basis that maximizes structured, reproducible changes in output distribution statistics under low-dimensional weight edits.

The RL framing ensures discovery is causal, not correlational.


## The Artifact: A Map of the Model’s Internal Axes

The goal of this experiment is to produce a static, readable artifact that makes the discovered weight directions legible. The output is not metrics or training curves, but a curated set of controlled generations showing how each learned knob changes the model’s behavior. For each direction, we fix a small set of prompts and generate continuations at multiple scalar settings (e.g. −3, −1, 0, +1, +3). The artifact is simply these grids of text, arranged so that the behavioral transformation is visually obvious.

Each section of the blog post introduces one discovered axis. It begins with a short quantitative summary: average KL divergence from the base model, prompt variance of that divergence, entropy shift, and coherence shift. Then it shows side-by-side generations under increasing knob strength. The reader should be able to see a smooth, monotonic change: perhaps structure becomes more elaborate, or confidence increases, or narrative perspective shifts. The key criterion is continuity — moving the knob slightly should produce a small, coherent behavioral change rather than chaos.

A second part of each section demonstrates hierarchy. We fix a “coarse” knob at two different values (low and high), and then sweep a “fine” knob within each regime. The artifact shows four grids: (coarse low, fine sweep) and (coarse high, fine sweep). If hierarchy has emerged, the fine knob’s effect will only become meaningful under certain coarse settings. This demonstrates nested control rather than independent axes.

The final part of the artifact aggregates patterns across prompts. For each knob, we include a brief automatically generated summary describing the common transformation it induces, derived from differences between generations (e.g. average length change, lexical diversity shift, structural complexity indicators). The blog post frames these as “discovered internal directions” — not labeled features, but operationally defined axes in weight space that produce stable, reproducible changes in behavior.

The result should read like an interpretability report: a sequence of controlled experiments revealing that the model’s behavior can be smoothly deformed along low-dimensional directions. The artifact is interesting if the reader can see that these deformations are coherent, hierarchical, and consistent across contexts — evidence that structured control lives in weight space.
