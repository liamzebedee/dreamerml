# DreamerML v2: Dreaming Models

## Core Idea

An LLM learns to **dream** — temporarily modifying its own weights, sampling from the altered state, and integrating the experience — to access cognitive operations its default weights can't reach. The dream process is driven by an RL agent that maps the weight-edit landscape. The text samples produced during exploration are byproducts that become training signal for a language-level dream interface (`<dream>|weightmod|</dream>`).

The landscape drifts as the model learns. We accept this — the world model tracks the drift through periodic real rollouts, and the dreamer continues exploring the evolving terrain.

---

## Architecture

### Three components, one substrate

```
┌─────────────────────────────────────────────────────┐
│  Base LLM (Qwen2.5-0.5B)                           │
│  ┌───────────────────────┐  ┌────────────────────┐  │
│  │ Base weights (frozen) │  │ Meta-adapter (LoRA) │  │
│  │ = the landscape       │  │ = learns to dream   │  │
│  └───────────────────────┘  └────────────────────┘  │
│                                                     │
│  LoRA basis directions (K rank-1 edits)             │
│  = the action space / knobs                         │
└─────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐    ┌──────────────────────────────┐
│   RL Dreamer    │    │       World Model             │
│   (explorer)    │    │  (transformer + diffusion)    │
│                 │◄──►│                               │
│ Maps landscape  │    │  Predicts weight-edit effects  │
│ via rewards     │    │  Tracks landscape drift        │
└─────────────────┘    └──────────────────────────────┘
```

### Base LLM + LoRA basis (the environment)

Unchanged from v1. Frozen LLM with K rank-1 LoRA basis directions applied to attention output (`o_proj`) and MLP output (`down_proj`). An action `a ∈ R^K` produces weight edits `Δw = Σ_k a_k * (A_k ⊗ B_k)`.

The base weights define the territory. They never change.

### Meta-adapter

A small LoRA adapter (separate from the basis directions) trained on the base model. This is the only component whose weights change during phase 2. It learns:
- When to invoke `<dream>` (recognise problems that benefit from self-perturbation)
- What `|weightmod(xxx,yyy,zzz)|` parameters to emit
- How to verbalise dream experiences in `<think>` blocks
- How to integrate dream insights into final output

Because the adapter is small, the landscape shift it introduces is minimal. The world model tracks whatever drift occurs.

### World Model (deep landscape model)

**Architecture**: Transformer encoder over trajectory sequences, with a diffusion head for multi-modal prediction.

**State representation**: Instead of predicting 4 scalar probes, the world model operates on richer observations:
- Per-layer activation statistics (mean, variance of hidden states at layers 1/4, 1/2, 3/4, L)
- Logit distribution features (top-k entropy, rank distribution shift vs baseline)
- Attention pattern summaries (head-averaged attention entropy per layer)

This gives an observation vector of ~64-128 dimensions per environment step, enough for the world model to discover structure the 4-probe summary hides.

**Sequence modelling**: The world model processes trajectories, not single steps:

```
Input:  [(a_0, obs_0), (a_1, obs_1), ..., (a_t, obs_t)]
Output: predicted obs_{t+1}, predicted reward_{t+1}
```

The transformer's context window IS the agent's memory of where it has been. This enables topology learning — the model can represent "region A is near region B" and "there's a discontinuity between C and D."

**Diffusion head**: The observation predictor is a small conditional diffusion model rather than a direct regression head. This lets it represent multi-modal predictions — a single action might lead to different behavioural outcomes depending on which attractor basin you're near. The modes of the diffusion model correspond to attractor basins in the landscape.

### RL Dreamer (explorer)

The dreamer's job is **cartography**, not task performance. Its reward incentivises:

1. **Coverage**: visit diverse regions of the action space. Measured by the spread of visited observations (e.g. determinantal point process on observation embeddings, or simpler: average pairwise distance in observation space).

2. **Surprise**: visit regions the world model can't yet predict well. The world model's prediction error is intrinsic reward — go where the map is wrong.

3. **Structure**: discover that the space has organisation. Reward for finding actions where small perturbations produce consistent (not random) behavioural changes. This incentivises finding smooth manifolds rather than noisy regions.

Combined reward:
```
R_explore = w_cov * R_coverage + w_surp * R_surprise + w_struct * R_structure
```

The dreamer uses GRPO (as in v1) but with this exploration reward instead of the v1 probe-based reward.

---

## Training

### Phase 1: Landscape exploration (dreamer + world model)

```
for step in range(N_explore):
    # Real rollout: dreamer samples actions, env returns rich observations
    actions = dreamer.sample(G)
    observations = [env.step_rich(a) for a in actions]

    # Store in replay buffer
    buffer.add_trajectory(actions, observations)

    # Train world model on replay data
    world_model.train_on_buffer(buffer)

    # Train dreamer with exploration reward
    # (uses world model prediction error as surprise signal)
    dreamer.grpo_update(actions, exploration_rewards)

    # Dream rollouts: dreamer explores in world model (no env needed)
    for _ in range(N_dream_per_real):
        dream_actions = dreamer.sample(G)
        dream_obs = world_model.predict(dream_actions)
        dream_rewards = compute_exploration_reward(dream_obs, world_model)
        dreamer.grpo_update(dream_actions, dream_rewards)
```

**Byproduct collection**: During real rollouts, we also generate text from the weight-edited model. These (action, text_sample) pairs are stored separately — they're the raw material for phase 2.

### Phase 2: Learning to dream (meta-adapter training)

The meta-adapter learns to invoke `<dream>` as a tool call via GRPO on downstream tasks.

**Training data format** (generated from phase 1 byproducts):

```
Task:   Write a poem about loss.
Response:
<dream>
|weightmod(0.3, -1.2, 0.0, 0.8, ...)|
[sample from modified model appears here]
</dream>
<think>
[model verbalises what it experienced]
</think>
[final output, informed by dream]
```

**Training loop**:

```
for step in range(N_meta):
    # Sample a task prompt
    task = sample_task()

    # Model generates response (may include <dream> invocations)
    # When |weightmod(...)| is encountered, execution is intercepted:
    #   1. Parse parameters → action vector
    #   2. Apply weight edit to base model via LoRA basis
    #   3. Generate dream sample from modified model
    #   4. Revert weight edit
    #   5. Insert dream sample into context
    #   6. Model continues generating (</dream>, <think>, final output)
    response = generate_with_dream_execution(model, task)

    # Score with reward model / judge
    reward = score(task, response)

    # GRPO update on meta-adapter only
    meta_adapter.grpo_update(response, reward)

    # Periodic: re-validate world model against real env
    if step % validation_freq == 0:
        world_model.validate_and_update(env, dreamer)
```

**Landscape drift handling**: The meta-adapter changes the forward pass, which means the same `|weightmod|` produces slightly different behaviour over training. We handle this by:
- Keeping the adapter small (rank 4-8) to minimise drift
- Periodic world model re-validation with real env rollouts
- The dreamer occasionally re-explores to update the map

### Phase 3: Emergent vocabulary (speculative / future)

As the model uses `<dream>` invocations, it may develop consistent shorthand in its `<think>` blocks for recurring dream experiences. If we add learnable special tokens to the vocabulary and allow the model to emit them during `<think>`, the pressure to efficiently refer to dream states could drive emergence of a "dream language."

This phase is not engineered — it's observed. We watch for:
- Consistent patterns in `<think>` verbalisation across similar weight edits
- Tokens or phrases that reliably co-occur with specific attractor basins
- Evidence that the model reuses dream descriptions across tasks

If these patterns emerge, we add special tokens and let GRPO train their embeddings. If they don't, the model is verbalising fine in English and that's OK too.

---

## Implementation plan

### What stays from v1

- `env.py`: `BaseModelEnv` and `LoRABasis` — unchanged except adding a `step_rich()` method that returns the full observation vector (not just 4 probes)
- GRPO mechanics in the actor
- Replay buffer concept

### What changes

| Component | v1 | v2 |
|---|---|---|
| Observation | 4 probes (KL, var, entropy, coherence) | ~64-128d rich observation vector |
| World model | 2-layer transformer, action→probes | Deep transformer over trajectories + diffusion head |
| Dreamer reward | Probe-based (R_coarse + R_fine + R_struct) | Exploration-based (coverage + surprise + structure) |
| Actor | Separate MLP policy | Separate MLP policy (phase 1) + meta-adapter on LLM (phase 2) |
| Hierarchy | Explicit gating (coarse/fine split) | Implicit — whatever structure the world model discovers |
| Dream interface | None | `<dream>\|weightmod\|</dream>` tool-use pattern |

### New files

```
v2/
  env.py              # Extended env with step_rich()
  world_model.py      # Transformer + diffusion world model
  dreamer.py          # RL explorer with exploration reward
  meta_adapter.py     # LoRA meta-adapter + dream execution engine
  train_explore.py    # Phase 1 training
  train_dream.py      # Phase 2 training
  rewards.py          # Exploration reward computation
  observations.py     # Rich observation extraction from LLM
```

### Key hyperparameters

```yaml
# Landscape
K: 16                    # LoRA basis directions
lora_scale: 0.05         # Basis init scale (validated in v1)
obs_dim: 96              # Rich observation vector size

# World model
wm_d_model: 128          # Transformer hidden dim
wm_n_layers: 4           # Deeper than v1's 2
wm_n_heads: 8
wm_context_len: 64       # Trajectory history length
diffusion_steps: 20      # Denoising steps for prediction

# Dreamer
G: 8                     # GRPO group size
explore_reward_weights:
  coverage: 1.0
  surprise: 0.5
  structure: 0.3

# Meta-adapter
adapter_rank: 8           # Small to minimise drift
adapter_target: "q_proj,v_proj"  # Attention only

# Training
phase1_steps: 500         # Real rollouts for exploration
dream_ratio: 4            # Dream rollouts per real rollout
phase2_steps: 1000        # Meta-adapter GRPO steps
validation_freq: 50       # Re-validate world model
```

---

## Open questions

1. **Dream execution mechanics**: When the model emits `|weightmod(0.3, -1.2, ...)|`, do we parse K floats from the token stream? Or does the model emit a discrete action index that maps to a learned weight edit? Continuous is more expressive but harder to generate accurately as text.

2. **Task distribution for phase 2**: What tasks does the meta-adapter train on? Creative writing, reasoning, math? The reward model / judge needs to be defined. Could start with a stronger LLM as judge (Claude/GPT-4) or a task-specific metric.

3. **Drift tolerance**: How much adapter drift can the world model absorb before the map is too stale? Empirical question — need to measure landscape stability as adapter trains.

4. **Scale**: This is specced for Qwen2.5-0.5B. The ideas should transfer to larger models, but compute costs scale with model size (each dream requires a forward pass through the modified base model).

5. **Observation design**: The ~96d observation vector is a design choice. Too small and we lose structure. Too large and the world model needs more data. Need to experiment with which features are informative.

6. **Is diffusion necessary?** The diffusion head adds complexity. It's justified if the landscape is genuinely multi-modal (same action, different outcomes depending on context). If the landscape is smooth and unimodal, a regression head (like v1) suffices. Test both.
