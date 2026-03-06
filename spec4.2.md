Goal

Demonstrate that a language model can access thoughts outside its default reasoning manifold by temporarily perturbing its weights. A learned policy over weight space generates coherent perturbations that produce alternative reasoning trajectories. The base model remains frozen; only the controller components learn.

The system consists of four parts:

1. frozen base LLM
2. fixed LoRA perturbation basis defining the latent weight space
3. perturbation policy network that generates weight modifications
4. GRPO-trained meta-adapter that decides when to invoke dreams

The perturbation policy learns which regions of weight space produce coherent alternative reasoning. The meta-adapter learns when those regions help solve problems.

---

Base model

Model

Qwen2.5-1.5B-Instruct

Weights remain frozen permanently.

Inference framework

Transformers + PEFT.

No gradient updates applied to base parameters.

---

Latent weight space

Define a small LoRA basis that parameterizes nearby models.

Number of directions

K = 16

Rank per direction

1

Perturbation layers

attention.o_proj
mlp.down_proj

Layer range

middle 40% of layers

Example (24 layers)

layers 8–16

Initialization

A_k ~ N(0,0.02)
B_k ~ N(0,0.02)

Perturbation magnitude

ΔW(a) = α Σ a_k A_k B_k

with

α = 0.04
a_k ∈ [-1,1]

This creates a 16-dimensional latent policy space over nearby weight configurations.

---

Dream operator

The model can emit

<dream>|weightmod|</dream>

Execution procedure

1. compute state vector s describing current reasoning state
2. feed s into perturbation policy π(s) to obtain action vector a
3. apply weight perturbation ΔW(a)
4. generate dream continuation
5. revert weights to base values
6. insert dream text into context

Dream generation parameters

tokens = 48
temperature = 0.9
top_p = 0.95

Maximum dream calls per response

2

Dream text becomes visible to the model for subsequent reasoning.

---

State representation

The perturbation policy needs a compact summary of the model’s cognitive state.

Compute state vector

s ∈ R¹⁶

Features

logit entropy
top-1 probability
top-5 probability mass
top-20 probability mass
last layer hidden mean
last layer hidden variance
attention entropy mean
attention entropy variance

These statistics are extracted during the forward pass and normalized.

This representation is extremely cheap to compute.

---

Perturbation policy

The perturbation policy generates weight modifications conditioned on the current state.

Mapping

a = π(s)

Policy architecture

MLP

16 → 128 → 128 → 16

Activation

ReLU

Output

a = tanh(MLP(s))

so

a ∈ [-1,1]^16

This vector controls the LoRA mixture.

Policy parameters

≈ 30k parameters.

---

Policy training phase

Before using dreams for tasks, train the perturbation policy to generate coherent but diverse alternative behaviors.

Sampling procedure

1. sample random prompt
2. compute state s
3. generate perturbation a = π(s)
4. run dream generation
5. measure text statistics

Rewards

Coherence reward

penalize collapse:

repetition rate
low entropy tokens

Novelty reward

encourage distribution shift from base model

Compute

R_nov = KL(p_dream || p_base)

Total reward

R = R_nov − λ * collapse_penalty

Train the perturbation policy using GRPO on this reward.

Goal

discover perturbations that produce coherent but distinct reasoning trajectories.

After training stabilizes, freeze the perturbation policy.

---

Meta-adapter

Train a LoRA adapter on the base model that controls dream usage.

Targets

attention.q_proj
attention.v_proj

Rank

8

Parameters

≈1–2M.

Responsibilities

decide when to dream
trigger dream operator
integrate dream outputs into reasoning

The adapter does not generate perturbations itself; it triggers the policy.

---

Task training (GRPO)

Dataset

mixture of tasks:

math reasoning
logic puzzles
coding tasks
creative prompts

Training loop

1. prompt sampled
2. model generates response
3. meta-adapter may emit <dream>
4. dream operator executes perturbation policy and generation
5. model finishes answer
6. judge scores output

Judge

task metric or strong LLM judge.

GRPO update

meta-adapter parameters only.

The perturbation policy remains frozen.

---

Evaluation

Compare three systems

base model
base + meta-adapter (no dreams)
base + meta-adapter + dreams

Metrics

math accuracy
coding pass rate
judge score for creativity
problem-solving success on difficult prompts

Success criterion

dream system significantly outperforms adapter-only baseline.

---

System behavior

During reasoning, the model can temporarily step into nearby weight configurations that emphasize different associations or reasoning paths.

These altered states generate thoughts the default model rarely produces.

Those thoughts are imported back through the context window.

The result is increased cognitive diversity without changing the base model’s permanent weights.

---

Implementation outline

modules

env.py
base model wrapper and LoRA basis

policy.py
perturbation policy network

state.py
state feature extraction

dream_executor.py
weight modification + generation

train_policy.py
policy exploration phase

train_grpo.py
task training

Total implementation

≈500 lines of code.

---

Outcome

If successful, the model learns to use weight perturbations as a cognitive tool, enabling reasoning paths that lie outside the default manifold of the base model.
