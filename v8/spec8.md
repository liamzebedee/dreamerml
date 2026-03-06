We introduce **Dreamer**, a secondary activation stream layered on top of an existing transformer that allows the model to modulate its internal feature activations during inference. The base model executes normally as a sequence of layers (l_0 \rightarrow l_1 \rightarrow l_2 \rightarrow \dots \rightarrow l_L). At each layer and token position we observe the residual activation (h_{l,t}). The Dreamer stream runs in parallel with the token stream and operates per token position, allowing the model to compute and optionally modify internal activation features synchronously with the forward pass.

Before Dreamer can operate, we first learn a structured feature basis for the residual stream using a sparse autoencoder (SAE). We collect residual activations (h_{l,t}) from the base model across the training corpus and train the SAE to reconstruct them:

(f_{l,t} = E(h_{l,t}))

(h_{l,t} \approx D f_{l,t})

Here (f_{l,t}) is a sparse feature vector and (D) is a decoder matrix whose columns correspond to learned activation directions. The sparsity constraint encourages these directions to represent independent and interpretable components of the activation manifold. The SAE therefore provides both a dictionary of meaningful activation features and a coordinate representation describing which features are currently active at each token position during inference.

Dreamer operates directly in this feature space. During the forward pass the transformer computes the residual state (h_{l,t}), the SAE encoder computes the corresponding feature activations (f_{l,t}), and Dreamer outputs a gain vector

(g_{l,t} = \text{Dreamer}(f_{l,t}))

which represents adjustments to these internal features. These gains act as “knobs” over the SAE feature dictionary. The residual stream is then modified through the decoder directions:

(h'*{l,t} = h*{l,t} + D g_{l,t})

This allows the model to increase or suppress specific internal features before the next layer processes the state. Because Dreamer operates per token position, the control stream is aligned with the token stream and evolves synchronously with the model’s computation.

To establish a baseline, we first run the pretrained model across the training corpus and compute the SAE feature activations (f_{l,t}) for every sequence. During this stage Dreamer is inactive and all feature gains are initialized to zero:

(g_{l,t} = 0)

so the residual state remains unchanged (h'*{l,t} = h*{l,t}). This produces a dataset of ((\text{base_input}, \text{activation_features})) describing the natural feature trajectories of the model under standard inference. The SAE feature space therefore becomes a stable coordinate system over the model’s activation manifold before any steering behavior is learned.

During post-training with GRPO, Dreamer becomes an internal policy over these feature gains. For each prompt the model generates multiple rollouts with different Dreamer gain trajectories ({g_{l,t}}). Rewards are computed on the resulting outputs and the Dreamer policy is updated relative to the group. To maintain stability within the learned feature manifold, gain magnitudes are regularized through penalties on (|g_{l,t}|) or the induced residual perturbation (|D g_{l,t}|). Over time Dreamer learns when and how to modulate internal features nonverbally, using the SAE feature dictionary as a compact interface for steering the model’s internal computation.

The first milestone of this system is the discovery of **monosemantic activation features**. Starting with TinyStories, we train the SAE over residual activations and analyze individual feature directions by sweeping their gains and observing changes in model outputs. For each feature we collect example prompts, baseline completions, and gain sweeps showing how the generated text changes as the feature is amplified or suppressed. A separate high-capability language model is then used to automatically interpret these features. Given samples of texts where the feature naturally activates and examples of completions under gain modulation, the model generates a short name and a short descriptive explanation of the feature. The interpreter also has the option of declaring a feature uninterpretable by emitting the shortname `no-idea/1` along with its best attempt at describing any weak pattern observed.

These automatically generated labels form a **meta-index of activation features** that allows the system to track and report how the model’s internal representation space is organized. Each feature is stored as a `.txt` artifact containing its index, the proposed shortname, the generated description, and example gain sweeps. This index becomes the reference vocabulary used when analyzing Dreamer behavior.

The second milestone is observing **emergent behavior during Dreamer training**. During GRPO we periodically sample prompts and generate outputs with the current Dreamer policy while logging the feature gain trajectories applied during generation. These runs produce `.txt` reports containing sampled outputs, feature gain statistics, and summaries referencing the interpreted feature names from the meta-index. By inspecting these artifacts across training iterations we can track how the model begins to intentionally activate or suppress particular internal features. This provides a continuous and interpretable record of how Dreamer learns to use the discovered feature space to influence the model’s reasoning and generation behavior over time.


---


Build the MVP against TinyStories. Use a small TinyStories transformer as the base model, train or load it on the 3090, choose one or two mid residual layers as the intervention points, and make the first end-to-end deliverable the full feature-discovery and gain-sweep pipeline before any GRPO work. The immediate system is: run TinyStories prompts through the model, capture residual activations at the chosen layer and per token position, train an SAE over those activations, encode the corpus into sparse feature activations, select candidate interpretable features, sweep gains on those features during generation, and emit structured .txt artifacts showing baseline outputs, modulated outputs, and feature statistics. Do not overbuild abstractions early. The MVP is successful if we can point to concrete feature directions that consistently alter TinyStories generations in meaningful ways.

The feature-labelling stage should use only Claude Haiku via the CLI `claude` locally installed as the external high-intelligence interpreter. For each candidate feature, prepare a compact prompt bundle containing: high-activation text samples, several gain-swept generations from the same prompts, feature statistics, and a strict instruction to output a shortname plus a concise interpretive description, with permission to output no-idea/1 when the pattern is weak or non-interpretable. Save each feature report as a .txt file. Then build a meta-index .txt file over all labelled features so later Dreamer gain-stream reports can refer to human-readable feature names rather than raw indices. This labelling pass is not optional; it is part of the MVP because the whole point is to make the feature space legible enough to inspect as training proceeds.

Implementation guidance: optimize aggressively for the FULL USAGE of the 3090 from the start. Keep the base model small enough to permit fast iteration, mixed precision, large batch throughput, and repeated forward passes without friction. Parallelize every embarrassingly parallel stage: activation extraction across dataset shards, SAE training data preparation, feature statistics computation, gain-sweep generation jobs, feature-labelling prompt construction, and report generation. Cache activations to disk in chunked binary form, avoid repeated forward passes when data can be reused, stream processing rather than materializing unnecessary intermediate objects, use pinned memory and efficient dataloaders, and batch generation wherever possible. Do not wait for a “perfect” architecture before running experiments: ship the minimal fast path, measure, profile, optimize the slowest component, and keep producing visible .txt artifacts at every stage so emergence, failure, and progress are observable immediately rather than inferred later.