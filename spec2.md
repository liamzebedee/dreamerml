The Artifact: A Map of the Model’s Internal Axes
The goal of this experiment is to produce a static, readable artifact that makes the discovered weight directions legible. The output is not metrics or training curves, but a curated set of controlled generations showing how each learned knob changes the model’s behavior. For each direction, we fix a small set of prompts and generate continuations at multiple scalar settings (e.g. −3, −1, 0, +1, +3). The artifact is simply these grids of text, arranged so that the behavioral transformation is visually obvious.

Each section of the blog post introduces one discovered axis. It begins with a short quantitative summary: average KL divergence from the base model, prompt variance of that divergence, entropy shift, and coherence shift. Then it shows side-by-side generations under increasing knob strength. The reader should be able to see a smooth, monotonic change: perhaps structure becomes more elaborate, or confidence increases, or narrative perspective shifts. The key criterion is continuity — moving the knob slightly should produce a small, coherent behavioral change rather than chaos.

A second part of each section demonstrates hierarchy. We fix a “coarse” knob at two different values (low and high), and then sweep a “fine” knob within each regime. The artifact shows four grids: (coarse low, fine sweep) and (coarse high, fine sweep). If hierarchy has emerged, the fine knob’s effect will only become meaningful under certain coarse settings. This demonstrates nested control rather than independent axes.

The final part of the artifact aggregates patterns across prompts. For each knob, we include a brief automatically generated summary describing the common transformation it induces, derived from differences between generations (e.g. average length change, lexical diversity shift, structural complexity indicators). The blog post frames these as “discovered internal directions” — not labeled features, but operationally defined axes in weight space that produce stable, reproducible changes in behavior.

The result should read like an interpretability report: a sequence of controlled experiments revealing that the model’s behavior can be smoothly deformed along low-dimensional directions. The artifact is interesting if the reader can see that these deformations are coherent, hierarchical, and consistent across contexts — evidence that structured control lives in weight space.






