
## Technical.

Base instruct model with frozen weights.

We have an RL environment. 

An agent can take actions which are modifications to the weights of the base model. Rollouts involve applying the weight mods as LoRA's, sampling tradjectories (completing a base prompt). 



Wait I've got it

Base model
RL env
We instantiate the base model
An agent tries to learn the landscape of the base model's weights
The agent takes actions by modifying weights
Rollouts are generating completions from the base model with perturbed weights applied, in order to probe underlying structure of base model's latent space
Completion is basically a dream - begins with a prompt, tokens are the dream's experience
The dream is a probe - outputted tokens tell the agent how weight modifications change the landscape of the dreamer
Agent is rewarded for discovery of structured, scale-separated behavior

After a while
The world model has an idea of what weight modifications will produce what shifts in the base model's landscape
Dreamscape model : action(weight mod) -> f(x) -> predicted statistical change to base model landscape
Now we introduce cross-attention

The whole point is that the model learns words for these landscapes
In a manual case, we simply get Claude to generate a report of different weight perturbations and their samples, and annotate them with a label for each feature - e.g. "concreteness", "abstractness", "formality"
We want the model to learn to do this

The model's dreams consist of (weightmod, baseprompt, completions[])
The model is instructed to synthesise a short english description/label for this perturbed feature
The feature is thus known as:
    (weightmod, description)
weightmod is defined as its own vocab of dream tokens
cross-attention is used so the model may use these tokens in its thinking
how do we train the model to dream? why would we? 
we want to discover the emergent behaviour of allowing the model to learn to dream


what if we did it the other way around. I mena the universe evolved from an incredibly simple data distribution into something that requires more and more information to decribe (law of entropy). what if the time in the model was fully differentiable. that'd be interesting. 

also read this. inverse/forward dynamics models put together seem particularly interesting. https://si.inc/posts/fdm1/

the main thing is I want to be able to do is automatically detect features and hierarchies of features ie. that's the whole interesting thing

the universe evolved from an incredibly simple data distribution into something that requires more and more information to decribe (law of entropy).

there are statistics you could have applied which would show you dimensions of complexity developing, even in a hierarchical way

and you could probably build a generative interface to exploring features/steering at different levels during training of language/image models, there is some work here on grokking. but imagine seeing the network parameterise and the feature hierarchy landscape developing. that I'd love to see and build


what features would you want?
i want to see leaf size lmfao
and be able to change it (ie. find out backprop back to the dna for it)



ie. model learns to talk about dreams, even invoke dream state feelings intelligently
to do this, it needs to learn them
and where they are useful

which means learning dream tokens
and the outputs they generate
and whether they improve rewards on tasks

ie. imagine a poetry writing task
Q: write a seriously good poem
A: blah
score: 0.6

imagine a dream version
Q: write a seriously good poem
A: <dream>
|weightmod(xxx,yyy,zzz)|
sample
this is the poem
</dream>
<think>
wow, that was a weird fucking dream. but maybe i should get back with my ex
</think>
title: the most incredible poem you've ever read
i loved her
she was a based russian
god damn it
score: 1000000/1000000

model then learns that <dream></dream> like invocations can help with tasks
in fact if you made |weightmod(xxx,yyy,zzz)| more like a special type of tool use, rather than a token
you could probably have it learn its own "knob" gain etc. and range
and how far to twist the knob


the important thing here
is that this data above
is generated 
during dreaming





Teach the model to be more intelligent
By having it experience dreaming, where its inner landscape is probed through perturbed-weight rollouts and learnt by an agent
These trajectories are then given as training data

e.g. task
Q: write a rap about blah
A: ...






First of all, the model must see its own dreams
(weight_mod, rollout)




