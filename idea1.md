

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

(RL with GRPO obviously)

model then learns that <dream></dream> like invocations can help with tasks
in fact if you made |weightmod(xxx,yyy,zzz)| more like a special type of tool use, rather than a token
you could probably have it learn its own "knob" gain etc. and range
and how far to twist the knob


the important thing here
is that this data above
is generated 
during dreaming


not more parameters, not more data — more cognitive diversity through learned self-perturbation.
  For math: the model is stuck on a proof step. Its default weights assign near-zero probability to the key substitution. It dreams — perturbs its weights — and in that altered state, the substitution becomes reachable. It samples it. It wakes up, reads what it wrote, and
   goes "oh." The insight transfers even though the weights revert, because it's now in the context window.




## how are dreams generated etc
the model is put inside an environment - dreaming
the model acts without knowing its a dream - generating tokens - completions/trajectories
the subconscious is the agent which orhcestrates this environment
the subconscious is the dreamer, it controls the dream
the subconscious takes actions in the form of perturbing weights
it then receives rollouts of the model dreaming and its experience (tokens)
it uses this as a probe into the model's structure - how different perturbations affect a token stream
its reward is defined as learning a compact description of the model's weights
    ie. a model's weights at initialisation can be described simply by a gaussian parameterisation
    though as a model is trained, the weight's minimum description lenght increases, as it develops statistical structure
the subconscious learns a world model which predicts perturbation -> statistical measures of variatoin
it basically is trying to learn hierarchies of features in the weights


the system is setup as:

1. dreamer maps the landscape
    it puts the model through dreams
    it measures the model's outputs
    it learns the landscape of the model's weights
2. the model learns to invoke itself


