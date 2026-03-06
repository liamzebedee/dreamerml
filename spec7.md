I want to figure out this idea
It would be cool to teach a model to dream
The idea being - a model is trained and its activation space begins to take a certain shape which replicates the statistical structure of the dataset
When questioned to solve a math problem, the model can use the skills it has learnt to do so
This activates certain areas of the model

What I am interested by is in the extreme versions of activation space that cannot be accessed by prompting
As the model has learnt some statistical structure, activation space contains useful learnt features (ie. transformer circuits, monosemanticity)
It is possible to turn these features up/down, or even turn them on simutaneously, to access model states that are inaccessible by normal prompting

Consider dreaming as one example of this
Dreams could be construed as an RL environment
Imagine a dream begins with a topic/prompt - "my day at work tomorrow"
The model is sampled and completes this tradjectory...but things are different and surreal
How? The activation space is modified such that some features are dialed up in extremity
The result? The dreamer is tested in a simulated rollout of a more extreme environment
Their behaviours/decisions remain intact
This rollout generates meaningful signal about the dreamer's subconscious's statistical structure
What are my true attractor basins? What really do I feel/gravitate to? 
The analogue is that the dreamer cannot articulate the feeling of the dream
In the same way the model cannot articulate the activations of its network
But a dream can provide meaningful completions that can be analysed later

The idea is this
What if a model could dream? ie. different regions of activation space are mapped, and then perturbed extremely. And the model is sampled for outputs. And somehow, optimization solves for discovering the shape of activation space. 
And then the model learns these dream configurations
ie. this generates a synthetic dataset like:
    prompt: [dream topic]
    completion: 
    START_DREAM
    DREAM_CONFIG_LENGTH <dream_config_data_binary>
    completed tokens...
    END_DREAM
    any other tokens
    END_OF_MESSAGE

    where the special tokens are:
        START_DREAM: len(vocab) + 1
        END_DREAM: len(vocab) + 2
        DREAM_CONFIG_LENGTH: len(vocab) + 3
    
    and dream_config_data_binary is a binary encoding of the data necessary to create the perturbed activation space. For example, in a naive realisation:
        [indices, factors] = decode_dream_config()
        for i in indices:
            layers[i].activation *= factors[i]

After that is done. The model is then taught how to dream.
So it can use these different activation spaces to solve problems creatively, perhaps in a more token efficient manner.
We run GRPO with the same <|dream|></|dream|> special format
and hopefully emergent behaviour comes out

Read 
https://arxiv.org/pdf/2602.15029











how about this. we add a "virtual architecture" on top of the model. what it does is measure the activations of the network as its happening. 

so normally:
l0 -> l1 -> l2 -> l3 -> l4

now we have

a1    a2    a3    a4      a5
l0 -> l1 -> l2 -> l3 -> l4

and those activation nodes are computed by the activatoins when we run inference l0 to l4
but then almost we run them perpendicular, they have inputs too

x
w1   w2 w3   w4     w5  
a1    a2    a3    a4      a5
l0 -> l1 -> l2 -> l3 -> l4

the idea is that we learn an embedding of activations for all our prompts our model is trained on
so every training sample now becmoes
(base_input, activations_input) 

do you see where I'm going with this?

Another interpretation is that you are learning a manifold model of activation space. Instead of perturbing activations blindly, the auxiliary system learns what typical trajectories look like and how they evolve.









We introduce a virtual architecture layered on top of an existing transformer. The base model executes normally as a sequence of layers (l_0 \rightarrow l_1 \rightarrow l_2 \rightarrow \dots \rightarrow l_L). At each layer we observe the residual activation (h_l). A parallel set of nodes (a_l) is computed from these activations via lightweight projections (a_l = g_l(h_l)). These nodes form a perpendicular pathway that summarizes the internal state of the network as it runs. Conceptually, the transformer computes tokens while the virtual layer computes an embedding of the activation trajectory.

The activation nodes are connected sequentially and can also accept external inputs. The structure therefore becomes two coupled streams: the standard forward computation (h_{l+1} = h_l + f_l(h_l)), and a parallel state evolution (a_l = g_l(h_l, a_{l-1}, x)). The (a_l) vectors form a compact representation of “how the model is thinking” during inference. Because these vectors are low-dimensional relative to the residual stream, they act as a bottlenecked embedding of the model’s activation manifold across layers.

Training treats the activation stream as part of the model’s state. For the original pretraining corpus we run the base model and record the activation embeddings for each example. The effective dataset becomes ((\text{base_input}, \text{activation_embedding})), allowing the virtual architecture to learn the distribution and dynamics of activation trajectories across the training distribution. The result is a learned embedding of the model’s internal computation that evolves alongside the token stream.

During post-training with GRPO, the activation stream becomes a control channel. In addition to generating tokens, the model can emit steering signals through the activation pathway which modify internal states at selected layers. Multiple rollouts are sampled for the same prompt, each with different activation steering trajectories. Rewards are computed on the resulting outputs and used to update the steering policy relative to the group. Over time the model learns when and how to modulate its own activations nonverbally, using the activation embedding space as a compact interface for exploring and exploiting different internal reasoning modes.






Do you think this could be demo'd simply on TinyStories? the main thing I'm curious about is finding a way to demonstrate how it can affect small features. Probably the most interesting thing would be this - already I have some results where we can show changing the outputs of tinystories through changing embeddings, which makes things like stories more/less abstract, or boy/girl protagonist. It would be interesting for the model to learn features automatically. Let's say this-

We compute all the activation embeddings for the entire TinyStories dataset.
Now we train a model which predicts a completion for a gain range of a simple activation embedding and the same prompt. Then we have a set of completions which vary a bit differently. We can use a very smart model to describe how they vary according to the tuning of the gain and what the pattern is. And then that is the label for the feature / activation embedding. 

The trick would be, again, discovering meaningful activations. Meaning - their descriptions are meaningful and sort of "monosemantic" in a way. 

