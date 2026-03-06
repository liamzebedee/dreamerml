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
