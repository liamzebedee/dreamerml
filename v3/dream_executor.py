"""Dream execution: state → policy → perturb weights → generate → revert.

The dream operator temporarily steps the model into a nearby weight
configuration, generates alternative reasoning, then reverts.
"""

import torch
from state import extract_state


class DreamExecutor:

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    @torch.inference_mode()
    def dream(self, input_ids, attention_mask,
              max_tokens=48, temperature=0.9, top_p=0.95):
        """Execute one dream. Returns (dream_text, action, state)."""
        out = self.env.model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        state = extract_state(out.logits, out.hidden_states, attention_mask)
        action = self.policy.deterministic(state)

        deltas = self.env.lora.compute_deltas(action)
        self.env.apply_perturbation(deltas)
        try:
            gen = self.env.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature, top_p=top_p, do_sample=True,
                pad_token_id=self.env.tokenizer.pad_token_id,
            )
        finally:
            self.env.remove_perturbation(deltas)

        n = input_ids.shape[1]
        text = self.env.tokenizer.decode(gen[0, n:], skip_special_tokens=True)
        return text, action, state

    @torch.inference_mode()
    def dream_batch(self, prompts,
                    max_tokens=48, temperature=0.9, top_p=0.95):
        """Dream for multiple prompts using shared action."""
        inputs = self.env.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=32,
        ).to(self.env.device)

        out = self.env.model(**inputs, output_hidden_states=True)
        state = extract_state(
            out.logits, out.hidden_states, inputs["attention_mask"]
        )
        action = self.policy.deterministic(state)
        texts = self.env.generate(
            action, prompts=prompts,
            max_new_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        return texts, action
