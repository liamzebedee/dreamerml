"""
Dream execution hook: on <dream> call planner then generate dream text.
"""

import torch
from env import DreamEnv, N_DREAM
from state import StateExtractor, ContextProjector
from planner import Planner
from forward_model import ForwardModel
from inverse_model import InverseModel
from collect_transitions import check_quality


class DreamExecutor:
    """Execute planned dreams during generation."""

    def __init__(self, env: DreamEnv, extractor: StateExtractor,
                 projector: ContextProjector, planner: Planner,
                 max_dreams: int = 2, device="cuda"):
        self.env = env
        self.extractor = extractor
        self.projector = projector
        self.planner = planner
        self.max_dreams = max_dreams
        self.device = device
        self.dream_count = 0

    def reset(self):
        self.dream_count = 0
        self.extractor.reset_history()

    def execute_dream(self, context_ids: torch.Tensor, prompt_ids: torch.Tensor) -> str:
        """
        Execute a single dream.

        context_ids: full current context (prompt + generated so far)
        prompt_ids: original prompt only (for context embedding)
        Returns: dream text (or empty string if degenerate/limit reached)
        """
        if self.dream_count >= self.max_dreams:
            return ""

        # Get state
        s_t = self.extractor.extract(context_ids, normalize=True)

        # Get context embedding
        with torch.no_grad():
            c_raw = self.extractor.extract_context_embedding(prompt_ids)
            c = self.projector(c_raw).squeeze(0)

        # Plan action
        a = self.planner.plan(s_t, c)

        # Apply and generate
        self.env.apply_action(a)
        dream_tokens = self.env.generate(context_ids, n_tokens=N_DREAM)
        self.env.revert_action()

        # Quality check
        quality = check_quality(dream_tokens, self.env)
        if quality == 0:
            self.dream_count += 1
            return ""  # degenerate, insert nothing

        dream_text = self.env.decode(dream_tokens)
        self.dream_count += 1
        return dream_text

    @torch.no_grad()
    def generate_with_dreams(self, prompt: str, max_tokens: int = 200,
                              force_dream_at: list = None) -> dict:
        """
        Generate text with optional dream insertions.

        force_dream_at: list of token positions to force a dream (for testing)
        Returns dict with full text, dream texts, and metadata
        """
        self.reset()
        prompt_ids = self.env.encode(prompt)
        context_ids = prompt_ids.clone()

        generated_text = ""
        dream_texts = []
        dream_positions = []

        token_count = 0
        while token_count < max_tokens:
            # Check if we should dream
            should_dream = False
            if force_dream_at and token_count in force_dream_at:
                should_dream = True

            if should_dream and self.dream_count < self.max_dreams:
                dream_text = self.execute_dream(context_ids, prompt_ids)
                if dream_text:
                    dream_texts.append(dream_text)
                    dream_positions.append(token_count)
                    # Insert dream text into context
                    dream_ids = self.env.encode(dream_text)
                    context_ids = torch.cat([context_ids, dream_ids], dim=-1)
                    generated_text += f" [DREAM: {dream_text}] "

            # Generate one token
            new_token = self.env.generate(context_ids, n_tokens=1)
            context_ids = torch.cat([context_ids, new_token.unsqueeze(0) if new_token.dim() == 1 else new_token], dim=-1)
            generated_text += self.env.decode(new_token)
            token_count += 1

        return {
            "prompt": prompt,
            "full_text": generated_text,
            "dream_texts": dream_texts,
            "dream_positions": dream_positions,
            "n_dreams": len(dream_texts),
        }


def load_dream_executor(
    checkpoint_dir: str = "checkpoints",
    data_dir: str = "data",
    device: str = "cuda",
) -> DreamExecutor:
    """Load all components and create dream executor."""
    env = DreamEnv(device=device)
    extractor = StateExtractor(env.model, device=device)

    # Load projector
    projector = ContextProjector(env.model.config.hidden_size).to(device)
    proj_path = f"{data_dir}/projector.pt"
    import os
    if os.path.exists(proj_path):
        projector.load_state_dict(torch.load(proj_path, map_location=device))

    # Load normalizer
    norm_path = f"{data_dir}/normalizer.pt"
    if os.path.exists(norm_path):
        extractor.normalizer.load_state_dict(torch.load(norm_path, map_location=device), device=device)
        extractor.normalizer.freeze()

    # Load forward/inverse models
    fwd = ForwardModel().to(device)
    fwd.load_state_dict(torch.load(f"{checkpoint_dir}/forward_model.pt", map_location=device))

    inv = InverseModel().to(device)
    inv.load_state_dict(torch.load(f"{checkpoint_dir}/inverse_model.pt", map_location=device))

    planner = Planner(fwd, inv, device)
    return DreamExecutor(env, extractor, projector, planner, device=device)
