"""
reward_cache.py
---------------
Module-level cache filled by `reward_answer_exact` at generation-scoring
time and read by GTPOEMAFlippedTrainer during _compute_loss.

Carries a per-sequence boolean mask (O+ membership) for the current step.
Adapted from exp_022's reward_cache to the same invariant: `_CACHE.mask`
is a tensor of shape (num_generations * batch_size,), overwritten each
step so there is no cross-step contamination.
"""

import torch


class _Cache:
    mask: torch.Tensor | None = None

    def set(self, scores: list[float], threshold: float) -> None:
        self.mask = torch.tensor(
            [s >= threshold for s in scores], dtype=torch.bool,
        )

    def get(self) -> torch.Tensor | None:
        return self.mask

    def clear(self) -> None:
        self.mask = None


_CACHE = _Cache()
