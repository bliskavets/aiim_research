"""
reward_cache.py
---------------
Module-level cache that `reward_answer_exact` populates with per-sequence
binary correctness (answer_exact >= threshold) so the custom trainer can
retrieve it in `_compute_loss` for the O+/O- split.

Flow:
  1. TRL calls reward_answer_exact(prompts, completions, answer) during
     generation scoring. We compute the usual scalar reward AND stash a
     bool mask into `_CACHE.mask`.
  2. TRL computes advantages, dispatches _compute_loss.
  3. Our trainer reads `_CACHE.mask` to get the binary O+/O- assignment.

The cache is a single tensor of shape (num_generations * batch_size,).
Subsequent calls overwrite it, so there is no cross-step contamination.
"""

import torch


class _Cache:
    mask: torch.Tensor | None = None

    def set(self, scores: list[float], threshold: float) -> None:
        self.mask = torch.tensor(
            [s >= threshold for s in scores], dtype=torch.bool
        )

    def get(self) -> torch.Tensor | None:
        return self.mask

    def clear(self) -> None:
        self.mask = None


_CACHE = _Cache()
