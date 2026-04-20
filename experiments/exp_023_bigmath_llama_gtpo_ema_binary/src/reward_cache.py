"""
reward_cache.py
---------------
Module-level cache populated by reward_answer_exact and consumed by the
custom GTPO-EMA trainer to get the binary O+/O- split based on
answer_exact >= threshold.
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
