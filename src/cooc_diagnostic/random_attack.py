"""Uniform random L-inf perturbations -- the mandatory random-noise control that
the gradient-based PGD attack's success rate must be compared against (Stage 3's
most important sanity check: if random noise flips about as often as the
targeted attack, the pipeline is measuring generic noise sensitivity, not
anything related to targeted co-occurrence-driven attacks).
"""

from __future__ import annotations

import torch


def random_perturbations(image01: torch.Tensor, epsilon: float, n_trials: int) -> list[torch.Tensor]:
    """Returns `n_trials` independent images, each `image01` perturbed by
    uniform noise within the L-inf epsilon ball and clamped back to [0,1]."""
    images = []
    for _ in range(n_trials):
        delta = (torch.rand_like(image01) * 2 - 1) * epsilon
        images.append(torch.clamp(image01 + delta, 0.0, 1.0))
    return images
