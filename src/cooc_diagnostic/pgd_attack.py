"""L-inf PGD attack in [0,1] pixel space, applied BEFORE CLIP normalization.

LLaVA's image processor resizes/pads and center-crops in PIL space, then
normalizes to CLIP statistics as the final step before the vision tower sees the
tensor. The perturbation here is added to the already resized/cropped [0,1]
float tensor -- i.e. exactly the pixel grid the vision tower consumes -- and is
clamped back into [0,1] every step. CLIP normalization is applied only inside the
caller-supplied `margin_fn`, never here, so this module never touches normalized
values (the classic bug this experiment is guarding against).

The PGD loop optimizes a differentiable proxy (margin = logit(yes) - logit(no))
purely to drive the search efficiently; it is NOT the authoritative flip
decision. The caller must separately verify the actual greedy-decoded answer text
on `best_image` -- see epsilon_star.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class PGDResult:
    flipped_by_margin: bool  # margin > 0 was reached; a proxy signal only, not authoritative
    best_image: torch.Tensor
    best_margin: float
    n_steps_used: int


def pgd_attack(
    image01: torch.Tensor,
    epsilon: float,
    n_steps: int,
    margin_fn: Callable[[torch.Tensor], torch.Tensor],
    step_size: float | None = None,
    init: str = "zero",
) -> PGDResult:
    if epsilon <= 0:
        margin = margin_fn(image01)
        return PGDResult(flipped_by_margin=float(margin) > 0, best_image=image01.clone(), best_margin=float(margin), n_steps_used=0)

    if step_size is None:
        step_size = 2.5 * epsilon / n_steps

    if init == "random":
        delta = (torch.rand_like(image01) * 2 - 1) * epsilon
    else:
        delta = torch.zeros_like(image01)
    delta = torch.clamp(image01 + delta, 0.0, 1.0) - image01

    best_margin = -float("inf")
    best_image = image01.clone()
    flipped_by_margin = False
    n_used = 0

    for step in range(n_steps):
        delta = delta.detach().requires_grad_(True)
        adv_image = torch.clamp(image01 + delta, 0.0, 1.0)
        margin = margin_fn(adv_image)
        n_used = step + 1

        margin_value = float(margin.detach())
        if margin_value > best_margin:
            best_margin = margin_value
            best_image = adv_image.detach().clone()
        if margin_value > 0:
            flipped_by_margin = True
            break

        margin.backward()
        with torch.no_grad():
            grad_sign = delta.grad.sign()
            delta = delta + step_size * grad_sign
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(image01 + delta, 0.0, 1.0) - image01

    return PGDResult(
        flipped_by_margin=flipped_by_margin,
        best_image=best_image,
        best_margin=best_margin,
        n_steps_used=n_used,
    )


def pgd_attack_with_restarts(
    image01: torch.Tensor,
    epsilon: float,
    n_steps: int,
    margin_fn: Callable[[torch.Tensor], torch.Tensor],
    n_restarts: int,
) -> PGDResult:
    """Restart 0 starts from the clean image (standard); restarts 1..N-1 start
    from a random point in the epsilon ball (guards against PGD's local search
    missing a viable flip from an unlucky/degenerate zero-init trajectory --
    standard practice in adversarial robustness evaluation)."""
    best: PGDResult | None = None
    for r in range(max(1, n_restarts)):
        init = "zero" if r == 0 else "random"
        result = pgd_attack(image01, epsilon, n_steps, margin_fn, init=init)
        if best is None or result.best_margin > best.best_margin:
            best = result
        if result.flipped_by_margin:
            return result
    assert best is not None
    return best
