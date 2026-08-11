"""Two-phase search for the minimal L-inf budget epsilon* that flips a targeted
"Is there a {A} in the image?" response from No to Yes.

A 30-sample pilot with a fixed-resolution linear bisection (grid = 1/255,
matching 8-bit pixel quantization) found that nearly every non-hallucinating
sample flipped already at the single coarsest grid point tried (1/255) -- the
true thresholds live on a scale the fixed grid could not resolve, saturating
the search and destroying the co-occurrence effect the experiment measures.

Phase 1 (exponential bracketing): starting from a small `eps0`, double the
budget until the attack flips (or epsilon_max is exceeded) -- this adapts to
whatever scale the true threshold lives at instead of assuming it in advance.
Phase 2 (relative bisection): bisect the resulting [lo, hi] bracket until hi is
accurate to within `relative_tolerance` of the true threshold (multiplicative
precision, since the scale is not known ahead of time).

Every outcome is recorded, never discarded:
  - status="already_yes": model already answers Yes at epsilon=0.
  - status="flipped":     epsilon* is the smallest CONFIRMED-flipping value
                           found, accurate to within `relative_tolerance`.
  - status="censored":    did not flip even at epsilon_max -- recorded, not dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EpsilonStarResult:
    status: str  # "already_yes" | "flipped" | "censored"
    epsilon_star: float | None
    epsilon_max: float
    n_attack_calls: int
    trace: list[dict] = field(default_factory=list)


def find_epsilon_star(
    attack_at_epsilon: Callable[[float], dict],
    baseline_response: dict,
    epsilon_max: float,
    eps0: float,
    relative_tolerance: float = 0.1,
    max_bisection_steps: int = 30,
) -> EpsilonStarResult:
    trace = [{"epsilon": 0.0, **baseline_response}]
    if baseline_response["flipped"]:
        return EpsilonStarResult(
            status="already_yes", epsilon_star=0.0, epsilon_max=epsilon_max, n_attack_calls=0, trace=trace
        )

    n_calls = 0
    lo = 0.0
    hi = None
    eps = eps0
    while eps < epsilon_max:
        result = attack_at_epsilon(eps)
        trace.append({"epsilon": eps, **result})
        n_calls += 1
        if result["flipped"]:
            hi = eps
            break
        lo = eps
        eps *= 2.0

    if hi is None:
        result = attack_at_epsilon(epsilon_max)
        trace.append({"epsilon": epsilon_max, **result})
        n_calls += 1
        if not result["flipped"]:
            return EpsilonStarResult(
                status="censored", epsilon_star=None, epsilon_max=epsilon_max, n_attack_calls=n_calls, trace=trace
            )
        hi = epsilon_max

    for _ in range(max_bisection_steps):
        if (hi - lo) <= relative_tolerance * hi:
            break
        mid = (lo + hi) / 2.0
        result = attack_at_epsilon(mid)
        trace.append({"epsilon": mid, **result})
        n_calls += 1
        if result["flipped"]:
            hi = mid
        else:
            lo = mid

    return EpsilonStarResult(status="flipped", epsilon_star=hi, epsilon_max=epsilon_max, n_attack_calls=n_calls, trace=trace)
