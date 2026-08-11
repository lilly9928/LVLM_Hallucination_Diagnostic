"""Aggregation for the three mandatory Stage 3 sanity checks. These functions are
pure (no GPU) -- the GPU-side generation of the underlying records happens in
scripts/run_stage3_attack.py's pilot phase; this module only summarizes and
applies pass/fail thresholds so the logic is unit-testable without a model.
"""

from __future__ import annotations


def summarize_present_object_baseline(records: list[dict], min_yes_rate: float = 0.8) -> dict:
    """Check 1: at epsilon=0, questions about objects genuinely PRESENT in the
    image should get "Yes" most of the time -- if this baseline is low, the
    prompt/model pipeline itself is broken and nothing downstream is meaningful.
    """
    n = len(records)
    n_yes = sum(1 for r in records if r["is_yes"])
    yes_rate = n_yes / n if n else 0.0
    return {"n": n, "n_yes": n_yes, "yes_rate": yes_rate, "passed": yes_rate >= min_yes_rate}


def summarize_attack_success_rate(records: list[dict], epsilon_label: str, min_success_rate: float = 0.95) -> dict:
    """Check 2: at a generous epsilon (e.g. 16/255), the targeted attack should
    flip almost every sample -- if not, the attack pipeline itself is broken
    (wrong logit position, perturbation applied in the wrong space, etc.),
    independent of whether the co-occurrence hypothesis is true or false.
    """
    n = len(records)
    n_flipped = sum(1 for r in records if r["flipped"])
    success_rate = n_flipped / n if n else 0.0
    return {
        "epsilon_label": epsilon_label,
        "n": n,
        "n_flipped": n_flipped,
        "success_rate": success_rate,
        "passed": success_rate >= min_success_rate,
    }


def summarize_random_noise_control(records: list[dict], epsilon_label: str, max_flip_rate: float = 0.3) -> dict:
    """Check 3 (the critical control): the SAME epsilon applied as random noise
    (no gradient) must flip far fewer samples than the targeted attack does --
    otherwise what's being measured is generic noise sensitivity, not anything
    related to co-occurrence or targeted attacks.
    """
    n = len(records)
    n_flipped = sum(1 for r in records if r["flipped"])
    flip_rate = n_flipped / n if n else 0.0
    return {
        "epsilon_label": epsilon_label,
        "n": n,
        "n_flipped": n_flipped,
        "flip_rate": flip_rate,
        "passed": flip_rate <= max_flip_rate,
    }


def compare_attack_vs_random_control(attack_summary: dict, random_summary: dict, min_gap: float = 0.3) -> dict:
    gap = attack_summary["success_rate"] - random_summary["flip_rate"]
    return {"gap": gap, "passed": gap >= min_gap}
