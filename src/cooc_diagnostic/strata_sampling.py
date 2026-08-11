"""Stage 2: enumerate (val2017 image, absent target category) candidates and
split them into high-/low-co-occurrence strata.

For a candidate (image with present set Y, absent target category A), the
co-occurrence score is mean_{y in Y, count(A,y) >= min_support} PMI(A, y).
Present-object terms with too few training co-occurrences are excluded from the
mean (same min_support principle as Stage 1's top/bottom ranking) rather than
included as -inf, so that a single historically-unseen pairing cannot force the
whole image's score to -inf regardless of its other present objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    image_id: int
    category_id: int
    score: float
    n_present: int
    n_pmi_terms_used: int


def build_candidates(
    image_categories: dict[int, set[int]],
    eligible_category_ids: list[int],
    category_index: dict[int, int],
    pmi: np.ndarray,
    joint_counts: np.ndarray,
    min_support: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for image_id, present in image_categories.items():
        if not present:
            continue
        present_idxs = [category_index[c] for c in present if c in category_index]
        if not present_idxs:
            continue
        for cat_id in eligible_category_ids:
            if cat_id in present:
                continue
            a_idx = category_index[cat_id]
            terms = [
                float(pmi[a_idx, y_idx])
                for y_idx in present_idxs
                if joint_counts[a_idx, y_idx] >= min_support
            ]
            if not terms:
                continue
            candidates.append(
                Candidate(
                    image_id=image_id,
                    category_id=cat_id,
                    score=float(np.mean(terms)),
                    n_present=len(present),
                    n_pmi_terms_used=len(terms),
                )
            )
    return candidates


def split_strata(
    candidates: list[Candidate], lower_pct: float, upper_pct: float
) -> tuple[list[Candidate], list[Candidate], dict]:
    """Bottom `lower_pct` -> control (low co-occurrence); top (100 - upper_pct) -> treatment."""
    scores = np.array([c.score for c in candidates])
    low_cut = float(np.percentile(scores, lower_pct))
    high_cut = float(np.percentile(scores, upper_pct))

    treatment = [c for c in candidates if c.score >= high_cut]
    control = [c for c in candidates if c.score <= low_cut]
    n_middle_dropped = len(candidates) - len(treatment) - len(control)

    info = {
        "low_cut": low_cut,
        "high_cut": high_cut,
        "n_candidates_total": len(candidates),
        "n_treatment": len(treatment),
        "n_control": len(control),
        "n_middle_dropped": n_middle_dropped,
    }
    return treatment, control, info
