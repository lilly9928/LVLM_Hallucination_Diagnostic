"""Stage 1: category co-occurrence statistics (raw conditional probability + PMI/lift).

Raw conditional probability P(B|A) is contaminated by marginal frequency (a
category that is common overall will look "co-occurring" with everything), so
PMI/lift is the primary metric. Raw conditional is kept as a sanity check only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CooccurrenceStats:
    category_ids: list[int]  # length K, index order shared by all matrices below
    n_images: int
    marginal_counts: np.ndarray  # shape (K,), int64
    joint_counts: np.ndarray  # shape (K, K), int64, symmetric, diagonal = marginal_counts
    pmi: np.ndarray  # shape (K, K), float, symmetric, diagonal NaN, -inf where joint_count == 0
    lift: np.ndarray  # shape (K, K), float, symmetric, diagonal NaN, 0.0 where joint_count == 0
    conditional: np.ndarray  # shape (K, K), float, conditional[a, b] = P(b | a); ASYMMETRIC, diagonal NaN


def compute_cooccurrence_stats(
    category_ids: list[int], image_categories: dict[int, set[int]]
) -> CooccurrenceStats:
    k = len(category_ids)
    index_of = {cid: i for i, cid in enumerate(category_ids)}
    n_images = len(image_categories)

    marginal_counts = np.zeros(k, dtype=np.int64)
    joint_counts = np.zeros((k, k), dtype=np.int64)

    for present in image_categories.values():
        idxs = sorted(index_of[c] for c in present if c in index_of)
        for i in idxs:
            marginal_counts[i] += 1
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                i, j = idxs[a_pos], idxs[b_pos]
                joint_counts[i, j] += 1
                joint_counts[j, i] += 1

    p_a = marginal_counts / n_images
    p_ab = joint_counts / n_images

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(p_a, p_a)
        pmi = np.log(p_ab / denom)
        conditional = joint_counts / marginal_counts[:, None]

    np.fill_diagonal(pmi, np.nan)
    np.fill_diagonal(conditional, np.nan)

    lift = np.where(joint_counts > 0, np.exp(pmi), 0.0)
    np.fill_diagonal(lift, np.nan)

    return CooccurrenceStats(
        category_ids=list(category_ids),
        n_images=n_images,
        marginal_counts=marginal_counts,
        joint_counts=joint_counts,
        pmi=pmi,
        lift=lift,
        conditional=conditional,
    )


def get_top_bottom_pairs(
    stats: CooccurrenceStats,
    category_names: dict[int, str],
    min_support: int,
    top_k: int,
) -> dict:
    """Rank category pairs by PMI, restricted to pairs with joint_count >= min_support.

    Pairs below min_support (including joint_count == 0, where PMI is -inf) are
    excluded from ranking only -- the underlying matrices returned by
    compute_cooccurrence_stats are never filtered or altered.
    """
    k = len(stats.category_ids)
    rows: list[dict] = []
    for i in range(k):
        for j in range(i + 1, k):
            joint = int(stats.joint_counts[i, j])
            rows.append(
                {
                    "category_a": category_names[stats.category_ids[i]],
                    "category_b": category_names[stats.category_ids[j]],
                    "joint_count": joint,
                    "pmi": float(stats.pmi[i, j]),
                    "lift": float(stats.lift[i, j]),
                    "p_b_given_a": float(stats.conditional[i, j]),
                    "p_a_given_b": float(stats.conditional[j, i]),
                }
            )

    eligible = [r for r in rows if r["joint_count"] >= min_support and math.isfinite(r["pmi"])]
    n_excluded = len(rows) - len(eligible)

    top = sorted(eligible, key=lambda r: r["pmi"], reverse=True)[:top_k]
    bottom = sorted(eligible, key=lambda r: r["pmi"])[:top_k]

    return {
        "top": top,
        "bottom": bottom,
        "n_pairs_total": len(rows),
        "n_pairs_excluded_by_min_support": n_excluded,
        "min_support": min_support,
    }


def iter_category_names(category_ids: Iterable[int], names_by_id: dict[int, str]) -> list[str]:
    return [names_by_id[cid] for cid in category_ids]
