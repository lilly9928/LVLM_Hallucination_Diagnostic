"""Stage 2: coarsened exact matching (CEM) between high- and low-co-occurrence strata.

Why CEM over propensity-score matching: only three covariates are involved, two
of them (marginal frequency, average area) are category-level constants shared
by every candidate of that category, so the "propensity" is largely a function of
*which category was picked as target A*, not of a smooth per-unit score. CEM makes
that structure explicit and auditable (every matched cell can be inspected: "N
treatment / M control units share this frequency/area/CLIP-similarity band") rather
than relying on a fitted propensity model that could silently extrapolate outside
common support -- important for a go/no-go diagnostic where the matching logic
itself must be defensible, not just its output.

Within an exact-matched cell, CLIP similarity is already coarsened to the same
quantile band for every member, so a full nearest-neighbor search adds little:
units are paired by matching rank order (i-th smallest treatment CLIP-sim with
i-th smallest control CLIP-sim) after sorting, which is O(n log n) instead of the
O(n^2) a greedy nearest-neighbor-with-removal search would cost at this sample size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchUnit:
    image_id: int
    category_id: int
    score: float
    freq: float
    area: float
    clip_sim: float


def _quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def _assign_bin(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right") - 1)


def coarsened_exact_match(
    treatment: list[MatchUnit],
    control: list[MatchUnit],
    freq_bins: int,
    area_bins: int,
    clip_bins: int,
) -> dict:
    all_units = treatment + control
    freq_edges = _quantile_bin_edges(np.array([u.freq for u in all_units]), freq_bins)
    area_edges = _quantile_bin_edges(np.array([u.area for u in all_units]), area_bins)
    clip_edges = _quantile_bin_edges(np.array([u.clip_sim for u in all_units]), clip_bins)

    def cell_key(u: MatchUnit) -> tuple[int, int, int]:
        return (
            _assign_bin(u.freq, freq_edges),
            _assign_bin(u.area, area_edges),
            _assign_bin(u.clip_sim, clip_edges),
        )

    treat_by_cell: dict[tuple, list[MatchUnit]] = {}
    control_by_cell: dict[tuple, list[MatchUnit]] = {}
    for u in treatment:
        treat_by_cell.setdefault(cell_key(u), []).append(u)
    for u in control:
        control_by_cell.setdefault(cell_key(u), []).append(u)

    pairs: list[tuple[MatchUnit, MatchUnit, tuple]] = []
    for cell, t_units in treat_by_cell.items():
        c_units = control_by_cell.get(cell)
        if not c_units:
            continue
        t_sorted = sorted(t_units, key=lambda u: u.clip_sim)
        c_sorted = sorted(c_units, key=lambda u: u.clip_sim)
        for t, c in zip(t_sorted, c_sorted):
            pairs.append((t, c, cell))

    n_treatment_dropped = len(treatment) - len(pairs)
    n_control_dropped = len(control) - len(pairs)

    return {
        "pairs": pairs,
        "freq_bin_edges": freq_edges.tolist(),
        "area_bin_edges": area_edges.tolist(),
        "clip_bin_edges": clip_edges.tolist(),
        "n_treatment_total": len(treatment),
        "n_control_total": len(control),
        "n_matched_pairs": len(pairs),
        "n_treatment_dropped_no_cell_overlap_or_excess": n_treatment_dropped,
        "n_control_dropped_no_cell_overlap_or_excess": n_control_dropped,
    }


def _standardized_mean_difference(treat_values: np.ndarray, control_values: np.ndarray) -> float:
    pooled_sd = np.sqrt((np.var(treat_values, ddof=1) + np.var(control_values, ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(treat_values) - np.mean(control_values)) / pooled_sd)


def balance_table(treatment: list[MatchUnit], control: list[MatchUnit]) -> list[dict]:
    covariate_getters = {
        "marginal_freq": lambda u: u.freq,
        "avg_area": lambda u: u.area,
        "clip_sim": lambda u: u.clip_sim,
    }
    rows = []
    for name, getter in covariate_getters.items():
        t_vals = np.array([getter(u) for u in treatment])
        c_vals = np.array([getter(u) for u in control])
        rows.append(
            {
                "covariate": name,
                "n_treatment": len(t_vals),
                "n_control": len(c_vals),
                "treat_mean": float(np.mean(t_vals)),
                "treat_sd": float(np.std(t_vals, ddof=1)),
                "control_mean": float(np.mean(c_vals)),
                "control_sd": float(np.std(c_vals, ddof=1)),
                "smd": _standardized_mean_difference(t_vals, c_vals),
            }
        )
    return rows
