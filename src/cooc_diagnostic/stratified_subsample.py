"""Proportional (largest-remainder) stratified subsampling, used to shrink
Stage 2's full matched-pair set down to a Stage-3-affordable size while keeping
each CEM cell's share of the total the same as in the full matched set --
otherwise a smaller run could silently drift the covariate balance CEM matching
was built to guarantee.
"""

from __future__ import annotations

import random
from collections import defaultdict


def stratified_subsample(rows: list[dict], cell_key_fields: list[str], target_n: int, rng: random.Random) -> list[dict]:
    if target_n >= len(rows):
        return list(rows)

    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[tuple(row[f] for f in cell_key_fields)].append(row)

    total = len(rows)
    quotas: dict[tuple, int] = {}
    remainders: list[tuple[float, tuple]] = []
    for cell, cell_rows in by_cell.items():
        exact = target_n * len(cell_rows) / total
        base = int(exact)
        quotas[cell] = min(base, len(cell_rows))
        remainders.append((exact - base, cell))

    allocated = sum(quotas.values())
    remainders.sort(key=lambda x: x[0], reverse=True)
    i = 0
    while allocated < target_n and i < len(remainders):
        _, cell = remainders[i]
        if quotas[cell] < len(by_cell[cell]):
            quotas[cell] += 1
            allocated += 1
        i += 1

    sampled = []
    for cell, cell_rows in by_cell.items():
        k = quotas[cell]
        sampled.extend(rng.sample(cell_rows, k) if k > 0 else [])
    return sampled
