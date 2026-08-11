"""Shrink Stage 2's full matched-pair set to a Stage-3-affordable size via
proportional (largest-remainder) stratified subsampling on CEM cell, so the
smaller run keeps the same covariate balance the full CEM match guaranteed.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/subsample_matched_pairs.py \
        --input outputs/.../stage2_sampling_matching/matched_pairs.csv \
        --output outputs/.../stage3_attack/matched_pairs_subsample_150.csv \
        --target-n 150 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.stratified_subsample import stratified_subsample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stratified subsample of Stage 2 matched pairs for Stage 3")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sampled = stratified_subsample(rows, ["freq_bin", "area_bin", "clip_bin"], args.target_n, random.Random(args.seed))
    print(f"[subsample] {len(rows)} matched pairs -> {len(sampled)} (target {args.target_n})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sampled)
    print(f"[subsample] written to {args.output}")


if __name__ == "__main__":
    main()
