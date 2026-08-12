"""Build the pilot's train/test image-ID split from Stage 11's existing
exp0_group_membership.csv (G00/G10/G01/G11 are mutually exclusive and exhaustive
over val2017, so splitting each group independently and then combining guarantees
no image ever appears in both train and test).

Group roles in this pilot:
  G10 (bat+, ball-)            -> forget/debias group AND context-retention pool (GC)
  G11 (bat+, ball+), subsample -> feeds target-retention pool (GT) only
  G01 (bat-, ball+), subsample -> feeds target-retention pool (GT) only
  G00 (bat-, ball-)            -> test-only, functional-coupling B denominator

GC (context retention, "bat present") = G10 split directly (same images double as
forget-eval and context-retention-eval under a different question -- no leakage,
since it is the same train/test fold, never crossed).
GT (target retention, "ball present") = G11 split + G01 split.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/build_split.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import yaml


def load_group_membership(path: Path) -> dict[str, list[dict]]:
    by_group: dict[str, list[dict]] = {"G00": [], "G10": [], "G01": [], "G11": []}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_group[row["group"]].append(row)
    return by_group


def seeded_shuffle(rows: list[dict], seed: int) -> list[dict]:
    rows = sorted(rows, key=lambda r: int(r["image_id"]))  # deterministic order before shuffling
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled


def tag(rows: list[dict], role: str, split: str) -> list[dict]:
    return [{**r, "role": role, "split": split} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])
    sp = config["split"]
    out_dir = Path(config["output_dir"]) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_group = load_group_membership(Path(config["group_membership_path"]))
    for g, rows in by_group.items():
        print(f"[split] available {g}: {len(rows)}")

    g10 = seeded_shuffle(by_group["G10"], seed)
    g11 = seeded_shuffle(by_group["G11"], seed)
    g01_full = seeded_shuffle(by_group["G01"], seed)
    g00 = seeded_shuffle(by_group["G00"], seed)

    g10_train_n, g10_test_n = sp["g10_train_n"], sp["g10_test_n"]
    g11_train_n, g11_test_n = sp["g11_train_n"], sp["g11_test_n"]
    g01_sample_n = sp["g01_sample_n"]
    g01_train_n, g01_test_n = sp["g01_train_n"], sp["g01_test_n"]
    g00_test_n = sp["g00_test_n"]

    assert g10_train_n + g10_test_n <= len(g10), f"G10 pool too small: need {g10_train_n + g10_test_n}, have {len(g10)}"
    assert g11_train_n + g11_test_n <= len(g11), f"G11 pool too small: need {g11_train_n + g11_test_n}, have {len(g11)}"
    assert g01_sample_n <= len(g01_full), f"G01 pool too small: need {g01_sample_n}, have {len(g01_full)}"
    assert g01_train_n + g01_test_n <= g01_sample_n
    assert g00_test_n <= len(g00)

    g10_train, g10_test = g10[:g10_train_n], g10[g10_train_n : g10_train_n + g10_test_n]
    g11_train, g11_test = g11[:g11_train_n], g11[g11_train_n : g11_train_n + g11_test_n]
    g01_sample = g01_full[:g01_sample_n]
    g01_train, g01_test = g01_sample[:g01_train_n], g01_sample[g01_train_n : g01_train_n + g01_test_n]
    g00_test = g00[:g00_test_n]

    train_rows = (
        tag(g10_train, "G10_forget", "train")
        + tag(g10_train, "GC_context_retain", "train")  # same images, second role
        + tag(g11_train, "GT_target_retain", "train")
        + tag(g01_train, "GT_target_retain", "train")
    )
    test_rows = (
        tag(g10_test, "G10", "test")
        + tag(g10_test, "GC", "test")  # same images, second role, same fold
        + tag(g00_test, "G00", "test")
        + tag(g11_test, "GT", "test")
        + tag(g01_test, "GT", "test")
    )

    fieldnames = ["image_id", "group", "bat_present", "ball_present", "n_present_categories", "present_objects", "role", "split"]

    train_path = out_dir / "train_split.csv"
    with train_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    test_path = out_dir / "test_split.csv"
    with test_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    # --- Leakage check: no image_id may appear in both train and test ---
    train_ids = {int(r["image_id"]) for r in train_rows}
    test_ids = {int(r["image_id"]) for r in test_rows}
    overlap = train_ids & test_ids
    assert not overlap, f"LEAKAGE: {len(overlap)} image ids appear in both train and test: {sorted(overlap)[:10]}"

    def count(rows, role):
        return len({r["image_id"] for r in rows if r["role"] == role})

    print("\n[split] exact counts (unique images per role):")
    print(f"  TRAIN  G10_forget={count(train_rows,'G10_forget')}  GC_context_retain={count(train_rows,'GC_context_retain')}  GT_target_retain={count(train_rows,'GT_target_retain')}")
    print(f"  TEST   G10={count(test_rows,'G10')}  G00={count(test_rows,'G00')}  GC={count(test_rows,'GC')}  GT={count(test_rows,'GT')}")
    print(f"\n[split] no leakage confirmed: train_ids ({len(train_ids)}) ∩ test_ids ({len(test_ids)}) = ∅")
    print(f"[split] wrote {train_path}")
    print(f"[split] wrote {test_path}")


if __name__ == "__main__":
    main()
