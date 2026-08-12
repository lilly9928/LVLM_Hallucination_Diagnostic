"""Part I-II (dataset + adversarial exposure), reused rather than regenerated.

This pilot's fixed case, split methodology (seed=42, split independently
within each of the 4 disjoint COCO groups G00/G10/G01/G11), and adversarial
attack (PGD, epsilon=16/255) are IDENTICAL to the prior
`adversarial_functional_debiasing_pilot`, which already produced them -- see
audit/repository_audit.md. Rerunning would not change any number and would
risk a different fixed epsilon reading as post-hoc tuning. This script copies
those artifacts verbatim (with provenance recorded) and additionally builds
the TRAIN-only dev/val split (70/30) required by Part VII for component
selection, which the prior pilot never needed.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import yaml


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dev_val_split(image_ids: list[int], seed: int, dev_fraction: float) -> dict[int, str]:
    ids = sorted(set(image_ids))
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_dev = round(len(shuffled) * dev_fraction)
    dev_ids = set(shuffled[:n_dev])
    return {iid: ("dev" if iid in dev_ids else "val") for iid in ids}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    prior_dir = Path(config["prior_pilot_dir"])
    data_dir = out_dir / "data"
    adv_img_dir = out_dir / "adversarial_images"
    data_dir.mkdir(parents=True, exist_ok=True)
    adv_img_dir.mkdir(parents=True, exist_ok=True)

    # --- Reuse train/test split verbatim ---
    train_rows = read_csv(prior_dir / "data" / "train_split.csv")
    test_rows = read_csv(prior_dir / "data" / "test_split.csv")
    fieldnames = ["image_id", "group", "bat_present", "ball_present", "n_present_categories", "present_objects", "role", "split"]
    write_csv(data_dir / "train_split.csv", train_rows, fieldnames)
    write_csv(data_dir / "test_split.csv", test_rows, fieldnames)
    print(f"[prepare-data] reused train_split.csv ({len(train_rows)} rows) and test_split.csv ({len(test_rows)} rows) from {prior_dir}")

    # --- Reuse adversarial forget set + images verbatim (paths repointed to
    # this pilot's own adversarial_images/ dir for self-containment) ---
    adv_rows = read_csv(prior_dir / "data" / "adversarial_forget_set.csv")
    adv_fieldnames = list(adv_rows[0].keys())

    n_copied = 0
    for src in sorted((prior_dir / "adversarial_images").glob("*.png")):
        dst = adv_img_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        n_copied += 1

    for r in adv_rows:
        r["adv_image_path"] = str(adv_img_dir / Path(r["adv_image_path"]).name)
    write_csv(data_dir / "adversarial_forget_set.csv", adv_rows, adv_fieldnames)
    print(f"[prepare-data] reused adversarial_forget_set.csv ({len(adv_rows)} rows) and {n_copied} adversarial images from {prior_dir}")

    success_rate = sum(r["attack_success"] == "True" for r in adv_rows) / len(adv_rows)
    mean_clean = sum(float(r["clean_s_ball"]) for r in adv_rows) / len(adv_rows)
    mean_adv = sum(float(r["adv_s_ball"]) for r in adv_rows) / len(adv_rows)
    mean_delta = sum(float(r["delta_s_ball"]) for r in adv_rows) / len(adv_rows)
    print(f"[prepare-data] N={len(adv_rows)} attack_success_rate={success_rate:.3f} "
          f"mean_clean_s_ball={mean_clean:.4f} mean_adv_s_ball={mean_adv:.4f} mean_delta_s_ball={mean_delta:.4f}")

    # --- Sample manifest: combine train+test with this pilot's G10/G00/GT/GC naming ---
    manifest_rows = []
    for r in train_rows:
        manifest_rows.append({**r, "fold": "train", "provenance": "reused_from_adversarial_functional_debiasing_pilot"})
    for r in test_rows:
        manifest_rows.append({**r, "fold": "test", "provenance": "reused_from_adversarial_functional_debiasing_pilot"})
    manifest_fieldnames = fieldnames + ["fold", "provenance"]
    write_csv(data_dir / "sample_manifest.csv", manifest_rows, manifest_fieldnames)
    print(f"[prepare-data] wrote sample_manifest.csv ({len(manifest_rows)} rows)")

    # --- Build TRAIN-only dev/val split (70/30) for component selection (Part VII) ---
    # GC_context_retain uses the SAME images as G10_forget (same fold, second
    # question) in the reused split -- so its dev/val assignment must mirror
    # G10_forget's exactly, not be drawn independently.
    dv_cfg = config["dev_val_split"]
    seed = int(dv_cfg["seed"])
    dev_fraction = float(dv_cfg["dev_fraction"])

    g10_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "G10_forget"})
    gt_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "GT_target_retain"})

    g10_split = build_dev_val_split(g10_ids, seed, dev_fraction)
    gt_split = build_dev_val_split(gt_ids, seed, dev_fraction)

    dev_val_rows = []
    for iid, dv in g10_split.items():
        dev_val_rows.append({"image_id": iid, "role": "G10_forget", "dev_val": dv})
        dev_val_rows.append({"image_id": iid, "role": "GC_context_retain", "dev_val": dv})  # mirrors G10_forget
    for iid, dv in gt_split.items():
        dev_val_rows.append({"image_id": iid, "role": "GT_target_retain", "dev_val": dv})

    write_csv(data_dir / "train_dev_val_split.csv", dev_val_rows, ["image_id", "role", "dev_val"])

    def count(role, dv):
        return sum(1 for r in dev_val_rows if r["role"] == role and r["dev_val"] == dv)

    print("[prepare-data] TRAIN dev/val split (for Part VII component selection only):")
    for role in ["G10_forget", "GC_context_retain", "GT_target_retain"]:
        print(f"  {role}: dev={count(role,'dev')} val={count(role,'val')}")
    print(f"[prepare-data] wrote {data_dir / 'train_dev_val_split.csv'}")


if __name__ == "__main__":
    main()
