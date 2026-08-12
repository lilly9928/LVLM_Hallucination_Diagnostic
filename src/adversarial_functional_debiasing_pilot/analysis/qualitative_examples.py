"""Optional: 5-10 representative TEST G10 examples with per-model s_ball, chosen
by delta (Adv Debias improvement over Original) covering BOTH a successful and
an unsuccessful tail -- not cherry-picked to only show positive cases. Reads
already-saved evaluation CSVs only; no model inference.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python analysis/qualitative_examples.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cooc_diagnostic.coco_index import load_coco_instances  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    eval_dir = out_dir / "evaluation"

    per_model = {}
    for m in ["original", "clean_debias", "adv_debias"]:
        with (eval_dir / f"{m}_results.csv").open("r", encoding="utf-8") as f:
            rows = {int(r["image_id"]): r for r in csv.DictReader(f) if r["role"] == "G10"}
        per_model[m] = rows

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    image_ids = sorted(per_model["original"].keys())
    records = []
    for image_id in image_ids:
        orig_s = float(per_model["original"][image_id]["s_score"])
        clean_s = float(per_model["clean_debias"][image_id]["s_score"])
        adv_s = float(per_model["adv_debias"][image_id]["s_score"])
        records.append(
            {
                "image_id": image_id,
                "image_path": str(image_dir / val_index.image_filenames[image_id]),
                "original_s_ball": orig_s,
                "clean_debias_s_ball": clean_s,
                "adv_debias_s_ball": adv_s,
                "adv_minus_original": adv_s - orig_s,
            }
        )

    records.sort(key=lambda r: r["adv_minus_original"])
    n_each = max(1, args.n // 2)
    selected = records[:n_each] + records[-n_each:]  # biggest reduction + biggest (non-)improvement/regression tails
    selected = sorted(selected, key=lambda r: r["image_id"])

    out_path = eval_dir / "qualitative_examples.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "Both a successful tail (largest s_ball reduction) and an unsuccessful/regression tail "
                "(smallest reduction or increase) are included -- not cherry-picked to only positive cases.",
                "examples": selected,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path} ({len(selected)} examples)")


if __name__ == "__main__":
    main()
