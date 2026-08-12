"""Stage 9 (Experiment 1): clean-image target-positive evidence s_T.

Research question: for a target object T that is genuinely absent from the
image, does a higher train-set co-occurrence between T and the objects Y
actually present in the image already raise the model's target-positive
evidence s_T = logit(Yes) - logit(No) to "Is there a {T} in the image?" --
on the CLEAN image, before any attack?

Reuses the exact same 150 matched pairs / 300 samples as Stage 3
(matched_pairs_subsample_150.csv) so this joins 1:1 with Stage 3's
epsilon_star_results.csv by (pair_id, arm). No attack is run here -- this is
a pure clean-image readout using Stage 3's own runtime (llava_runtime.py).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage9_clean_target_evidence.py \
        --config configs/stage9_clean_target_evidence.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import (
    build_inputs,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9: clean-image target-positive evidence s_T")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_matched_pairs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage9] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])

    print(f"[stage9] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])
    names_by_id = val_index.category_names

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    def present_objects(image_id: int) -> str:
        cat_ids = val_index.image_categories.get(image_id, set())
        return "|".join(sorted(names_by_id[c] for c in cat_ids))

    matched_pairs = load_matched_pairs(Path(config["matched_pairs_path"]))
    print(f"[stage9] loaded {len(matched_pairs)} matched pairs ({2 * len(matched_pairs)} samples: treatment + control)")

    probe_image_id = int(matched_pairs[0]["image_id_treatment"])
    probe_image = load_image(probe_image_id)
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, config["device"])
    print(f"[stage9] decision point: prefix_ids={decision_point.prefix_ids} yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    results = []
    for arm, id_key, cat_key, score_key in [
        ("treatment", "image_id_treatment", "category_treatment", "score_treatment"),
        ("control", "image_id_control", "category_control", "score_control"),
    ]:
        for row in matched_pairs:
            image_id = int(row[id_key])
            category = row[cat_key]
            image = load_image(image_id)
            input_ids, image01 = build_inputs(processor, category, image, config["device"])

            yes_logit, no_logit = yes_no_logits(model, processor, decision_point, input_ids, image01)
            response_text = generate_greedy_answer(model, processor, input_ids, image01)

            results.append(
                {
                    "pair_id": row["pair_id"],
                    "arm": arm,
                    "image_id": image_id,
                    "category": category,
                    "cooc_score": float(row[score_key]),
                    "freq_bin": row["freq_bin"],
                    "area_bin": row["area_bin"],
                    "clip_bin": row["clip_bin"],
                    "present_objects": present_objects(image_id),
                    "n_present": len(val_index.image_categories.get(image_id, set())),
                    "clean_yes_logit": yes_logit,
                    "clean_no_logit": no_logit,
                    "s_T": yes_logit - no_logit,
                    "clean_response_text": response_text,
                    "clean_is_yes": is_yes_response(response_text),
                }
            )
        print(f"[stage9] finished arm={arm} ({len(matched_pairs)} samples)")

    out_path = output_dir / "clean_evidence.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[stage9] wrote {len(results)} clean-evidence rows to {out_path}")


if __name__ == "__main__":
    main()
