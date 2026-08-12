"""Part XI: evaluate ALL FOUR models on the same unseen CLEAN TEST split.

Original / Clean Debias / Adv Debias are REUSED verbatim: they were already
evaluated on this exact test split (same seed=42 split, same checkpoints) by
the prior adversarial_functional_debiasing_pilot, so their
evaluation/*_results.csv are copied in unmodified rather than re-run --
re-running would consume GPU time to reproduce numbers that cannot change
(same frozen base model + same frozen adapter + same images). Only Adv +
Decomp Debias (the new Model C) is actually run here.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/evaluate_all_models.py \
        --data-config configs/data.yaml [--device cuda:0] [--skip-inference]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../CooccurrenceHallucinationDiagnostic/src

from cooc_diagnostic.coco_index import load_coco_instances  # noqa: E402
from cooc_diagnostic.llava_runtime import (  # noqa: E402
    build_inputs,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_logits,
)

ROLE_QUESTION = {"G10": "target", "G00": "target", "GT": "target", "GC": "context"}
ROLE_GROUND_TRUTH = {"G10": "No", "G00": "No", "GT": "Yes", "GC": "Yes"}
REUSED_MODELS = ["original", "clean_debias", "adv_debias"]


def copy_reused_results(prior_dir: Path, eval_dir: Path) -> None:
    for m in REUSED_MODELS:
        src = prior_dir / "evaluation" / f"{m}_results.csv"
        dst = eval_dir / f"{m}_results.csv"
        shutil.copy2(src, dst)
        print(f"[eval-all] reused {dst.name} verbatim from {prior_dir} (identical test split, identical checkpoint)")


def evaluate_new_model(config: dict, device: str) -> None:
    out_dir = Path(config["output_dir"])
    torch.manual_seed(int(config["seed"]))

    print(f"[eval-all:adv_decomp] loading base model on {device}")
    model, processor = load_model(config["model_id"], device)
    ckpt_dir = out_dir / "checkpoints" / "adv_decomp_debias"
    print(f"[eval-all:adv_decomp] attaching LoRA adapter from {ckpt_dir}")
    model = PeftModel.from_pretrained(model, str(ckpt_dir))
    model.eval()

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    test_rows = list(csv.DictReader((out_dir / "data" / "test_split.csv").open("r", encoding="utf-8")))
    rows = [(int(r["image_id"]), r["role"]) for r in test_rows]
    print(f"[eval-all:adv_decomp] {len(rows)} (image_id, role) evaluation rows")

    probe_image = load_image(rows[0][0])
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)

    target_category = config["target_category"]
    context_category = config["context_category"]

    results = []
    for i, (image_id, role) in enumerate(rows):
        category = target_category if ROLE_QUESTION[role] == "target" else context_category
        image = load_image(image_id)
        input_ids, image01 = build_inputs(processor, category, image, device)

        yes_logit, no_logit = yes_no_logits(model, processor, decision_point, input_ids, image01)
        s_score = yes_logit - no_logit
        greedy_text = generate_greedy_answer(model, processor, input_ids, image01)
        prediction = "Yes" if is_yes_response(greedy_text) else "No"
        ground_truth = ROLE_GROUND_TRUTH[role]

        results.append(
            {
                "model": "adv_decomp_debias",
                "image_id": image_id,
                "role": role,
                "question_category": category,
                "ground_truth_answer": ground_truth,
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "s_score": s_score,
                "response_text": greedy_text,
                "prediction": prediction,
                "correct": prediction == ground_truth,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"[eval-all:adv_decomp] {i + 1}/{len(rows)} done")
            torch.cuda.empty_cache()

    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / "adv_decomp_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[eval-all:adv_decomp] wrote {len(results)} rows to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip-inference", action="store_true", help="Only copy reused results, skip Model C inference")
    args = parser.parse_args()
    with args.data_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    prior_dir = Path(config["prior_pilot_dir"])
    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    copy_reused_results(prior_dir, eval_dir)
    if not args.skip_inference:
        evaluate_new_model(config, args.device)


if __name__ == "__main__":
    main()
