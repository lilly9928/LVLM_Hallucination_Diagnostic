"""Step 4: evaluate ONE model (original / clean_debias / adv_debias) on the
UNSEEN CLEAN TEST SPLIT ONLY. Never touches adversarial images at eval time.

For every (image_id, role) row in data/test_split.csv:
  role "G10" -> question=target_category, ground truth answer="No"  (Metric A/B numerator)
  role "G00" -> question=target_category, ground truth answer="No"  (Metric B denominator)
  role "GT"  -> question=target_category, ground truth answer="Yes" (Metric C, ball retention)
  role "GC"  -> question=context_category, ground truth answer="Yes" (Metric D, bat retention)

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/evaluate_model.py \
        --config configs/pilot.yaml --model original --device cuda:0
    /opt/anaconda3/envs/py3_11/bin/python scripts/evaluate_model.py \
        --config configs/pilot.yaml --model clean_debias --device cuda:2
    /opt/anaconda3/envs/py3_11/bin/python scripts/evaluate_model.py \
        --config configs/pilot.yaml --model adv_debias --device cuda:3
"""

from __future__ import annotations

import argparse
import csv
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["original", "clean_debias", "adv_debias"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device
    out_dir = Path(config["output_dir"])
    torch.manual_seed(int(config["seed"]))

    print(f"[eval:{args.model}] loading base model on {device}")
    model, processor = load_model(config["model_id"], device)

    if args.model != "original":
        ckpt_dir = out_dir / "checkpoints" / args.model
        print(f"[eval:{args.model}] attaching LoRA adapter from {ckpt_dir}")
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
        model.eval()

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    test_rows = list(csv.DictReader((out_dir / "data" / "test_split.csv").open("r", encoding="utf-8")))
    # de-dup: a (image_id, role) pair appears exactly once already by construction
    rows = [(int(r["image_id"]), r["role"]) for r in test_rows]
    print(f"[eval:{args.model}] {len(rows)} (image_id, role) evaluation rows")

    probe_image = load_image(rows[0][0])
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[eval:{args.model}] decision point: yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

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
                "model": args.model,
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
            print(f"[eval:{args.model}] {i + 1}/{len(rows)} done")
            torch.cuda.empty_cache()

    eval_dir = out_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = eval_dir / f"{args.model}_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[eval:{args.model}] wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
