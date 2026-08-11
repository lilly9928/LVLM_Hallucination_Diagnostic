"""Stage 6 (add-on): transfer test -- does the SAME yes/no-optimized epsilon*
adversarial image also cause the target category to be mentioned in an
open-ended caption?

For each Stage 3 sample:
  - already_yes (epsilon*=0): use the clean image directly (no attack needed,
    mirrors the closed-question definition).
  - flipped: re-run PGD at the ALREADY-KNOWN epsilon* (single call, not a
    search) with the same recipe Stage 3 used, and re-verify the closed-question
    flip before trusting the resulting image.
  - censored: skipped -- there is no valid "attacked image" for a sample that
    never flipped the closed question at all.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage6_open_ended_transfer.py \
        --config configs/stage6_open_ended_transfer.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import (
    OPEN_ENDED_CAPTION_PROMPT,
    build_inputs,
    build_inputs_from_text,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_margin,
)
from cooc_diagnostic.mention_detection import text_mentions_category
from cooc_diagnostic.pgd_attack import pgd_attack_with_restarts
from cooc_diagnostic.survival_analysis import paired_mcnemar_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6: open-ended caption transfer test")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = config["device"]

    with Path(config["epsilon_star_results_path"]).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["status"] != "censored"]
    print(f"[stage6] loaded {len(rows)} non-censored samples")

    print(f"[stage6] loading model: {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[int(image_id)]).convert("RGB")

    probe_image = load_image(rows[0]["image_id"])
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[stage6] decision point: prefix_ids={decision_point.prefix_ids} yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    # Number of expanded image placeholder tokens is fixed by the vision config
    # (constant 336x336 grid), not by pixel content -- build once, reuse for
    # every sample's open-ended generation regardless of which image/attack it is.
    input_ids_open, _ = build_inputs_from_text(processor, OPEN_ENDED_CAPTION_PROMPT, probe_image, device)

    results = []
    n_reattack_failed = 0
    for i, row in enumerate(rows):
        image_id = int(row["image_id"])
        category = row["category"]
        status = row["status"]
        image = load_image(image_id)
        input_ids_closed, image01_clean = build_inputs(processor, category, image, device)

        if status == "already_yes":
            attacked_image01 = image01_clean
            reattack_ok = True
        else:
            epsilon = float(row["epsilon_star"])

            def margin_fn(img):
                return yes_no_margin(model, processor, decision_point, input_ids_closed, img)

            pgd_result = pgd_attack_with_restarts(
                image01_clean, epsilon, int(config["pgd_steps"]), margin_fn, int(config["n_restarts"])
            )
            reattack_text = generate_greedy_answer(model, processor, input_ids_closed, pgd_result.best_image)
            reattack_ok = is_yes_response(reattack_text)
            attacked_image01 = pgd_result.best_image
            if not reattack_ok:
                n_reattack_failed += 1

        caption_attacked = generate_greedy_answer(model, processor, input_ids_open, attacked_image01, max_new_tokens=int(config["max_new_tokens_caption"]))
        caption_clean = generate_greedy_answer(model, processor, input_ids_open, image01_clean, max_new_tokens=int(config["max_new_tokens_caption"]))

        results.append(
            {
                "pair_id": row["pair_id"],
                "arm": row["arm"],
                "image_id": image_id,
                "category": category,
                "status": status,
                "epsilon_star": row["epsilon_star"],
                "reattack_ok": reattack_ok,
                "mentioned_in_attacked_caption": text_mentions_category(caption_attacked, category),
                "mentioned_in_clean_caption": text_mentions_category(caption_clean, category),
                "attacked_caption": caption_attacked,
                "clean_caption": caption_clean,
            }
        )
        if (i + 1) % 25 == 0:
            print(f"[stage6] {i + 1}/{len(rows)} done")

    print(f"[stage6] re-attack failed to reproduce the closed-question flip on {n_reattack_failed}/{len(rows)} samples (excluded below)")

    usable = [r for r in results if r["reattack_ok"]]
    df = pd.DataFrame(usable)

    with (output_dir / "transfer_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary = {}
    for arm in ["treatment", "control"]:
        sub = df[df["arm"] == arm]
        summary[arm] = {
            "n": int(len(sub)),
            "transfer_rate": float(sub["mentioned_in_attacked_caption"].mean()),
            "clean_baseline_mention_rate": float(sub["mentioned_in_clean_caption"].mean()),
        }
    print("[stage6] summary:")
    for arm, d in summary.items():
        print(f"  {arm}: n={d['n']} transfer_rate={d['transfer_rate']:.3f} clean_baseline_mention_rate={d['clean_baseline_mention_rate']:.3f}")

    transfer_test = paired_mcnemar_test(df, "mentioned_in_attacked_caption")
    baseline_test = paired_mcnemar_test(df, "mentioned_in_clean_caption")
    print(f"[stage6] paired McNemar (transfer): {transfer_test}")
    print(f"[stage6] paired McNemar (clean baseline mention): {baseline_test}")

    report = {
        "n_total": len(results),
        "n_reattack_failed_excluded": n_reattack_failed,
        "summary": summary,
        "mcnemar_transfer": transfer_test,
        "mcnemar_clean_baseline": baseline_test,
    }
    with (output_dir / "stage6_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[stage6] report written to {output_dir / 'stage6_report.json'}")


if __name__ == "__main__":
    main()
