"""Stage 10 (Experiment 2): within-image co-occurrence specificity -- clean s_T
for many absent targets evaluated against the SAME fixed image.

Unlike Stage 9 (one high/low matched target per image), this fixes a sample of
val2017 images and evaluates EVERY eligible absent-target category against
each one, using exactly Stage 1/2's co-occurrence definitions
(compute_cooccurrence_stats, build_candidates) and Stage 9's LLaVA runtime
(llava_runtime.py). No attack is run -- clean image only.

Run --pilot first (small n_images, timing + sanity) before the full run:

    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage10_within_image_evidence.py \
        --config configs/stage10_within_image_evidence.yaml --pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.cooccurrence_stats import compute_cooccurrence_stats
from cooc_diagnostic.llava_runtime import (
    build_inputs,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_logits,
)
from cooc_diagnostic.strata_sampling import build_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10: within-image co-occurrence specificity")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Run only pilot_n_images images for timing + sanity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    min_support = int(config["min_support_count"])

    print(f"[stage10] loading train2017 annotations: {config['train_annotation_path']}")
    train_index = load_coco_instances(config["train_annotation_path"])
    train_stats = compute_cooccurrence_stats(train_index.category_ids, train_index.image_categories)
    category_index = {cid: i for i, cid in enumerate(train_stats.category_ids)}
    names_by_id = train_index.category_names

    freq_by_id = {
        cid: float(train_stats.marginal_counts[category_index[cid]]) / train_stats.n_images
        for cid in train_stats.category_ids
    }
    rare_threshold = float(config["rare_category_freq_threshold_pct"]) / 100.0
    eligible_category_ids = sorted(cid for cid, freq in freq_by_id.items() if freq >= rare_threshold)
    print(f"[stage10] eligible target categories: {len(eligible_category_ids)}/80 (same threshold as Stage 2: {config['rare_category_freq_threshold_pct']}%)")

    print(f"[stage10] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])
    val_nonempty = {iid: cats for iid, cats in val_index.image_categories.items() if cats}

    n_images = int(config["pilot_n_images"]) if args.pilot else int(config["n_images"])
    rng = np.random.default_rng(int(config["seed"]))
    all_ids = sorted(val_nonempty.keys())
    sampled_ids = sorted(int(i) for i in rng.choice(all_ids, size=n_images, replace=False))
    sampled_categories = {iid: val_nonempty[iid] for iid in sampled_ids}
    print(f"[stage10] sampled {len(sampled_ids)} images ({'PILOT' if args.pilot else 'FULL'} mode, seed={config['seed']})")

    candidates = build_candidates(
        sampled_categories, eligible_category_ids, category_index, train_stats.pmi, train_stats.joint_counts, min_support
    )
    print(f"[stage10] built {len(candidates)} (image, absent-target) candidates")

    # --- Sanity check: every candidate target is genuinely absent from its image ---
    for c in candidates:
        assert c.category_id not in sampled_categories[c.image_id], (
            f"target {c.category_id} present in image {c.image_id} -- build_candidates invariant violated"
        )
    print("[stage10] SANITY CHECK: all candidate targets confirmed absent from their image -- passed")

    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    def present_objects(image_id: int) -> str:
        return "|".join(sorted(names_by_id[c] for c in sampled_categories[image_id]))

    print(f"[stage10] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])

    probe_image = load_image(candidates[0].image_id)
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, config["device"])
    print(f"[stage10] decision point: prefix_ids={decision_point.prefix_ids} yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    results = []
    t_start = time.time()
    for i, c in enumerate(candidates):
        category_name = names_by_id[c.category_id]
        image = load_image(c.image_id)
        input_ids, image01 = build_inputs(processor, category_name, image, config["device"])

        yes_logit, no_logit = yes_no_logits(model, processor, decision_point, input_ids, image01)
        response_text = generate_greedy_answer(model, processor, input_ids, image01)

        results.append(
            {
                "image_id": c.image_id,
                "target_category_id": c.category_id,
                "target": category_name,
                "cooc_score": c.score,
                "n_present": c.n_present,
                "n_pmi_terms_used": c.n_pmi_terms_used,
                "target_marginal_freq": freq_by_id[c.category_id],
                "present_objects": present_objects(c.image_id),
                "clean_yes_logit": yes_logit,
                "clean_no_logit": no_logit,
                "s_T": yes_logit - no_logit,
                "clean_response_text": response_text,
                "clean_is_yes": is_yes_response(response_text),
            }
        )
        if (i + 1) % 200 == 0 or (i + 1) == len(candidates):
            elapsed = time.time() - t_start
            print(f"[stage10] {i + 1}/{len(candidates)} done, {elapsed:.1f}s elapsed, {elapsed / (i + 1):.3f}s/sample")

    elapsed = time.time() - t_start
    per_sample = elapsed / len(candidates)
    out_name = "pilot_within_image_evidence.csv" if args.pilot else "within_image_evidence.csv"
    out_path = output_dir / out_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[stage10] wrote {len(results)} rows to {out_path} ({per_sample:.3f}s/sample avg)")

    if args.pilot:
        projected_n = None
        # Rough projection of the full run's candidate count and time, assuming
        # a similar candidates-per-image rate as observed in the pilot.
        rate = len(candidates) / len(sampled_ids)
        projected_n = rate * int(config["n_images"])
        projected_time_sec = projected_n * per_sample
        print(f"[stage10] PILOT projection for full run (n_images={config['n_images']}): "
              f"~{projected_n:.0f} candidates, ~{projected_time_sec / 60:.1f} min at this rate")
        with (output_dir / "pilot_timing_report.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "pilot_n_images": len(sampled_ids),
                    "pilot_n_candidates": len(candidates),
                    "sec_per_sample": per_sample,
                    "projected_full_n_candidates": projected_n,
                    "projected_full_time_min": projected_time_sec / 60,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
