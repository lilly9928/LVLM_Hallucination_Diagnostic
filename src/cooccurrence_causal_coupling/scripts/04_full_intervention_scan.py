"""Experiment 4, Step 4: full lambda grid (real direction) + random/shuffled
controls (single lambda) + genuine-target and general-stability controls, on
the TEST split, for the 4 layers approved after screening (L3, L13, L16, L24).

Never touches the train/val split; lambda is never tuned against these
results (fixed grid, pre-registered in the audit doc before any test-split
result was seen).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python 04_full_intervention_scan.py \
        --config ../configs/04_full_intervention_scan.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import (
    build_inputs,
    build_inputs_from_text,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    load_model,
    normalize,
    yes_no_logits,
)
from cooccurrence_causal_coupling.common import ResidualProjectionHook, image_level_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def entropy_at_decision_position(model, processor, decision_point, input_ids, image01) -> float:
    with torch.no_grad():
        if decision_point.prefix_ids:
            prefix = torch.tensor([decision_point.prefix_ids], device=input_ids.device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, prefix], dim=1)
        else:
            full_ids = input_ids
        pixel_values = normalize(processor, image01).to(model.dtype)
        outputs = model(input_ids=full_ids, pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits[0, -1, :].float(), dim=-1)
        return float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    patching_dir = Path(config["output_dir"])
    controls_dir = Path(config["controls_dir"])
    patching_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(config["evidence_path"])
    image_ids = sorted(evidence["image_id"].unique())
    split = image_level_split(image_ids, seed=int(config["seed"]), n_train=int(config["n_train_images"]), n_val=int(config["n_val_images"]))
    evidence["split"] = evidence["image_id"].map(split)
    test_rows = evidence[evidence["split"] == "test"].reset_index(drop=True)
    test_image_ids = sorted(test_rows["image_id"].unique())
    print(f"[scan] test rows: {len(test_rows)} ({len(test_image_ids)} images)")

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])
    names_by_id = val_index.category_names

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    print(f"[scan] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])
    decision_point = detect_yes_no_decision_point(model, processor, load_image(test_image_ids[0]), config["device"])

    # --- Cache (input_ids, image01) for every test row (main test + control 2's low/high split reuse this) ---
    print("[scan] caching inputs + baseline s_T for all test rows")
    cached_main, baseline_main = [], []
    for _, row in test_rows.iterrows():
        input_ids, image01 = build_inputs(processor, row["target"], load_image(int(row["image_id"])), config["device"])
        cached_main.append((input_ids, image01))
        yes0, no0 = yes_no_logits(model, processor, decision_point, input_ids, image01)
        baseline_main.append(yes0 - no0)
    baseline_main = np.array(baseline_main)

    # --- Control 1 setup: one genuine-present category per test image ---
    genuine_targets = []
    for image_id in test_image_ids:
        present = val_index.image_categories.get(image_id, set())
        if present:
            genuine_targets.append((image_id, names_by_id[next(iter(present))]))
    cached_genuine, baseline_genuine = [], []
    for image_id, category in genuine_targets:
        input_ids, image01 = build_inputs(processor, category, load_image(image_id), config["device"])
        cached_genuine.append((input_ids, image01))
        yes0, no0 = yes_no_logits(model, processor, decision_point, input_ids, image01)
        baseline_genuine.append(yes0 - no0)
    baseline_genuine = np.array(baseline_genuine)
    print(f"[scan] control1 (genuine target) rows: {len(genuine_targets)}")

    directions = np.load(config["directions_path"])
    layers = config["layers"]
    lambdas_real = config["lambdas_real"]
    lambda_control = float(config["lambda_control"])
    n_random = int(config["n_random_directions"])

    main_rows, genuine_rows, random_rows, shuffled_rows, stability_rows = [], [], [], [], []

    for L in layers:
        reference = torch.tensor(directions[f"reference_{L}"])

        # --- Main test: real direction, full lambda grid ---
        real_dir = torch.tensor(directions[f"real_{L}"])
        hook = ResidualProjectionHook(model, L, real_dir, reference)
        for lam in lambdas_real:
            hook.lam = lam
            hook.armed = True
            for idx, (input_ids, image01) in enumerate(cached_main):
                yes1, no1 = yes_no_logits(model, processor, decision_point, input_ids, image01)
                row = test_rows.iloc[idx]
                main_rows.append(
                    {"layer": L, "lambda": lam, "direction_type": "real", "image_id": int(row["image_id"]),
                     "target": row["target"], "cooc_score": float(row["cooc_score"]),
                     "s_before": float(baseline_main[idx]), "s_after": yes1 - no1}
                )
            # control 1 at every lambda, real direction only
            for idx, (input_ids, image01) in enumerate(cached_genuine):
                yes1, no1 = yes_no_logits(model, processor, decision_point, input_ids, image01)
                genuine_rows.append(
                    {"layer": L, "lambda": lam, "image_id": genuine_targets[idx][0], "category": genuine_targets[idx][1],
                     "s_before": float(baseline_genuine[idx]), "s_after": yes1 - no1}
                )
        hook.armed = False
        print(f"[scan] layer={L} real-direction full lambda grid done")

        # --- Control 5: general stability (real direction, lambda=1.0 only) ---
        hook.lam = 1.0
        for image_id in test_image_ids[: int(config["n_stability_images"])]:
            image = load_image(image_id)
            input_ids, image01 = build_inputs_from_text(processor, "Describe this image in detail.", image, config["device"])
            hook.armed = False
            cap0 = generate_greedy_answer(model, processor, input_ids, image01, max_new_tokens=40)
            ent0 = entropy_at_decision_position(model, processor, decision_point, input_ids, image01)
            hook.armed = True
            cap1 = generate_greedy_answer(model, processor, input_ids, image01, max_new_tokens=40)
            ent1 = entropy_at_decision_position(model, processor, decision_point, input_ids, image01)
            hook.armed = False
            stability_rows.append({"layer": L, "image_id": image_id, "caption_before": cap0, "caption_after": cap1,
                                    "entropy_before": ent0, "entropy_after": ent1})
        hook.remove()
        print(f"[scan] layer={L} control5 (stability) done")

        # --- Control 3: random directions, lambda_control only ---
        for seed_i in range(n_random):
            rand_dir = torch.tensor(directions[f"random{seed_i}_{L}"])
            hook_r = ResidualProjectionHook(model, L, rand_dir, reference)
            hook_r.lam = lambda_control
            hook_r.armed = True
            for idx, (input_ids, image01) in enumerate(cached_main):
                yes1, no1 = yes_no_logits(model, processor, decision_point, input_ids, image01)
                row = test_rows.iloc[idx]
                random_rows.append(
                    {"layer": L, "lambda": lambda_control, "seed": seed_i, "image_id": int(row["image_id"]),
                     "target": row["target"], "cooc_score": float(row["cooc_score"]),
                     "s_before": float(baseline_main[idx]), "s_after": yes1 - no1}
                )
            hook_r.armed = False
            hook_r.remove()
        print(f"[scan] layer={L} control3 (random x{n_random}) done")

        # --- Control 4: shuffled-score direction, lambda_control only ---
        shuf_dir = torch.tensor(directions[f"shuffled_{L}"])
        hook_s = ResidualProjectionHook(model, L, shuf_dir, reference)
        hook_s.lam = lambda_control
        hook_s.armed = True
        for idx, (input_ids, image01) in enumerate(cached_main):
            yes1, no1 = yes_no_logits(model, processor, decision_point, input_ids, image01)
            row = test_rows.iloc[idx]
            shuffled_rows.append(
                {"layer": L, "lambda": lambda_control, "image_id": int(row["image_id"]),
                 "target": row["target"], "cooc_score": float(row["cooc_score"]),
                 "s_before": float(baseline_main[idx]), "s_after": yes1 - no1}
            )
        hook_s.armed = False
        hook_s.remove()
        print(f"[scan] layer={L} control4 (shuffled) done")

    def write_csv(rows: list[dict], path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[scan] wrote {len(rows)} rows to {path}")

    write_csv(main_rows, patching_dir / "layerwise_intervention_results.csv")
    write_csv(genuine_rows, controls_dir / "genuine_target_results.csv")
    write_csv(random_rows, controls_dir / "random_direction_results.csv")
    write_csv(shuffled_rows, controls_dir / "shuffled_direction_results.csv")
    write_csv(stability_rows, controls_dir / "general_stability_results.csv")


if __name__ == "__main__":
    main()
