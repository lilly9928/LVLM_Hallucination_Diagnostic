"""Experiment 4, Step 3: cheap screening pass (audit Sec.8, staged compute
plan, stage 1 of 3). Real direction only, lambda=1.0 only, all 9 candidate
layers, on the VALIDATION split (never test). Goal: identify which layers
show ANY sign of a co-occurrence-selective effect before spending the full
lambda-grid x 5-control battery on them at test time.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python 03_screen_layers.py \
        --config ../configs/03_screen_layers.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import stats

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import build_inputs, detect_yes_no_decision_point, load_model, yes_no_logits
from cooccurrence_causal_coupling.common import CANDIDATE_LAYERS, ResidualProjectionHook, image_level_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(config["evidence_path"])
    image_ids = sorted(evidence["image_id"].unique())
    split = image_level_split(image_ids, seed=int(config["seed"]), n_train=int(config["n_train_images"]), n_val=int(config["n_val_images"]))
    evidence["split"] = evidence["image_id"].map(split)
    val_rows = evidence[evidence["split"] == "val"].reset_index(drop=True)
    print(f"[screen] validation rows: {len(val_rows)} ({sum(v == 'val' for v in split.values())} images)")

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    print(f"[screen] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])
    decision_point = detect_yes_no_decision_point(model, processor, load_image(int(val_rows.iloc[0]["image_id"])), config["device"])

    directions = np.load(config["directions_path"])
    lam = float(config["lambda"])

    # --- Baseline s_T for every val row (computed once, shared across all layers) ---
    print("[screen] computing baseline s_T for all val rows")
    baselines = []
    cached_inputs = []
    for _, row in val_rows.iterrows():
        image = load_image(int(row["image_id"]))
        input_ids, image01 = build_inputs(processor, row["target"], image, config["device"])
        cached_inputs.append((input_ids, image01))
        yes0, no0 = yes_no_logits(model, processor, decision_point, input_ids, image01)
        baselines.append(yes0 - no0)
    baselines = np.array(baselines)

    per_layer_results = []
    for L in CANDIDATE_LAYERS:
        direction = torch.tensor(directions[f"real_{L}"])
        reference = torch.tensor(directions[f"reference_{L}"])
        hook = ResidualProjectionHook(model, L, direction, reference)
        hook.lam = lam
        hook.armed = True

        after = []
        for input_ids, image01 in cached_inputs:
            yes1, no1 = yes_no_logits(model, processor, decision_point, input_ids, image01)
            after.append(yes1 - no1)
        hook.armed = False
        hook.remove()
        after = np.array(after)
        delta = after - baselines  # expect negative for a causally-supporting direction

        cooc = val_rows["cooc_score"].to_numpy(dtype=float)
        corr_r, corr_p = stats.pearsonr(cooc, delta)
        wilcoxon = stats.wilcoxon(baselines, after)

        per_layer_results.append(
            {
                "layer": L,
                "mean_delta_sT": float(delta.mean()),
                "median_delta_sT": float(np.median(delta)),
                "sd_delta_sT": float(delta.std(ddof=1)),
                "wilcoxon_p": float(wilcoxon.pvalue),
                "corr_delta_vs_cooc_score": float(corr_r),
                "corr_p": float(corr_p),
                "n": len(delta),
            }
        )
        print(f"[screen] layer={L:2d} mean_delta_sT={delta.mean():+.4f} wilcoxon_p={wilcoxon.pvalue:.3g} "
              f"corr(delta, cooc_score)={corr_r:+.3f} (p={corr_p:.3g})")

    report = {"lambda": lam, "split": "validation", "n_val_rows": len(val_rows), "per_layer": per_layer_results}
    out_path = output_dir / "screen_layers_val.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[screen] wrote {out_path}")


if __name__ == "__main__":
    main()
