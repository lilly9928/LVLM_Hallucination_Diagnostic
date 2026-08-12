"""Stage 11 (Experiment 3): layer-wise localization via logit lens.

For every (image, absent target) pair already collected in Stage 10, re-run a
single clean forward pass with output_hidden_states=True and apply the
model's own final RMSNorm + lm_head to EVERY LLM decoder layer's hidden state
at the decision position (logit lens) -- unsupervised, no probe fitting.
layer index N (N = number of LLM decoder layers) exactly reproduces Stage 10's
real s_T, since that is the actual computation path; this is checked, not
assumed.

Vision-tower and projector layers are intentionally NOT probed here: they
never see the text/target, so a fixed image produces an identical
vision/projector hidden state regardless of which target is asked about --
within-image beta_l there is structurally 0 by construction, not an empirical
finding. Stage 5's between-group excess-AUC result is the correct test for
those layers and is not repeated.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_layer_localization.py \
        --config configs/stage11_layer_localization.yaml --pilot
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import (
    build_inputs,
    detect_yes_no_decision_point,
    layerwise_logit_lens,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11: layer-wise logit-lens localization")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Run only pilot_n_rows rows for timing + logit-lens validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = pd.read_csv(config["evidence_path"])
    if args.pilot:
        evidence = evidence.iloc[: int(config["pilot_n_rows"])].reset_index(drop=True)
    print(f"[stage11] processing {len(evidence)} rows from {config['evidence_path']} ({'PILOT' if args.pilot else 'FULL'} mode)")

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    print(f"[stage11] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])

    probe_image = load_image(int(evidence.iloc[0]["image_id"]))
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, config["device"])
    n_llm_layers = len(model.model.language_model.layers)
    print(f"[stage11] decision point: prefix_ids={decision_point.prefix_ids} yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")
    print(f"[stage11] n_llm_layers={n_llm_layers} (hidden_states indices 0..{n_llm_layers}; index {n_llm_layers} == true final layer)")

    rows = []
    t_start = time.time()
    for i, row in evidence.iterrows():
        image_id = int(row["image_id"])
        category = row["target"]
        image = load_image(image_id)
        input_ids, image01 = build_inputs(processor, category, image, config["device"])

        layer_logits = layerwise_logit_lens(model, processor, decision_point, input_ids, image01)
        for layer_idx, (yes_logit, no_logit) in layer_logits.items():
            rows.append(
                {
                    "image_id": image_id,
                    "target": category,
                    "cooc_score": row["cooc_score"],
                    "layer": layer_idx,
                    "yes_logit_l": yes_logit,
                    "no_logit_l": no_logit,
                    "s_T_l": yes_logit - no_logit,
                }
            )
        if (i + 1) % 200 == 0 or (i + 1) == len(evidence):
            elapsed = time.time() - t_start
            print(f"[stage11] {i + 1}/{len(evidence)} rows done, {elapsed:.1f}s elapsed, {elapsed / (i + 1):.3f}s/row")

    elapsed = time.time() - t_start
    out_name = "pilot_layerwise_evidence.csv" if args.pilot else "layerwise_evidence.csv"
    out_path = output_dir / out_name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[stage11] wrote {len(rows)} (row x layer) records to {out_path} ({elapsed / len(evidence):.3f}s/original-row avg)")

    # --- Validation: final-layer logit lens must reproduce Stage 10's real s_T ---
    final_layer_df = pd.DataFrame([r for r in rows if r["layer"] == n_llm_layers])
    merged = final_layer_df.merge(evidence[["image_id", "target", "s_T"]], on=["image_id", "target"], suffixes=("_lens", "_real"))
    diff = (merged["s_T_l"] - merged["s_T"]).abs()
    print(f"[stage11] VALIDATION: final-layer logit-lens vs Stage 10 real s_T -- "
          f"max_abs_diff={diff.max():.6f} mean_abs_diff={diff.mean():.6f} (should be ~0)")


if __name__ == "__main__":
    main()
