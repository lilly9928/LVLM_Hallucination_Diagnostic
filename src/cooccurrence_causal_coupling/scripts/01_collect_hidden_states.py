"""Experiment 4, Step 1: collect decision-position hidden states at the 9
candidate layers (audit Sec.6), for train+val images only.

Reuses Stage 10's exact 3,039 (image, target) pairs and Stage 9/10/11's
LLaVA runtime. New: the image-level train/val/test split (audit Sec.5) and
saving the raw 4096-dim hidden vectors themselves (Stage 11 only ever saved
scalar logit-lens readouts).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python 01_collect_hidden_states.py \
        --config ../configs/01_collect_hidden_states.yaml --pilot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[2]  # .../CooccurrenceHallucinationDiagnostic/src (contains cooc_diagnostic and cooccurrence_causal_coupling)
sys.path.insert(0, str(SRC_DIR))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import build_inputs, detect_yes_no_decision_point, load_model, normalize
from cooccurrence_causal_coupling.common import CANDIDATE_LAYERS, image_level_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Process only the first pilot_n_rows rows")
    parser.add_argument("--pilot-n-rows", type=int, default=60)
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

    train_val = evidence[evidence["split"].isin(["train", "val"])].reset_index(drop=True)
    print(f"[collect] split sizes (images): "
          f"train={sum(v=='train' for v in split.values())} val={sum(v=='val' for v in split.values())} test={sum(v=='test' for v in split.values())}")
    print(f"[collect] train+val rows: {len(train_val)} / {len(evidence)} total")

    if args.pilot:
        train_val = train_val.iloc[: args.pilot_n_rows].reset_index(drop=True)
        print(f"[collect] PILOT mode: {len(train_val)} rows")

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    print(f"[collect] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])
    decision_point = detect_yes_no_decision_point(model, processor, load_image(int(train_val.iloc[0]["image_id"])), config["device"])
    print(f"[collect] decision point: prefix_ids={decision_point.prefix_ids} yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    import torch

    layer_vectors = {L: [] for L in CANDIDATE_LAYERS}
    t_start = time.time()
    for i, row in train_val.iterrows():
        image_id = int(row["image_id"])
        category = row["target"]
        image = load_image(image_id)
        input_ids, image01 = build_inputs(processor, category, image, config["device"])

        with torch.no_grad():
            if decision_point.prefix_ids:
                prefix = torch.tensor([decision_point.prefix_ids], device=input_ids.device, dtype=input_ids.dtype)
                full_ids = torch.cat([input_ids, prefix], dim=1)
            else:
                full_ids = input_ids
            pixel_values = normalize(processor, image01).to(model.dtype)
            outputs = model(input_ids=full_ids, pixel_values=pixel_values, output_hidden_states=True)

        for L in CANDIDATE_LAYERS:
            layer_vectors[L].append(outputs.hidden_states[L][0, -1, :].float().cpu().numpy())

        if (i + 1) % 200 == 0 or (i + 1) == len(train_val):
            elapsed = time.time() - t_start
            print(f"[collect] {i + 1}/{len(train_val)} done, {elapsed:.1f}s elapsed, {elapsed / (i + 1):.3f}s/row")

    npz_payload = {f"layer_{L}": np.stack(layer_vectors[L]) for L in CANDIDATE_LAYERS}
    out_prefix = "pilot_" if args.pilot else ""
    npz_path = output_dir / f"{out_prefix}hidden_states_train_val.npz"
    np.savez_compressed(npz_path, **npz_payload)
    meta_path = output_dir / f"{out_prefix}hidden_states_train_val_meta.csv"
    train_val[["image_id", "target", "cooc_score", "split"]].to_csv(meta_path, index=False)
    print(f"[collect] wrote {npz_path} ({sum(v.nbytes for v in npz_payload.values()) / 1e6:.1f} MB uncompressed) and {meta_path}")


if __name__ == "__main__":
    main()
