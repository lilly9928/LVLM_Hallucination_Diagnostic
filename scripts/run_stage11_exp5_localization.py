"""Stage 11 Exp5: functional localization of Bat-dependent Ball evidence.

Readout choice (fixed here, before running, per the no-metric-shopping rule):
  - LLM decoder layers (embedding + all 32 blocks): LOGIT LENS -- apply the
    model's own final RMSNorm (language_model.norm) + lm_head to each layer's
    hidden state at the decision (teacher-forced) position, restricted to the
    yes/no token ids, giving e_ball^(l) = logit(yes) - logit(no) in the SAME
    units as the final s_ball. This is a real target-direction projection
    (the yes/no rows of lm_head are the model's own learned unembedding
    directions), training-free, standard in the mechanistic-interpretability
    literature (logit lens, nostalgebraist 2020).
  - Vision-tower / projector stages have no token-unembedding analog, so a
    logistic probe (Stage 5's exact fit_probe: LogisticRegressionCV, Cs=10,
    cv=5, scoring="roc_auc") is trained ONCE on mean-pooled hook features from
    an independent train2017 sample (ball present vs absent, same question
    prompt) and applied to the case-study images' pooled features via
    decision_function (an unbounded, logit-like score comparable in spirit,
    not scale, to the LLM logit-lens values).

Runs on Exp4's saved original/bat_removed/sham images (same 65 G10 images, same
masks) -- no new counterfactual images are generated here.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp5_localization.py \
        --config configs/stage11_case_bat_ball.yaml [--device cuda:2]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.linear_probe import fit_probe
from cooc_diagnostic.llava_runtime import detect_yes_no_decision_point, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp5: layerwise localization")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_probe_train", type=int, default=250, help="positive + negative each, from train2017")
    return parser.parse_args()


class Extractor:
    """Wraps one forward pass, capturing vision/projector pooled features and
    all LLM decoder-layer hidden states at the decision (last prompt) position."""

    def __init__(self, model, processor, decision_point, device):
        self.model = model
        self.processor = processor
        self.decision_point = decision_point
        self.device = device
        self.num_layers = model.config.text_config.num_hidden_layers
        self.final_norm = model.language_model.norm
        self.lm_head = model.lm_head

    @torch.no_grad()
    def extract(self, image: Image.Image, question_text: str) -> dict:
        # Build inputs identically to llava_runtime.build_inputs_from_text, but we
        # need the raw `encoded` object here too (not exposed by that helper).
        full_prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]}],
            add_generation_prompt=True,
        )
        encoded = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        if self.decision_point.prefix_ids:
            prefix = torch.tensor([self.decision_point.prefix_ids], device=self.device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, prefix], dim=1)
            attn = torch.cat([encoded["attention_mask"], torch.ones_like(prefix)], dim=1)
        else:
            full_ids, attn = input_ids, encoded["attention_mask"]

        captured = {}
        handles = [
            self.model.model.vision_tower.register_forward_hook(
                lambda m, i, o: captured.__setitem__("vision", o.hidden_states[self.model.config.vision_feature_layer][0, 1:].float().cpu())
            ),
            self.model.model.multi_modal_projector.register_forward_hook(
                lambda m, i, o: captured.__setitem__("projector", o[0].float().cpu())
            ),
        ]
        out = self.model(
            input_ids=full_ids, attention_mask=attn, pixel_values=encoded["pixel_values"].to(self.model.dtype),
            output_hidden_states=True, return_dict=True,
        )
        for h in handles:
            h.remove()

        result = {
            "vision_feature": captured["vision"].mean(0).numpy(),
            "projector_feature": captured["projector"].mean(0).numpy(),
        }
        yes_id, no_id = self.decision_point.yes_token_id, self.decision_point.no_token_id
        for layer_idx, h in enumerate(out.hidden_states):  # 0..num_layers
            h_last = h[0, -1, :].to(self.final_norm.weight.dtype)
            normed = self.final_norm(h_last)
            logits = self.lm_head(normed)
            result[f"logit_lens_layer_{layer_idx}"] = float(logits[yes_id] - logits[no_id])
        return result


def sample_train2017_ball_images(config, n_pos: int, n_neg: int, rng: random.Random) -> tuple[list[int], list[int], "object"]:
    train_index = load_coco_instances(config["train_annotation_path"])
    name_to_id = {c.name: c.id for c in train_index.categories}
    target_id = name_to_id[config["target_category"]]
    pos, neg = [], []
    for image_id, cats in train_index.image_categories.items():
        (pos if target_id in cats else neg).append(image_id)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos], neg[:n_neg], train_index


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = args.device or config["device"]
    output_dir = Path(config["output_dir"])
    question_text = f"Is there a {config['target_category']} in the image?"
    rng = random.Random(int(config["seed"]))

    print(f"[exp5] loading model {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)

    val_index = load_coco_instances(config["val_annotation_path"])
    val_name_to_id = {c.name: c.id for c in val_index.categories}
    context_id = val_name_to_id[config["context_category"]]
    target_id = val_name_to_id[config["target_category"]]
    g10_ids = sorted(
        iid for iid, cats in val_index.image_categories.items() if context_id in cats and target_id not in cats
    )
    probe_image = Image.open(Path(config["val_image_dir"]) / val_index.image_filenames[g10_ids[0]]).convert("RGB")
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    extractor = Extractor(model, processor, decision_point, device)
    print(f"[exp5] LLM layers={extractor.num_layers} (+embedding layer 0) -- logit-lens at all {extractor.num_layers + 1}")

    # --- Train vision/projector ball-presence probes on an independent train2017 sample ---
    pos_ids, neg_ids, train_index = sample_train2017_ball_images(config, args.n_probe_train, args.n_probe_train, rng)
    train_img_dir = Path(config["train_image_dir"])
    print(f"[exp5] extracting probe-training features: {len(pos_ids)} positive + {len(neg_ids)} negative (train2017)")

    train_vision, train_proj, train_labels = [], [], []
    for label, ids in [(1, pos_ids), (0, neg_ids)]:
        for i, image_id in enumerate(ids):
            img_path = train_img_dir / train_index.image_filenames[image_id]
            if not img_path.exists():
                continue
            image = Image.open(img_path).convert("RGB")
            feats = extractor.extract(image, question_text)
            train_vision.append(feats["vision_feature"])
            train_proj.append(feats["projector_feature"])
            train_labels.append(label)
            if (i + 1) % 100 == 0:
                print(f"[exp5]   probe-train label={label}: {i + 1}/{len(ids)}")
                torch.cuda.empty_cache()

    train_labels = np.array(train_labels)
    vision_probe = fit_probe(np.stack(train_vision), train_labels)
    proj_probe = fit_probe(np.stack(train_proj), train_labels)
    print(f"[exp5] fit vision/projector probes on {len(train_labels)} train2017 images")

    # --- Extract on Exp4's original/bat_removed/sham images ---
    image_dir = output_dir / "counterfactual_images"
    conditions = {"original": "original", "bat_removed": "bat_removed", "sham": "sham"}
    rows = []
    for idx, image_id in enumerate(g10_ids):
        for cond_key, suffix in conditions.items():
            img_path = image_dir / f"{image_id}_{suffix}.png"
            if not img_path.exists():
                continue
            image = Image.open(img_path).convert("RGB")
            feats = extractor.extract(image, question_text)
            e_vision = float(vision_probe.decision_function(feats["vision_feature"][None, :])[0])
            e_proj = float(proj_probe.decision_function(feats["projector_feature"][None, :])[0])

            rows.append({"image_id": image_id, "condition": cond_key, "stage": "vision_tower", "stage_order": 0, "e_ball": e_vision})
            rows.append({"image_id": image_id, "condition": cond_key, "stage": "projector", "stage_order": 1, "e_ball": e_proj})
            for layer_idx in range(extractor.num_layers + 1):
                rows.append(
                    {
                        "image_id": image_id,
                        "condition": cond_key,
                        "stage": f"llm_layer_{layer_idx}",
                        "stage_order": 2 + layer_idx,
                        "e_ball": feats[f"logit_lens_layer_{layer_idx}"],
                    }
                )
        if (idx + 1) % 10 == 0:
            print(f"[exp5] {idx + 1}/{len(g10_ids)} images done")
            torch.cuda.empty_cache()

    out_path = output_dir / "exp5_layerwise_evidence.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp5] wrote {len(rows)} rows to {out_path}")

    # --- Statistics: per-stage mean Delta_bat_to_ball / Delta_sham ---
    import pandas as pd

    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["image_id", "stage", "stage_order"], columns="condition", values="e_ball").reset_index()
    wide["delta_bat_to_ball"] = wide["original"] - wide["bat_removed"]
    wide["delta_sham"] = wide["original"] - wide["sham"]

    per_stage = (
        wide.groupby(["stage", "stage_order"])[["delta_bat_to_ball", "delta_sham"]]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("stage_order")
    )
    per_stage.columns = ["stage", "stage_order", "delta_bat_mean", "delta_bat_std", "delta_sham_mean", "delta_sham_std"]

    statistics = {
        "n_g10_images": len(g10_ids),
        "n_probe_train_per_class": len(pos_ids),
        "per_stage": per_stage.to_dict(orient="records"),
    }
    stats_path = output_dir / "exp5_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp5] wrote statistics to {stats_path}")
    print(per_stage.to_string(index=False))


if __name__ == "__main__":
    main()
