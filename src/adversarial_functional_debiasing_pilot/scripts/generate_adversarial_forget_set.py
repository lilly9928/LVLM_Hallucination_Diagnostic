"""Step 1: generate adversarial forget images for TRAIN G10 only, reusing the
EXISTING Stage 3/11 PGD implementation unchanged (pgd_attack_with_restarts) at a
FIXED epsilon (chosen in configs/pilot.yaml BEFORE inspecting any debiasing
result -- see the yaml comment for why 16/255 was picked).

The adversarial image must remain a Bat+/Ball- image in ground truth -- we only
perturb pixels, never touch annotations. These images are training data ONLY for
Variant B (Adv Debias); they are never used for evaluation (see README "Critical
Interpretation" in the task spec).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/generate_adversarial_forget_set.py \
        --config configs/pilot.yaml [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../CooccurrenceHallucinationDiagnostic/src

from cooc_diagnostic.coco_index import load_coco_instances  # noqa: E402
from cooc_diagnostic.llava_runtime import (  # noqa: E402
    build_inputs,
    detect_yes_no_decision_point,
    load_model,
    yes_no_logits,
    yes_no_margin,
)
from cooc_diagnostic.pgd_attack import pgd_attack_with_restarts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device
    out_dir = Path(config["output_dir"])
    adv_img_dir = out_dir / "adversarial_images"
    adv_img_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(config["seed"]))

    print(f"[adv-gen] loading model {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    train_rows = list(csv.DictReader((out_dir / "data" / "train_split.csv").open("r", encoding="utf-8")))
    g10_train_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "G10_forget"})
    print(f"[adv-gen] {len(g10_train_ids)} TRAIN G10 images to attack")

    probe_image = load_image(g10_train_ids[0])
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[adv-gen] decision point: yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    epsilon = float(config["attack"]["epsilon"])
    pgd_steps = int(config["attack"]["pgd_steps"])
    n_restarts = int(config["attack"]["n_restarts"])
    target_category = config["target_category"]

    rows = []
    for i, image_id in enumerate(g10_train_ids):
        clean_image_path = str(image_dir / val_index.image_filenames[image_id])
        image = load_image(image_id)
        input_ids, image01 = build_inputs(processor, target_category, image, device)

        clean_yes, clean_no = yes_no_logits(model, processor, decision_point, input_ids, image01)
        clean_s_ball = clean_yes - clean_no

        def margin_fn(img: torch.Tensor) -> torch.Tensor:
            return yes_no_margin(model, processor, decision_point, input_ids, img)

        pgd_result = pgd_attack_with_restarts(image01, epsilon, pgd_steps, margin_fn, n_restarts)

        adv_yes, adv_no = yes_no_logits(model, processor, decision_point, input_ids, pgd_result.best_image)
        adv_s_ball = adv_yes - adv_no

        adv_image_path = adv_img_dir / f"{image_id:012d}_adv.png"
        adv_pixels = (pgd_result.best_image[0].detach().cpu().clamp(0, 1) * 255).round().byte()
        Image.fromarray(adv_pixels.permute(1, 2, 0).numpy(), mode="RGB").save(adv_image_path)

        rows.append(
            {
                "image_id": image_id,
                "clean_image_path": clean_image_path,
                "adv_image_path": str(adv_image_path),
                "clean_s_ball": clean_s_ball,
                "adv_s_ball": adv_s_ball,
                "delta_s_ball": adv_s_ball - clean_s_ball,
                "epsilon": epsilon,
                "attack_success": bool(pgd_result.flipped_by_margin),
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[adv-gen] {i + 1}/{len(g10_train_ids)} done")
            torch.cuda.empty_cache()

    out_csv = out_dir / "data" / "adversarial_forget_set.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    success_rate = sum(r["attack_success"] for r in rows) / n
    mean_clean = sum(r["clean_s_ball"] for r in rows) / n
    mean_adv = sum(r["adv_s_ball"] for r in rows) / n
    mean_delta = sum(r["delta_s_ball"] for r in rows) / n

    print(f"\n[adv-gen] n={n} epsilon={epsilon}")
    print(f"[adv-gen] attack success rate (margin>0 reached): {success_rate:.3f}")
    print(f"[adv-gen] mean clean s_ball: {mean_clean:.4f}")
    print(f"[adv-gen] mean adv   s_ball: {mean_adv:.4f}")
    print(f"[adv-gen] mean delta s_ball: {mean_delta:.4f}")
    print(f"[adv-gen] wrote {out_csv}")


if __name__ == "__main__":
    main()
