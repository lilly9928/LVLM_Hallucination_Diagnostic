"""Stage 11 Exp4 (+4B): co-occurrence specificity via counterfactual bat removal.

For every G10 (bat+, ball-) image: gray-fill the bat region (validated technique
from extract_features_counterfactual.py::mask_dog_regions, generalized) and a
sham region (mirrored + translated, see masking.py docstring), then compare
clean s_ball across original / bat-removed / sham using the same
llava_runtime.yes_no_logits readout as Exp2.

Delta_bat_to_ball = s_ball(original) - s_ball(bat_removed)
Delta_sham        = s_ball(original) - s_ball(sham)

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp4_counterfactual.py \
        --config configs/stage11_case_bat_ball.yaml [--device cuda:2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import build_inputs, detect_yes_no_decision_point, load_model, yes_no_logits
from cooc_diagnostic.masking import apply_bat_removal, apply_sham_removal, load_raw_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp4: counterfactual bat removal")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device or config["device"]
    output_dir = Path(config["output_dir"])
    image_out_dir = output_dir / "counterfactual_images"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    context_name = config["context_category"]
    target_name = config["target_category"]

    val_index = load_coco_instances(config["val_annotation_path"])
    name_to_id = {c.name: c.id for c in val_index.categories}
    context_id = name_to_id[context_name]
    target_id = name_to_id[target_name]
    image_dir = Path(config["val_image_dir"])

    raw = load_raw_annotations(config["val_annotation_path"])

    g10_ids = sorted(
        iid for iid, cats in val_index.image_categories.items() if context_id in cats and target_id not in cats
    )
    print(f"[exp4] {len(g10_ids)} G10 images")

    print(f"[exp4] loading model {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)
    probe_image = Image.open(image_dir / val_index.image_filenames[g10_ids[0]]).convert("RGB")
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)

    def s_ball(image: Image.Image) -> float:
        input_ids, image01 = build_inputs(processor, target_name, image, device)
        yes_logit, no_logit = yes_no_logits(model, processor, decision_point, input_ids, image01)
        return yes_logit - no_logit

    rows = []
    for image_id in g10_ids:
        img_path = image_dir / val_index.image_filenames[image_id]
        original = Image.open(img_path).convert("RGB")
        anns = raw.image_annotations.get(image_id, [])
        bat_anns = [a for a in anns if int(a["category_id"]) == context_id]
        other_anns = [a for a in anns if int(a["category_id"]) != context_id]
        if not bat_anns:
            continue  # defensive; should not happen given G10 filter above

        bat_removed = apply_bat_removal(original, bat_anns)
        sham = apply_sham_removal(original, bat_removed.mask, other_anns)

        s_orig = s_ball(original)
        s_bat = s_ball(bat_removed.image)
        s_sham = s_ball(sham.image)

        orig_path = image_out_dir / f"{image_id}_original.png"
        bat_path = image_out_dir / f"{image_id}_bat_removed.png"
        sham_path = image_out_dir / f"{image_id}_sham.png"
        original.save(orig_path)
        bat_removed.image.save(bat_path)
        sham.image.save(sham_path)

        base_row = {
            "image_id": image_id,
            "mask_area_px": bat_removed.mask_area_px,
            "sham_mask_area_px": sham.mask_area_px,
            "sham_overlap_residual_px": sham.metadata["sham_overlap_residual_px"],
            "sham_overlap_resolved": sham.metadata["sham_overlap_resolved"],
            "sham_vertical_shift_frac": sham.metadata["sham_vertical_shift_frac"],
            "n_bat_instances": bat_removed.metadata["n_bat_instances"],
            "original_s_ball": s_orig,
        }
        rows.append({**base_row, "condition": "bat_removed", "intervention_s_ball": s_bat, "delta": s_orig - s_bat, "image_path": str(bat_path)})
        rows.append({**base_row, "condition": "sham", "intervention_s_ball": s_sham, "delta": s_orig - s_sham, "image_path": str(sham_path)})

        if len(rows) % 20 == 0:
            print(f"[exp4] {len(rows)//2}/{len(g10_ids)} images done")

    out_path = output_dir / "exp4_counterfactual.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp4] wrote {len(rows)} rows ({len(rows)//2} images) to {out_path}")

    # --- Statistics ---
    import numpy as np
    from scipy import stats as sstats

    delta_bat = np.array([r["delta"] for r in rows if r["condition"] == "bat_removed"])
    delta_sham = np.array([r["delta"] for r in rows if r["condition"] == "sham"])
    n = len(delta_bat)

    wilcoxon_bat_gt_0 = sstats.wilcoxon(delta_bat, alternative="greater") if n > 0 else None
    wilcoxon_paired = sstats.wilcoxon(delta_bat, delta_sham, alternative="greater") if n > 0 else None

    rng = np.random.default_rng(int(config["seed"]))
    n_boot = 5000
    boot_mean_bat, boot_mean_diff = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_mean_bat.append(delta_bat[idx].mean())
        boot_mean_diff.append((delta_bat[idx] - delta_sham[idx]).mean())
    boot_mean_bat.sort()
    boot_mean_diff.sort()

    statistics = {
        "n_g10_images": n,
        "delta_bat_to_ball": {
            "mean": float(delta_bat.mean()),
            "median": float(np.median(delta_bat)),
            "sd": float(delta_bat.std(ddof=1)),
            "bootstrap_95ci_mean": [float(boot_mean_bat[int(0.025 * n_boot)]), float(boot_mean_bat[int(0.975 * n_boot)])],
            "wilcoxon_greater_than_0_p": float(wilcoxon_bat_gt_0.pvalue) if wilcoxon_bat_gt_0 else None,
        },
        "delta_sham": {
            "mean": float(delta_sham.mean()),
            "median": float(np.median(delta_sham)),
            "sd": float(delta_sham.std(ddof=1)),
        },
        "delta_bat_minus_delta_sham": {
            "mean": float((delta_bat - delta_sham).mean()),
            "bootstrap_95ci": [float(boot_mean_diff[int(0.025 * n_boot)]), float(boot_mean_diff[int(0.975 * n_boot)])],
            "wilcoxon_bat_gt_sham_p": float(wilcoxon_paired.pvalue) if wilcoxon_paired else None,
        },
        "n_sham_overlap_unresolved": int(sum(1 for r in rows if r["condition"] == "sham" and not r["sham_overlap_resolved"])),
        "hypotheses": {
            "delta_bat_to_ball_gt_0": bool(delta_bat.mean() > 0),
            "delta_bat_to_ball_gt_delta_sham": bool(delta_bat.mean() > delta_sham.mean()),
        },
    }
    stats_path = output_dir / "exp4_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp4] wrote statistics to {stats_path}")
    print(f"[exp4] mean Delta_bat_to_ball={delta_bat.mean():.3f}  mean Delta_sham={delta_sham.mean():.3f}")
    print(f"[exp4] hypotheses: {statistics['hypotheses']}")


if __name__ == "__main__":
    main()
