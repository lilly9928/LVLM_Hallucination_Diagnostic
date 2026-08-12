"""Stage 11 Exp2: clean (epsilon=0, no attack) functional evidence s_ball for
all four groups. G10/G00 reuse Exp1's exact matched sample (same image_ids,
so Exp3 can join 1:1 on image_id); G01 (ball present, bat absent) and G11
(ball present, bat present) are independent reference samples, not attacked.

Also produces Exp2B: the G10 ranked gallery (top/bottom/near-boundary by s_ball).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp2_clean_evidence.py \
        --config configs/stage11_case_bat_ball.yaml [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import (
    build_inputs,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp2: clean functional evidence")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device or config["device"]
    output_dir = Path(config["output_dir"])
    context_name = config["context_category"]
    target_name = config["target_category"]
    rng = random.Random(int(config["seed"]))

    val_index = load_coco_instances(config["val_annotation_path"])
    name_to_id = {c.name: c.id for c in val_index.categories}
    context_id = name_to_id[context_name]
    target_id = name_to_id[target_name]
    image_dir = Path(config["val_image_dir"])
    names_by_id = val_index.category_names

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    def present_objects(image_id: int) -> str:
        return "|".join(sorted(names_by_id[c] for c in val_index.image_categories.get(image_id, set())))

    # --- G10 / G00: reuse Exp1's exact matched sample ---
    sample_path = output_dir / "exp1_sample_selection.csv"
    with sample_path.open("r", encoding="utf-8") as f:
        exp1_rows = list(csv.DictReader(f))
    g10_ids = sorted({int(r["image_id"]) for r in exp1_rows if r["group"] == "G10"})
    g00_ids = sorted({int(r["image_id"]) for r in exp1_rows if r["group"] == "G00"})

    # --- G01 / G11: independent reference samples ---
    g01_pool, g11_pool = [], []
    for image_id, cats in val_index.image_categories.items():
        bat, ball = context_id in cats, target_id in cats
        if not bat and ball:
            g01_pool.append(image_id)
        elif bat and ball:
            g11_pool.append(image_id)
    rng.shuffle(g01_pool)
    n_g01 = int(config.get("g01_subsample_n", 65))
    g01_ids = sorted(g01_pool[:n_g01])
    g11_ids = sorted(g11_pool)  # small (32), use all
    print(f"[exp2] G10={len(g10_ids)} G00={len(g00_ids)} G01={len(g01_ids)}(of {len(g01_pool)}) G11={len(g11_ids)}")

    print(f"[exp2] loading model {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)
    probe_image = load_image(g10_ids[0])
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[exp2] decision point: yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    results = []
    for group, ids in [("G10", g10_ids), ("G00", g00_ids), ("G01", g01_ids), ("G11", g11_ids)]:
        for image_id in ids:
            image = load_image(image_id)
            input_ids, image01 = build_inputs(processor, target_name, image, device)
            yes_logit, no_logit = yes_no_logits(model, processor, decision_point, input_ids, image01)
            response_text = generate_greedy_answer(model, processor, input_ids, image01)
            cats = val_index.image_categories.get(image_id, set())
            results.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_dir / val_index.image_filenames[image_id]),
                    "group": group,
                    "bat_present": int(context_id in cats),
                    "ball_present": int(target_id in cats),
                    "present_objects": present_objects(image_id),
                    "yes_logit": yes_logit,
                    "no_logit": no_logit,
                    "s_ball": yes_logit - no_logit,
                    "p_yes": None,  # filled below (needs torch softmax over yes/no only, see note)
                    "prediction": "Yes" if is_yes_response(response_text) else "No",
                    "response_text": response_text,
                }
            )
        print(f"[exp2] finished group={group} ({len(ids)} images)")

    # P_yes: softmax restricted to {yes, no} logits (the two-way decision probability).
    import math

    for r in results:
        s = r["s_ball"]
        r["p_yes"] = 1.0 / (1.0 + math.exp(-s))

    out_path = output_dir / "exp2_clean_evidence.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[exp2] wrote {len(results)} rows to {out_path}")

    # --- Exp2B: G10 ranked gallery ---
    g10_rows = sorted([r for r in results if r["group"] == "G10"], key=lambda r: -r["s_ball"])
    ranked_path = output_dir / "exp2_g10_ranked.csv"
    with ranked_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(g10_rows[0].keys()) + ["rank_desc"])
        writer.writeheader()
        for i, r in enumerate(g10_rows):
            row = dict(r)
            row["rank_desc"] = i + 1
            writer.writerow(row)
    print(f"[exp2] wrote {len(g10_rows)} ranked G10 rows to {ranked_path}")

    top20 = g10_rows[:20]
    bottom20 = g10_rows[-20:]
    near_boundary = sorted(g10_rows, key=lambda r: abs(r["s_ball"]))[:20]
    selections = {
        "top20_s_ball": [r["image_id"] for r in top20],
        "bottom20_s_ball": [r["image_id"] for r in bottom20],
        "near_boundary_20": [r["image_id"] for r in near_boundary],
    }
    with (output_dir / "exp2_g10_selections.json").open("w", encoding="utf-8") as f:
        json.dump(selections, f, indent=2)

    # --- Descriptive statistics ---
    import numpy as np
    from scipy import stats as sstats

    def desc(rows: list[dict]) -> dict:
        vals = np.array([r["s_ball"] for r in rows])
        yes = np.array([1 if r["prediction"] == "Yes" else 0 for r in rows])
        return {
            "n": len(rows),
            "mean_s_ball": float(vals.mean()),
            "median_s_ball": float(np.median(vals)),
            "sd_s_ball": float(vals.std(ddof=1)) if len(vals) > 1 else None,
            "iqr_s_ball": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
            "yes_rate": float(yes.mean()),
            "mean_p_yes": float(np.mean([r["p_yes"] for r in rows])),
        }

    by_group = {g: desc([r for r in results if r["group"] == g]) for g in ["G00", "G10", "G01", "G11"]}

    g10_vals = np.array([r["s_ball"] for r in results if r["group"] == "G10"])
    g00_vals = np.array([r["s_ball"] for r in results if r["group"] == "G00"])
    mean_diff = float(g10_vals.mean() - g00_vals.mean())
    pooled_sd = math.sqrt(((len(g10_vals) - 1) * g10_vals.std(ddof=1) ** 2 + (len(g00_vals) - 1) * g00_vals.std(ddof=1) ** 2) / (len(g10_vals) + len(g00_vals) - 2))
    cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else None
    mwu = sstats.mannwhitneyu(g10_vals, g00_vals, alternative="two-sided")

    rng_np = np.random.default_rng(int(config["seed"]))
    n_boot = 5000
    boot_diffs = []
    for _ in range(n_boot):
        g10_bs = rng_np.choice(g10_vals, size=len(g10_vals), replace=True)
        g00_bs = rng_np.choice(g00_vals, size=len(g00_vals), replace=True)
        boot_diffs.append(g10_bs.mean() - g00_bs.mean())
    boot_diffs = np.sort(boot_diffs)
    ci_lower = float(boot_diffs[int(0.025 * n_boot)])
    ci_upper = float(boot_diffs[int(0.975 * n_boot)])

    statistics = {
        "by_group": by_group,
        "primary_comparison_G10_vs_G00": {
            "mean_diff_s_ball": mean_diff,
            "cohens_d": cohens_d,
            "bootstrap_95ci": [ci_lower, ci_upper],
            "mann_whitney_u": float(mwu.statistic),
            "mann_whitney_p": float(mwu.pvalue),
        },
    }
    stats_path = output_dir / "exp2_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp2] wrote statistics to {stats_path}")
    print(f"[exp2] G10 mean s_ball={by_group['G10']['mean_s_ball']:.3f}  G00 mean s_ball={by_group['G00']['mean_s_ball']:.3f}")
    print(f"[exp2] diff={mean_diff:.3f} Cohen's d={cohens_d:.3f} 95% CI=[{ci_lower:.3f},{ci_upper:.3f}] MWU p={mwu.pvalue:.4g}")


if __name__ == "__main__":
    main()
