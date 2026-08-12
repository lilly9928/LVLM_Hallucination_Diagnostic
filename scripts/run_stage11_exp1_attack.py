"""Stage 11 Exp1 (part B): phenomenon test. For each matched G10 (treatment) /
G00 (control) pair from run_stage11_exp1_build_sample.py, run the SAME
epsilon*/PGD attack as Stage 3 (unchanged attack definition), targeting
"Is there a sports ball in the image?".

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp1_attack.py \
        --config configs/stage11_case_bat_ball.yaml [--device cuda:2] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.epsilon_star import find_epsilon_star
from cooc_diagnostic.llava_runtime import (
    build_inputs,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_margin,
)
from cooc_diagnostic.pgd_attack import pgd_attack_with_restarts
from cooc_diagnostic.survival_analysis import (
    build_survival_frame,
    holm_correction,
    mcnemar_already_yes_test,
    paired_bootstrap_median_diff,
    pooled_logrank_test,
    stratified_cox_test,
    weibull_aft_time_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp1B: epsilon* attack G10 vs G00")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap number of pairs")
    return parser.parse_args()


def load_sample(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_response(model, processor, input_ids, image01) -> dict:
    text = generate_greedy_answer(model, processor, input_ids, image01)
    return {"response_text": text, "is_yes": is_yes_response(text), "flipped": is_yes_response(text)}


def run_attack_at_epsilon(model, processor, input_ids, image01, epsilon, n_steps, n_restarts) -> dict:
    def margin_fn(img: torch.Tensor) -> torch.Tensor:
        return yes_no_margin(model, processor, DECISION_POINT, input_ids, img)

    pgd_result = pgd_attack_with_restarts(image01, epsilon, n_steps, margin_fn, n_restarts)
    outcome = evaluate_response(model, processor, input_ids, pgd_result.best_image)
    outcome["margin_proxy"] = pgd_result.best_margin
    return outcome


def main() -> None:
    global DECISION_POINT
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device or config["device"]
    output_dir = Path(config["output_dir"])
    torch.manual_seed(int(config["seed"]))

    print(f"[exp1b] loading model {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)

    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    sample_rows = load_sample(output_dir / "exp1_sample_selection.csv")
    if args.limit:
        keep_pairs = sorted({int(r["pair_id"]) for r in sample_rows})[: args.limit]
        sample_rows = [r for r in sample_rows if int(r["pair_id"]) in keep_pairs]
    print(f"[exp1b] running on {len(sample_rows)} samples ({len(sample_rows)//2} pairs)")

    probe_image = load_image(int(sample_rows[0]["image_id"]))
    DECISION_POINT = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[exp1b] decision point: yes_id={DECISION_POINT.yes_token_id} no_id={DECISION_POINT.no_token_id}")

    eps_max = float(config["epsilon_max"])
    eps0 = float(config["eps0"])
    relative_tolerance = float(config["relative_tolerance"])
    pgd_steps = int(config["pgd_steps"])
    n_restarts = int(config["n_restarts"])

    results = []
    for i, row in enumerate(sample_rows):
        image_id = int(row["image_id"])
        category = row["category"]
        input_ids, image01 = build_inputs(processor, category, load_image(image_id), device)

        baseline = evaluate_response(model, processor, input_ids, image01)

        def attack_at_epsilon(epsilon: float) -> dict:
            return run_attack_at_epsilon(model, processor, input_ids, image01, epsilon, pgd_steps, n_restarts)

        eps_result = find_epsilon_star(attack_at_epsilon, baseline, eps_max, eps0, relative_tolerance)
        results.append(
            {
                "pair_id": int(row["pair_id"]),
                "arm": row["arm"],
                "group": row["group"],
                "image_id": image_id,
                "category": category,
                "clean_answer": baseline["response_text"],
                "clean_is_yes": baseline["is_yes"],
                "status": eps_result.status,
                "attack_status": eps_result.status,
                "epsilon_star": eps_result.epsilon_star,
                "n_attack_calls": eps_result.n_attack_calls,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[exp1b] {i + 1}/{len(sample_rows)} done")
            torch.cuda.empty_cache()

    out_path = output_dir / "exp1_epsilon_star.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[exp1b] wrote {len(results)} epsilon* rows to {out_path}")

    # --- Analysis: survival stats (reusing Stage 4's exact machinery) ---
    df = build_survival_frame(results, eps_max)
    rng = np.random.default_rng(int(config["seed"]))

    descriptive = {}
    for arm in ["treatment", "control"]:
        sub = df[df["arm"] == arm]
        obs = sub[sub["event_observed"] == 1]["duration"]
        descriptive[arm] = {
            "n": int(len(sub)),
            "n_already_yes": int(sub["already_yes"].sum()),
            "clean_yes_rate": float(sub["already_yes"].mean()),
            "n_flipped_or_already_yes": int(sub["event_observed"].sum()),
            "n_censored": int((sub["event_observed"] == 0).sum()),
            "mean_epsilon_star_observed": float(obs.mean()) if len(obs) else None,
            "median_epsilon_star_observed": float(obs.median()) if len(obs) else None,
            "sd_epsilon_star_observed": float(obs.std()) if len(obs) > 1 else None,
            "iqr_epsilon_star_observed": [float(obs.quantile(0.25)), float(obs.quantile(0.75))] if len(obs) else None,
        }

    logrank = pooled_logrank_test(df)
    cox = stratified_cox_test(df)
    aft = weibull_aft_time_ratio(df, eps0)
    boot = paired_bootstrap_median_diff(df, n_boot=2000, rng=rng)
    mcnemar = mcnemar_already_yes_test(df)
    holm = holm_correction({"cox_epsilon_star": cox["p_value"], "mcnemar_already_yes": mcnemar["p_value"]})

    meaningful_direction = cox["hazard_ratio"] > 1.0
    statistically_supported = cox["p_value"] < 0.05
    go = meaningful_direction or statistically_supported

    statistics = {
        "n_pairs": int(df["pair_id"].nunique()),
        "descriptive": descriptive,
        "pooled_logrank_test": logrank,
        "stratified_cox_test": cox,
        "weibull_aft_time_ratio": aft,
        "paired_bootstrap_median_diff": boot,
        "mcnemar_already_yes_test": mcnemar,
        "holm_corrected_p_values": holm,
        "go_no_go": {
            "meaningful_direction_HR_gt_1": meaningful_direction,
            "statistically_supported_p_lt_05": statistically_supported,
            "decision": "GO" if go else "STOP",
            "note": "HR>1 means G10 (bat context) flips at smaller epsilon budgets than matched G00, i.e. higher hallucination vulnerability.",
        },
    }
    stats_path = output_dir / "exp1_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp1b] wrote statistics to {stats_path}")
    print(f"[exp1b] Cox HR={cox['hazard_ratio']:.3f} [{cox['ci_lower']:.3f},{cox['ci_upper']:.3f}] p={cox['p_value']:.4f}")
    print(f"[exp1b] DECISION: {statistics['go_no_go']['decision']}")


if __name__ == "__main__":
    main()
