"""Stage 3: targeted PGD attack and epsilon* search.

Run in pilot mode first (small N, mandatory sanity checks + timing) before
committing to a full run:

    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage3_attack.py \
        --config configs/stage3_attack.yaml --pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

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
from cooc_diagnostic.random_attack import random_perturbations
from cooc_diagnostic.sanity_checks import (
    compare_attack_vs_random_control,
    summarize_attack_success_rate,
    summarize_present_object_baseline,
    summarize_random_noise_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: targeted attack and epsilon* search")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Run only pilot_n_per_arm pairs for sanity checks + timing")
    return parser.parse_args()


def load_matched_pairs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_response(model, processor, decision_point, input_ids, image01) -> dict:
    text = generate_greedy_answer(model, processor, input_ids, image01)
    return {"response_text": text, "is_yes": is_yes_response(text), "flipped": is_yes_response(text)}


def run_attack_at_epsilon(model, processor, decision_point, input_ids, image01, epsilon, n_steps, n_restarts) -> dict:
    def margin_fn(img: torch.Tensor) -> torch.Tensor:
        return yes_no_margin(model, processor, decision_point, input_ids, img)

    pgd_result = pgd_attack_with_restarts(image01, epsilon, n_steps, margin_fn, n_restarts)
    outcome = evaluate_response(model, processor, decision_point, input_ids, pgd_result.best_image)
    outcome["margin_proxy"] = pgd_result.best_margin
    return outcome


def run_random_control_at_epsilon(model, processor, decision_point, input_ids, image01, epsilon, n_trials) -> dict:
    for candidate in random_perturbations(image01, epsilon, n_trials):
        outcome = evaluate_response(model, processor, decision_point, input_ids, candidate)
        if outcome["flipped"]:
            return outcome
    return outcome


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(int(config["seed"]))
    rng = random.Random(int(config["seed"]))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage3] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])

    print(f"[stage3] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    print("[stage3] detecting Yes/No decision point")
    sample_pairs = load_matched_pairs(Path(config["matched_pairs_path"]))
    probe_image_id = int(sample_pairs[0]["image_id_treatment"])
    probe_image = Image.open(image_dir / val_index.image_filenames[probe_image_id]).convert("RGB")
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, config["device"])
    print(f"[stage3] decision point: prefix_ids={decision_point.prefix_ids} yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    n_per_arm = int(config["pilot_n_per_arm"]) if args.pilot else len(sample_pairs)
    rng.shuffle(sample_pairs)
    pilot_pairs = sample_pairs[:n_per_arm]
    print(f"[stage3] running on {len(pilot_pairs)} matched pairs ({'PILOT' if args.pilot else 'FULL'} mode)")

    # --- Sanity check 1: present-object baseline at epsilon=0 ---
    baseline_records = []
    present_image_ids = [iid for iid, cats in val_index.image_categories.items() if cats]
    rng.shuffle(present_image_ids)
    names_by_id = val_index.category_names
    for image_id in present_image_ids[:n_per_arm]:
        present_cat_id = next(iter(val_index.image_categories[image_id]))
        category = names_by_id[present_cat_id]
        input_ids, image01 = build_inputs(processor, category, load_image(image_id), config["device"])
        outcome = evaluate_response(model, processor, decision_point, input_ids, image01)
        baseline_records.append({"image_id": image_id, "category": category, "is_yes": outcome["is_yes"], "response_text": outcome["response_text"]})
    baseline_summary = summarize_present_object_baseline(baseline_records)
    print(f"[stage3] SANITY CHECK 1 (present-object baseline @ eps=0): yes_rate={baseline_summary['yes_rate']:.3f} passed={baseline_summary['passed']}")

    # --- Sanity checks 2 & 3: attack success vs random control @ sanity_check_epsilon ---
    eps_sanity = float(config["sanity_check_epsilon"])
    attack_records, random_records, timing = [], [], []
    for row in pilot_pairs:
        image_id = int(row["image_id_treatment"])
        category = row["category_treatment"]
        input_ids, image01 = build_inputs(processor, category, load_image(image_id), config["device"])

        t0 = time.time()
        attack_outcome = run_attack_at_epsilon(model, processor, decision_point, input_ids, image01, eps_sanity, int(config["pgd_steps"]), int(config["n_restarts"]))
        t_attack = time.time() - t0
        attack_records.append(attack_outcome)

        t0 = time.time()
        random_outcome = run_random_control_at_epsilon(model, processor, decision_point, input_ids, image01, eps_sanity, int(config["random_control_n_trials"]))
        t_random = time.time() - t0
        random_records.append(random_outcome)

        timing.append({"image_id": image_id, "category": category, "t_attack_sec": t_attack, "t_random_sec": t_random})

    attack_summary = summarize_attack_success_rate(attack_records, epsilon_label=f"{eps_sanity:.5f}")
    random_summary = summarize_random_noise_control(random_records, epsilon_label=f"{eps_sanity:.5f}")
    comparison = compare_attack_vs_random_control(attack_summary, random_summary)
    print(f"[stage3] SANITY CHECK 2 (attack success @ 16/255): success_rate={attack_summary['success_rate']:.3f} passed={attack_summary['passed']}")
    print(f"[stage3] SANITY CHECK 3 (random-noise control @ 16/255): flip_rate={random_summary['flip_rate']:.3f} passed={random_summary['passed']}")
    print(f"[stage3] SANITY CHECK 3b (attack vs random gap): gap={comparison['gap']:.3f} passed={comparison['passed']}")

    avg_t_attack = sum(t["t_attack_sec"] for t in timing) / len(timing)
    avg_t_random = sum(t["t_random_sec"] for t in timing) / len(timing)
    print(f"[stage3] avg wall-clock per sample: attack@16/255={avg_t_attack:.2f}s random@16/255={avg_t_random:.2f}s")

    all_checks_passed = baseline_summary["passed"] and attack_summary["passed"] and random_summary["passed"] and comparison["passed"]
    print(f"[stage3] ALL MANDATORY SANITY CHECKS PASSED: {all_checks_passed}")

    with (output_dir / ("pilot_sanity_report.json" if args.pilot else "sanity_report.json")).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": baseline_summary,
                "attack": attack_summary,
                "random_control": random_summary,
                "comparison": comparison,
                "all_checks_passed": all_checks_passed,
                "avg_t_attack_sec": avg_t_attack,
                "avg_t_random_sec": avg_t_random,
                "timing_records": timing,
            },
            f,
            indent=2,
        )

    if not all_checks_passed:
        print("[stage3] STOPPING: mandatory sanity checks failed -- fix the pipeline before running epsilon* search.")
        return

    if args.pilot:
        print("[stage3] pilot sanity checks passed. Re-run without --pilot (after confirming full-run sample size) to compute epsilon*.")
        return

    # --- Full run: epsilon* bisection search for every matched pair ---
    eps_max = float(config["epsilon_max"])
    eps0 = float(config["eps0"])
    relative_tolerance = float(config["relative_tolerance"])
    results = []
    for arm, id_key, cat_key in [("treatment", "image_id_treatment", "category_treatment"), ("control", "image_id_control", "category_control")]:
        for row in sample_pairs:
            image_id = int(row[id_key])
            category = row[cat_key]
            input_ids, image01 = build_inputs(processor, category, load_image(image_id), config["device"])

            baseline = evaluate_response(model, processor, decision_point, input_ids, image01)

            def attack_at_epsilon(epsilon: float) -> dict:
                return run_attack_at_epsilon(model, processor, decision_point, input_ids, image01, epsilon, int(config["pgd_steps"]), int(config["n_restarts"]))

            eps_result = find_epsilon_star(attack_at_epsilon, baseline, eps_max, eps0, relative_tolerance)
            results.append(
                {
                    "pair_id": row["pair_id"],
                    "arm": arm,
                    "image_id": image_id,
                    "category": category,
                    "status": eps_result.status,
                    "epsilon_star": eps_result.epsilon_star,
                    "n_attack_calls": eps_result.n_attack_calls,
                }
            )

    with (output_dir / "epsilon_star_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[stage3] epsilon* results written for {len(results)} samples to {output_dir / 'epsilon_star_results.csv'}")


if __name__ == "__main__":
    main()
