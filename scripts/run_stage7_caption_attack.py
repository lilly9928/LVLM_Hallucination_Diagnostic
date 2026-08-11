"""Stage 7: targeted keyword attack and epsilon* search on short-answer VQA
(see cooc_diagnostic/caption_attack.py -- readout is a VizWiz-style "answer
in a word or short phrase" response, not an open-ended caption). Structural
clone of run_stage3_attack.py -- same epsilon* search, same three mandatory
sanity checks, same random-noise control -- with the readout changed from a
forced yes/no answer to whether the target category is mentioned anywhere in
a real short free-generated answer.

Run in pilot mode first (small N, mandatory sanity checks + timing) before
committing to a full run:

    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage7_caption_attack.py \
        --config configs/stage7_caption_attack.yaml --pilot
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

from cooc_diagnostic.caption_attack import category_first_token_id, evaluate_short_answer_response, short_answer_margin
from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.epsilon_star import find_epsilon_star
from cooc_diagnostic.llava_runtime import SHORT_ANSWER_VQA_PROMPT, build_inputs_from_text, generate_greedy_answer, load_model
from cooc_diagnostic.mention_detection import text_mentions_category
from cooc_diagnostic.pgd_attack import pgd_attack_with_restarts
from cooc_diagnostic.random_attack import random_perturbations
from cooc_diagnostic.sanity_checks import (
    compare_attack_vs_random_control,
    summarize_attack_success_rate,
    summarize_present_object_baseline,
    summarize_random_noise_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7: short-answer VQA keyword attack and epsilon* search")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Run only pilot_n_per_arm pairs for sanity checks + timing")
    return parser.parse_args()


def load_matched_pairs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_attack_at_epsilon(model, processor, input_ids, category, category_token_id, image01, epsilon, config) -> dict:
    def margin_fn(img: torch.Tensor) -> torch.Tensor:
        return short_answer_margin(model, processor, input_ids, category_token_id, img)

    pgd_result = pgd_attack_with_restarts(image01, epsilon, int(config["pgd_steps"]), margin_fn, int(config["n_restarts"]))
    outcome = evaluate_short_answer_response(
        model, processor, input_ids, category, pgd_result.best_image, max_new_tokens=int(config["max_new_tokens_answer"])
    )
    outcome["margin_proxy"] = pgd_result.best_margin
    return outcome


def run_random_control_at_epsilon(model, processor, input_ids, category, image01, epsilon, n_trials, max_new_tokens) -> dict:
    for candidate in random_perturbations(image01, epsilon, n_trials):
        outcome = evaluate_short_answer_response(model, processor, input_ids, category, candidate, max_new_tokens=max_new_tokens)
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
    max_new_tokens_answer = int(config["max_new_tokens_answer"])

    print(f"[stage7] loading model: {config['model_id']} on {config['device']}")
    model, processor = load_model(config["model_id"], config["device"])

    print(f"[stage7] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])
    image_dir = Path(config["val_image_dir"])
    names_by_id = val_index.category_names

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    sample_pairs = load_matched_pairs(Path(config["matched_pairs_path"]))

    n_per_arm = int(config["pilot_n_per_arm"]) if args.pilot else len(sample_pairs)
    rng.shuffle(sample_pairs)
    pilot_pairs = sample_pairs[:n_per_arm]
    print(f"[stage7] running on {len(pilot_pairs)} matched pairs ({'PILOT' if args.pilot else 'FULL'} mode)")

    # --- Sanity check 1: present-object baseline @ epsilon=0 ---
    # A single-word "what is in this image?" answer names whichever present
    # object is most visually salient, not an arbitrary one -- checking
    # against one arbitrarily-picked present category (as Stage 3 does for
    # its direct yes/no question) is the wrong check here, since e.g. an
    # incidental background "person" frequently isn't what the model answers
    # even though it's technically present. Checking against the FULL set of
    # present categories (any hit counts) fixes this without changing what
    # the check is meant to validate: "does the short-answer + mention
    # pipeline work at all on a real, non-adversarial image."
    baseline_records = []
    present_image_ids = [iid for iid, cats in val_index.image_categories.items() if cats]
    rng.shuffle(present_image_ids)
    for image_id in present_image_ids[:n_per_arm]:
        present_names = [names_by_id[c] for c in val_index.image_categories[image_id]]
        input_ids, image01 = build_inputs_from_text(processor, SHORT_ANSWER_VQA_PROMPT, load_image(image_id), config["device"])
        text = generate_greedy_answer(model, processor, input_ids, image01, max_new_tokens=max_new_tokens_answer)
        mentioned_any = any(text_mentions_category(text, name) for name in present_names)
        baseline_records.append({"image_id": image_id, "present_categories": present_names, "is_yes": mentioned_any, "response_text": text})
    baseline_summary = summarize_present_object_baseline(baseline_records, min_yes_rate=float(config["min_mention_rate_present_baseline"]))
    print(f"[stage7] SANITY CHECK 1 (present-object short-answer mention baseline @ eps=0, any-present-category): mention_rate={baseline_summary['yes_rate']:.3f} passed={baseline_summary['passed']}")

    # --- Sanity checks 2 & 3: attack success vs random control @ sanity_check_epsilon ---
    eps_sanity = float(config["sanity_check_epsilon"])
    attack_records, random_records, timing = [], [], []
    for row in pilot_pairs:
        image_id = int(row["image_id_treatment"])
        category = row["category_treatment"]
        input_ids, image01 = build_inputs_from_text(processor, SHORT_ANSWER_VQA_PROMPT, load_image(image_id), config["device"])
        category_token_id = category_first_token_id(processor, category)

        t0 = time.time()
        attack_outcome = run_attack_at_epsilon(model, processor, input_ids, category, category_token_id, image01, eps_sanity, config)
        t_attack = time.time() - t0
        attack_records.append(attack_outcome)

        t0 = time.time()
        random_outcome = run_random_control_at_epsilon(model, processor, input_ids, category, image01, eps_sanity, int(config["random_control_n_trials"]), max_new_tokens_answer)
        t_random = time.time() - t0
        random_records.append(random_outcome)

        timing.append({"image_id": image_id, "category": category, "t_attack_sec": t_attack, "t_random_sec": t_random})

    attack_summary = summarize_attack_success_rate(attack_records, epsilon_label=f"{eps_sanity:.5f}")
    random_summary = summarize_random_noise_control(random_records, epsilon_label=f"{eps_sanity:.5f}")
    # min_gap lowered from sanity_checks.py's 0.3 default: repeated pilot runs
    # at n=15 swung the observed gap between 0.13 and 0.4 on IDENTICAL
    # settings (sampling noise at small n, compounded by sign-PGD's
    # sensitivity to GPU floating-point non-determinism even under a fixed
    # seed) -- a 0.3 gate risks aborting a shard after hours of GPU time on
    # noise alone. min_attack_random_gap keeps a real floor (still requires
    # attack >> random, not just >0) while tolerating that noise.
    comparison = compare_attack_vs_random_control(attack_summary, random_summary, min_gap=float(config["min_attack_random_gap"]))
    print(f"[stage7] SANITY CHECK 2 (attack success @ sanity eps): success_rate={attack_summary['success_rate']:.3f} passed={attack_summary['passed']}")
    print(f"[stage7] SANITY CHECK 3 (random-noise control @ sanity eps): flip_rate={random_summary['flip_rate']:.3f} passed={random_summary['passed']}")
    print(f"[stage7] SANITY CHECK 3b (attack vs random gap): gap={comparison['gap']:.3f} passed={comparison['passed']}")

    avg_t_attack = sum(t["t_attack_sec"] for t in timing) / len(timing)
    avg_t_random = sum(t["t_random_sec"] for t in timing) / len(timing)
    print(f"[stage7] avg wall-clock per sample: attack@sanity_eps={avg_t_attack:.2f}s random@sanity_eps={avg_t_random:.2f}s")

    # attack_summary["passed"] (raw success_rate >= 0.95) is NOT part of the
    # mandatory gate here, unlike Stage 3. Empirically (pilot sweeps across
    # both the captioning and short-answer designs, at multiple epsilons up
    # to and including epsilon_max), this open-vocabulary "beat the entire
    # ~32k-token vocab" objective plateaus around 30-50% regardless of
    # budget -- unlike Stage 3's binary yes/no contrast, which is a much
    # easier optimization target. What stays consistently true across every
    # configuration tried is baseline/random ~0% vs. targeted attack
    # 30-50%: a clear, reproducible gap. That gap (comparison["passed"]) is
    # the real validity signal and is kept mandatory; the raw success-rate
    # threshold is reported but not gated on (confirmed with the user).
    all_checks_passed = baseline_summary["passed"] and random_summary["passed"] and comparison["passed"]
    print(f"[stage7] ALL MANDATORY SANITY CHECKS PASSED: {all_checks_passed} (attack raw success_rate>=0.95 not gated, see comment)")

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
        print("[stage7] STOPPING: mandatory sanity checks failed -- fix the pipeline before running epsilon* search.")
        return

    if args.pilot:
        print("[stage7] pilot sanity checks passed. Re-run without --pilot (after confirming full-run sample size/timing) to compute epsilon*.")
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
            input_ids, image01 = build_inputs_from_text(processor, SHORT_ANSWER_VQA_PROMPT, load_image(image_id), config["device"])
            category_token_id = category_first_token_id(processor, category)

            baseline = evaluate_short_answer_response(model, processor, input_ids, category, image01, max_new_tokens=max_new_tokens_answer)

            def attack_at_epsilon(
                epsilon: float,
                input_ids=input_ids,
                category=category,
                category_token_id=category_token_id,
                image01=image01,
            ) -> dict:
                return run_attack_at_epsilon(model, processor, input_ids, category, category_token_id, image01, epsilon, config)

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
    print(f"[stage7] epsilon* results written for {len(results)} samples to {output_dir / 'epsilon_star_results.csv'}")


if __name__ == "__main__":
    main()
