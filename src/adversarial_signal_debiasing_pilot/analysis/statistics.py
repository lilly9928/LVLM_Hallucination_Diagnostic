"""Part XI-XII: aggregate all four models' evaluation/*_results.csv (three
reused verbatim, one newly evaluated -- see evaluate_all_models.py) into the
primary result table, bootstrap statistics, and the pilot's GO/STRONG GO/
WEAK GO/NO-GO(A-D) decision (Part "GO / NO-GO DECISION" in the task brief).

Same bootstrap conventions as the prior pilot's analysis/compute_statistics.py
(new call site, extended from 3 to 4 models; that file is not imported or
modified).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python analysis/statistics.py --data-config configs/data.yaml
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

MODELS = ["original", "clean_debias", "adv_debias", "adv_decomp_debias"]
N_BOOT = 5000
RETENTION_DROP_THRESHOLD = 0.10  # 10pp accuracy drop -- descriptive, fixed before inspecting these numbers


def load_results(eval_dir: Path, model: str) -> list[dict]:
    fname = "adv_decomp_results.csv" if model == "adv_decomp_debias" else f"{model}_results.csv"
    with (eval_dir / fname).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["yes_logit"] = float(r["yes_logit"])
        r["no_logit"] = float(r["no_logit"])
        r["s_score"] = float(r["s_score"])
        r["correct"] = r["correct"] == "True"
    return rows


def by_role(rows: list[dict], role: str) -> list[dict]:
    return sorted((r for r in rows if r["role"] == role), key=lambda r: int(r["image_id"]))


def bootstrap_mean_ci(values: list[float], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    arr = np.array(values, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "ci_lower": float(np.percentile(boots, 2.5)), "ci_upper": float(np.percentile(boots, 97.5)), "n": len(arr)}


def bootstrap_unpaired_diff_ci(a: list[float], b: list[float], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    arr_a, arr_b = np.array(a, dtype=float), np.array(b, dtype=float)
    boots_a = rng.choice(arr_a, size=(n_boot, len(arr_a)), replace=True).mean(axis=1)
    boots_b = rng.choice(arr_b, size=(n_boot, len(arr_b)), replace=True).mean(axis=1)
    diffs = boots_a - boots_b
    return {"observed": float(arr_a.mean() - arr_b.mean()), "ci_lower": float(np.percentile(diffs, 2.5)), "ci_upper": float(np.percentile(diffs, 97.5))}


def bootstrap_paired_diff_ci(a: list[float], b: list[float], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    arr_a, arr_b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = len(arr_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)
    return {"observed": float(arr_a.mean() - arr_b.mean()), "ci_lower": float(np.percentile(diffs, 2.5)), "ci_upper": float(np.percentile(diffs, 97.5))}


def binomial_ci(n_correct: int, n_total: int) -> dict:
    from scipy.stats import beta

    if n_total == 0:
        return {"acc": None, "ci_lower": None, "ci_upper": None, "n": 0}
    alpha = 0.05
    lo = 0.0 if n_correct == 0 else beta.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
    hi = 1.0 if n_correct == n_total else beta.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
    return {"acc": n_correct / n_total, "ci_lower": float(lo), "ci_upper": float(hi), "n": n_total}


def load_component_selectivity_summary(out_dir: Path) -> dict:
    path = out_dir / "decomposition" / "component_selectivity.csv"
    if not path.exists():
        return {"available": False}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    decomposed = [r for r in rows if r["candidate_type"] in ("pca_component", "pls_component")]
    mean_row = next((r for r in rows if r["candidate_type"] == "mean_direction"), None)
    best_random = max((r for r in rows if r["candidate_type"] == "random_direction"), key=lambda r: float(r["selectivity_min_over_lambda"]), default=None)
    if not decomposed:
        return {"available": False}
    best = max(decomposed, key=lambda r: float(r["selectivity_min_over_lambda"]))
    beats_mean = mean_row is not None and float(best["selectivity_min_over_lambda"]) > float(mean_row["selectivity_min_over_lambda"])
    beats_random = best_random is not None and float(best["selectivity_min_over_lambda"]) > float(best_random["selectivity_min_over_lambda"])
    return {
        "available": True,
        "best_component": f"{best['candidate_type']}:{best['candidate_id']}",
        "best_selectivity": float(best["selectivity_min_over_lambda"]),
        "mean_direction_selectivity": float(mean_row["selectivity_min_over_lambda"]) if mean_row else None,
        "best_random_selectivity": float(best_random["selectivity_min_over_lambda"]) if best_random else None,
        "beats_mean_direction": beats_mean,
        "beats_best_random_direction": beats_random,
        "decomposition_isolates_selective_component": bool(beats_mean and beats_random),
    }


def classify_go_no_go(component_summary: dict, B: dict, ball_acc: dict, bat_acc: dict) -> dict:
    ball_drop = {m: ball_acc["original"] - ball_acc[m] for m in MODELS}
    bat_drop = {m: bat_acc["original"] - bat_acc[m] for m in MODELS}
    retention_ok = {m: ball_drop[m] <= RETENTION_DROP_THRESHOLD and bat_drop[m] <= RETENTION_DROP_THRESHOLD for m in MODELS}

    decomp_isolates_component = component_summary.get("decomposition_isolates_selective_component", False)
    decomp_reduces_vs_original = B["adv_decomp_debias"] < B["original"]
    decomp_better_than_adv = B["adv_decomp_debias"] < B["adv_debias"]
    clean_at_least_as_good = B["clean_debias"] <= B["adv_decomp_debias"] and retention_ok["clean_debias"]
    decomp_more_selective_than_adv = (
        decomp_better_than_adv
        and ball_drop["adv_decomp_debias"] < ball_drop["adv_debias"]
        and bat_drop["adv_decomp_debias"] < bat_drop["adv_debias"]
    )

    if not component_summary.get("available", False) or not decomp_isolates_component:
        decision = ("NO-GO A: no separable component -- the best PCA/PLS component does not clearly beat the "
                    "mean-direction and random-direction baselines on the internal VAL selectivity test")
    elif decomp_reduces_vs_original and not retention_ok["adv_decomp_debias"]:
        decision = ("NO-GO C: generic suppression -- Adv+Decomp Debias reduces clean-test coupling B but degrades "
                     f"genuine Ball+ and/or Bat+ retention by more than {RETENTION_DROP_THRESHOLD:.0%}")
    elif not decomp_reduces_vs_original:
        decision = ("NO-GO B: component exists but training fails to transfer -- the selected component is "
                     "functionally selective on TRAIN VAL images, but L_spur training does not reduce coupling B "
                     "on unseen CLEAN TEST images")
    elif clean_at_least_as_good and B["clean_debias"] <= B["adv_debias"]:
        decision = "NO-GO D: adversarial exposure not needed -- Clean Debias performs as well as or better than Adv Debias and Adv+Decomp Debias"
    elif decomp_more_selective_than_adv and retention_ok["adv_decomp_debias"]:
        decision = ("STRONG GO: the selected component is more selective than mean/random baselines, and Adv+Decomp "
                     "Debias achieves lower clean-test coupling B than Adv Debias while preserving genuine Ball/Bat retention better")
    else:
        decision = ("WEAK GO: decomposition improves selectivity on the internal VAL test, but final Adv+Decomp "
                     "Debias is not clearly better than plain Adv Debias on the clean test set")

    return {
        "ball_plus_acc_drop": ball_drop,
        "bat_plus_acc_drop": bat_drop,
        "retention_ok": retention_ok,
        "retention_drop_threshold": RETENTION_DROP_THRESHOLD,
        "decomp_isolates_selective_component": decomp_isolates_component,
        "decomp_reduces_coupling_vs_original": decomp_reduces_vs_original,
        "decomp_better_than_adv": decomp_better_than_adv,
        "decomp_more_selective_than_adv": decomp_more_selective_than_adv,
        "clean_at_least_as_good_as_others": clean_at_least_as_good,
        "decision": decision,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    args = parser.parse_args()
    with args.data_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    eval_dir = out_dir / "evaluation"
    rng = np.random.default_rng(int(config["seed"]))

    per_model = {m: load_results(eval_dir, m) for m in MODELS}

    descriptive = {}
    g10_scores = {}
    g00_scores = {}
    for m in MODELS:
        rows = per_model[m]
        g10, g00, gt, gc = by_role(rows, "G10"), by_role(rows, "G00"), by_role(rows, "GT"), by_role(rows, "GC")
        g10_scores[m] = [r["s_score"] for r in g10]
        g00_scores[m] = [r["s_score"] for r in g00]

        coupling = bootstrap_unpaired_diff_ci(g10_scores[m], g00_scores[m], rng)
        g10_stats = bootstrap_mean_ci(g10_scores[m], rng)
        gt_acc = binomial_ci(sum(r["correct"] for r in gt), len(gt))
        gc_acc = binomial_ci(sum(r["correct"] for r in gc), len(gc))

        descriptive[m] = {
            "n_g10": len(g10), "n_g00": len(g00), "n_gt": len(gt), "n_gc": len(gc),
            "coupling_B": coupling, "g10_s_ball": g10_stats,
            "g10_yes_rate": float(np.mean([r["prediction"] == "Yes" for r in g10])),
            "g00_s_ball_mean": float(np.mean(g00_scores[m])),
            "ball_plus_acc": gt_acc, "bat_plus_acc": gc_acc,
        }

    paired = {
        "adv_decomp_minus_adv_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["adv_decomp_debias"], g10_scores["adv_debias"], rng),
        "adv_decomp_minus_clean_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["adv_decomp_debias"], g10_scores["clean_debias"], rng),
        "adv_decomp_minus_original_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["adv_decomp_debias"], g10_scores["original"], rng),
        "adv_minus_clean_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["adv_debias"], g10_scores["clean_debias"], rng),
    }

    B = {m: descriptive[m]["coupling_B"]["observed"] for m in MODELS}
    ball_acc = {m: descriptive[m]["ball_plus_acc"]["acc"] for m in MODELS}
    bat_acc = {m: descriptive[m]["bat_plus_acc"]["acc"] for m in MODELS}

    component_summary = load_component_selectivity_summary(out_dir)
    go_no_go = classify_go_no_go(component_summary, B, ball_acc, bat_acc)

    statistics = {
        "descriptive": descriptive,
        "paired_bootstrap_g10_s_ball": paired,
        "coupling_B": B,
        "ball_plus_acc": ball_acc,
        "bat_plus_acc": bat_acc,
        "spurious_reduction_R_S": {m: B["original"] - B[m] for m in MODELS},
        "target_degradation_D_T": {m: ball_acc["original"] - ball_acc[m] for m in MODELS},
        "context_degradation_D_C": {m: bat_acc["original"] - bat_acc[m] for m in MODELS},
        "component_selectivity_summary": component_summary,
        "go_no_go": go_no_go,
    }
    stats_path = eval_dir / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"wrote {stats_path}")

    summary_rows = []
    for m in MODELS:
        summary_rows.append({
            "method": m, "coupling_B": B[m], "g10_s_ball_mean": descriptive[m]["g10_s_ball"]["mean"],
            "ball_plus_acc": ball_acc[m], "bat_plus_acc": bat_acc[m],
            "delta_coupling_B_from_original": B[m] - B["original"],
            "delta_ball_plus_acc_from_original": ball_acc[m] - ball_acc["original"],
            "delta_bat_plus_acc_from_original": bat_acc[m] - bat_acc["original"],
        })
    summary_path = eval_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")

    print("\n=== Primary Result Table ===")
    print(f"{'Method':<18}{'Coupling B':>12}{'G10 s_ball':>12}{'Ball+ Acc':>11}{'Bat+ Acc':>10}")
    for r in summary_rows:
        print(f"{r['method']:<18}{r['coupling_B']:>12.4f}{r['g10_s_ball_mean']:>12.4f}{r['ball_plus_acc']:>11.3f}{r['bat_plus_acc']:>10.3f}")
    print(f"\nComponent selectivity summary: {component_summary}")
    print(f"\nGo/No-Go: {go_no_go['decision']}")


if __name__ == "__main__":
    main()
