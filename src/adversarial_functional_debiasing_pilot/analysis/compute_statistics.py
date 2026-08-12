"""Reads evaluation/{original,clean_debias,adv_debias}_results.csv (already
produced by scripts/evaluate_model.py on the UNSEEN CLEAN TEST SPLIT) and
computes the primary result table + simple bootstrap statistics. No model
inference happens here -- this script only aggregates saved CSVs.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python analysis/compute_statistics.py --config configs/pilot.yaml
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml

MODELS = ["original", "clean_debias", "adv_debias"]
N_BOOT = 5000


def load_results(eval_dir: Path, model: str) -> list[dict]:
    with (eval_dir / f"{model}_results.csv").open("r", encoding="utf-8") as f:
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
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "ci_lower": float(np.percentile(boots, 2.5)),
        "ci_upper": float(np.percentile(boots, 97.5)),
        "n": len(arr),
    }


def bootstrap_unpaired_diff_ci(a: list[float], b: list[float], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """B = mean(a) - mean(b), a and b are different (non-overlapping) image sets."""
    arr_a, arr_b = np.array(a, dtype=float), np.array(b, dtype=float)
    boots_a = rng.choice(arr_a, size=(n_boot, len(arr_a)), replace=True).mean(axis=1)
    boots_b = rng.choice(arr_b, size=(n_boot, len(arr_b)), replace=True).mean(axis=1)
    diffs = boots_a - boots_b
    observed = float(arr_a.mean() - arr_b.mean())
    return {
        "observed": observed,
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }


def bootstrap_paired_diff_ci(a: list[float], b: list[float], rng: np.random.Generator, n_boot: int = N_BOOT) -> dict:
    """a, b indexed by the SAME (same-order) images -- e.g. G10 test s_ball for
    two different models on identical test images. Resample image indices jointly."""
    arr_a, arr_b = np.array(a, dtype=float), np.array(b, dtype=float)
    n = len(arr_a)
    assert n == len(arr_b)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)
    observed = float(arr_a.mean() - arr_b.mean())
    return {
        "observed": observed,
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }


def binomial_ci(n_correct: int, n_total: int) -> dict:
    from scipy.stats import beta

    if n_total == 0:
        return {"acc": None, "ci_lower": None, "ci_upper": None, "n": 0}
    alpha = 0.05
    lo = 0.0 if n_correct == 0 else beta.ppf(alpha / 2, n_correct, n_total - n_correct + 1)
    hi = 1.0 if n_correct == n_total else beta.ppf(1 - alpha / 2, n_correct + 1, n_total - n_correct)
    return {"acc": n_correct / n_total, "ci_lower": float(lo), "ci_upper": float(hi), "n": n_total}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
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
        g10 = by_role(rows, "G10")
        g00 = by_role(rows, "G00")
        gt = by_role(rows, "GT")
        gc = by_role(rows, "GC")

        g10_scores[m] = [r["s_score"] for r in g10]
        g00_scores[m] = [r["s_score"] for r in g00]

        coupling = bootstrap_unpaired_diff_ci(g10_scores[m], g00_scores[m], rng)
        g10_stats = bootstrap_mean_ci(g10_scores[m], rng)
        gt_acc = binomial_ci(sum(r["correct"] for r in gt), len(gt))
        gc_acc = binomial_ci(sum(r["correct"] for r in gc), len(gc))

        descriptive[m] = {
            "n_g10": len(g10),
            "n_g00": len(g00),
            "n_gt": len(gt),
            "n_gc": len(gc),
            "coupling_B": coupling,
            "g10_s_ball": g10_stats,
            "g10_yes_rate": float(np.mean([r["prediction"] == "Yes" for r in g10])),
            "g00_s_ball_mean": float(np.mean(g00_scores[m])),
            "ball_plus_acc": gt_acc,
            "bat_plus_acc": gc_acc,
        }

    # Paired comparisons on G10 test s_ball (identical 20 test images across all 3 models)
    paired = {
        "adv_minus_clean_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["adv_debias"], g10_scores["clean_debias"], rng),
        "adv_minus_original_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["adv_debias"], g10_scores["original"], rng),
        "clean_minus_original_g10_s_ball": bootstrap_paired_diff_ci(g10_scores["clean_debias"], g10_scores["original"], rng),
    }

    B = {m: descriptive[m]["coupling_B"]["observed"] for m in MODELS}
    ball_acc = {m: descriptive[m]["ball_plus_acc"]["acc"] for m in MODELS}
    bat_acc = {m: descriptive[m]["bat_plus_acc"]["acc"] for m in MODELS}

    # --- Go/No-Go classification (see task spec's decision tree) ---
    order_go = B["adv_debias"] < B["clean_debias"] < B["original"]
    adv_reduces_vs_original = B["adv_debias"] < B["original"]
    adv_better_than_clean = B["adv_debias"] < B["clean_debias"]

    retention_drop_threshold = 0.10  # 10 pp accuracy drop -- descriptive threshold, not tuned post hoc
    ball_drop_adv = ball_acc["original"] - ball_acc["adv_debias"]
    bat_drop_adv = bat_acc["original"] - bat_acc["adv_debias"]
    non_selective = adv_reduces_vs_original and (ball_drop_adv > retention_drop_threshold or bat_drop_adv > retention_drop_threshold)

    if non_selective:
        decision = "NO-GO A: non-selective (Adv Debias reduces coupling but degrades genuine Ball/Bat retention)"
    elif not adv_reduces_vs_original:
        decision = "NO-GO B/C: no generalized reduction (Adv Debias does not reduce clean-test coupling vs Original)"
    elif order_go:
        decision = "GO: B_Adv < B_Clean < B_Original, with retention preserved"
    elif adv_better_than_clean:
        decision = "WEAK GO: Adv Debias beats Original and Clean Debias on coupling, but ordering is not the strict GO pattern"
    else:
        decision = "NO-GO C: no advantage (Clean Debias performs as well as or better than Adv Debias)"

    statistics = {
        "descriptive": descriptive,
        "paired_bootstrap_g10_s_ball": paired,
        "coupling_B": B,
        "ball_plus_acc": ball_acc,
        "bat_plus_acc": bat_acc,
        "spurious_reduction_R_S": {m: B["original"] - B[m] for m in MODELS},
        "target_degradation_D_T": {m: ball_acc["original"] - ball_acc[m] for m in MODELS},
        "context_degradation_D_C": {m: bat_acc["original"] - bat_acc[m] for m in MODELS},
        "go_no_go": {
            "order_B_Adv_lt_Clean_lt_Original": order_go,
            "adv_reduces_coupling_vs_original": adv_reduces_vs_original,
            "adv_better_than_clean": adv_better_than_clean,
            "ball_acc_drop_adv_debias": ball_drop_adv,
            "bat_acc_drop_adv_debias": bat_drop_adv,
            "retention_drop_threshold": retention_drop_threshold,
            "decision": decision,
        },
    }

    stats_path = eval_dir / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"wrote {stats_path}")

    summary_rows = []
    for m in MODELS:
        summary_rows.append(
            {
                "method": m,
                "coupling_B": B[m],
                "g10_s_ball_mean": descriptive[m]["g10_s_ball"]["mean"],
                "ball_plus_acc": ball_acc[m],
                "bat_plus_acc": bat_acc[m],
                "delta_coupling_B_from_original": B[m] - B["original"],
                "delta_ball_plus_acc_from_original": ball_acc[m] - ball_acc["original"],
                "delta_bat_plus_acc_from_original": bat_acc[m] - bat_acc["original"],
            }
        )
    summary_path = eval_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {summary_path}")

    print("\n=== Primary Result Table ===")
    print(f"{'Method':<14}{'Coupling B':>12}{'G10 s_ball':>12}{'Ball+ Acc':>11}{'Bat+ Acc':>10}")
    for r in summary_rows:
        print(f"{r['method']:<14}{r['coupling_B']:>12.4f}{r['g10_s_ball_mean']:>12.4f}{r['ball_plus_acc']:>11.3f}{r['bat_plus_acc']:>10.3f}")
    print(f"\nGo/No-Go: {decision}")


if __name__ == "__main__":
    main()
