"""Stage 4: survival analysis on Stage 3's epsilon* results.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage4_survival_analysis.py \
        --config configs/stage4_survival_analysis.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.survival_analysis import (
    build_survival_frame,
    fit_km_by_arm,
    holm_correction,
    mcnemar_already_yes_test,
    paired_bootstrap_median_diff,
    pooled_logrank_test,
    stratified_cox_test,
    weibull_aft_time_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: survival analysis")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilon_max = float(config["epsilon_max"])

    with Path(config["epsilon_star_results_path"]).open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[stage4] loaded {len(rows)} samples from {config['epsilon_star_results_path']}")

    df = build_survival_frame(rows, epsilon_max=epsilon_max)

    # --- Descriptive summary ---
    descriptive = {}
    for arm in ["treatment", "control"]:
        sub = df[df["arm"] == arm]
        descriptive[arm] = {
            "n": int(len(sub)),
            "n_already_yes": int(sub["already_yes"].sum()),
            "already_yes_rate": float(sub["already_yes"].mean()),
            "n_censored": int((sub["event_observed"] == 0).sum()),
            "censored_rate": float((sub["event_observed"] == 0).mean()),
        }
    print("[stage4] descriptive summary:")
    for arm, d in descriptive.items():
        print(f"  {arm}: n={d['n']} already_yes={d['n_already_yes']} ({100*d['already_yes_rate']:.1f}%) censored={d['n_censored']} ({100*d['censored_rate']:.1f}%)")

    # --- Kaplan-Meier ---
    km_fits = fit_km_by_arm(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    for arm, kmf in km_fits.items():
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("epsilon (L-inf budget, [0,1] pixel units)")
    ax.set_ylabel("P(epsilon* > x)  (has not yet flipped)")
    ax.set_title("Kaplan-Meier: targeted-attack budget to flip absent-object hallucination")
    fig.tight_layout()
    fig.savefig(output_dir / "km_curves.png", dpi=150)
    plt.close(fig)

    km_medians = {arm: (float(kmf.median_survival_time_) if np.isfinite(kmf.median_survival_time_) else None) for arm, kmf in km_fits.items()}
    print(f"[stage4] KM median epsilon*: {km_medians}")

    # --- Primary test: stratified Cox (respects matched-pair design) ---
    cox_result = stratified_cox_test(df)
    print(f"[stage4] PRIMARY stratified Cox test: HR={cox_result['hazard_ratio']:.4f} "
          f"95%CI=[{cox_result['ci_lower']:.4f}, {cox_result['ci_upper']:.4f}] p={cox_result['p_value']:.6g}")

    # --- Sensitivity check: naive pooled log-rank (ignores pairing) ---
    logrank_result = pooled_logrank_test(df)
    print(f"[stage4] sensitivity check, pooled (unstratified) log-rank: "
          f"stat={logrank_result['test_statistic']:.4f} p={logrank_result['p_value']:.6g}")

    # --- Secondary effect size: Weibull AFT time ratio ---
    aft_result = weibull_aft_time_ratio(df, epsilon_floor=float(config["epsilon_floor"]))
    print(f"[stage4] Weibull AFT time ratio (treatment/control epsilon* scale): "
          f"{aft_result['time_ratio']:.4f} 95%CI=[{aft_result['ci_lower']:.4f}, {aft_result['ci_upper']:.4f}] p={aft_result['p_value']:.6g}")

    # --- Descriptive effect size: paired bootstrap median difference ---
    rng = np.random.default_rng(int(config["seed"]))
    bootstrap_result = paired_bootstrap_median_diff(df, n_boot=int(config["n_bootstrap"]), rng=rng)
    print(f"[stage4] paired bootstrap median(treatment epsilon*) - median(control epsilon*): "
          f"{bootstrap_result['observed_median_diff']:.5f} 95%CI=[{bootstrap_result['ci_lower']:.5f}, {bootstrap_result['ci_upper']:.5f}] "
          f"(n_pairs_used={bootstrap_result['n_pairs_used']}, excluded_censored={bootstrap_result['n_pairs_excluded_censored']})")

    # --- Secondary test: paired already_yes rate (McNemar) ---
    mcnemar_result = mcnemar_already_yes_test(df)
    print(f"[stage4] secondary McNemar test (already_yes rate): "
          f"treatment_only={mcnemar_result['treatment_only_already_yes']} control_only={mcnemar_result['control_only_already_yes']} "
          f"p={mcnemar_result['p_value']:.6g}")

    # --- Multiple comparison correction across the two distinct hypotheses ---
    raw_pvalues = {"cox_epsilon_star": cox_result["p_value"], "mcnemar_already_yes": mcnemar_result["p_value"]}
    adjusted_pvalues = holm_correction(raw_pvalues)
    alpha = float(config["alpha"])
    print(f"[stage4] Holm-corrected p-values (alpha={alpha}):")
    for name, p in adjusted_pvalues.items():
        print(f"  {name}: raw_p={raw_pvalues[name]:.6g} holm_p={p:.6g} significant={p < alpha}")

    report = {
        "descriptive": descriptive,
        "km_median_epsilon_star": km_medians,
        "primary_stratified_cox": cox_result,
        "sensitivity_pooled_logrank": logrank_result,
        "secondary_weibull_aft_time_ratio": aft_result,
        "descriptive_paired_bootstrap_median_diff": bootstrap_result,
        "secondary_mcnemar_already_yes": mcnemar_result,
        "holm_corrected_pvalues": adjusted_pvalues,
        "alpha": alpha,
    }
    with (output_dir / "stage4_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[stage4] full report written to {output_dir / 'stage4_report.json'}")


if __name__ == "__main__":
    main()
