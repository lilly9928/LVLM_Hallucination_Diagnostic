"""Stage 9 analysis (Experiment 1): does high train-set co-occurrence already
raise clean target-positive evidence s_T = logit(Yes) - logit(No), before any
attack, for a target object that is genuinely absent from the image?

Joins Stage 9's clean_evidence.csv (this experiment) with Stage 3's
epsilon_star_results.csv (existing) on (pair_id, arm) to additionally test
s_T's relationship with epsilon*. No mitigation, no representation editing,
no attack is performed here -- this is a read-only statistical analysis of
already-collected clean-image evidence and Stage 3's existing epsilon* results.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage9_analysis.py \
        --config configs/stage9_analysis.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 9 analysis: clean target-positive evidence s_T")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def cohens_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def bootstrap_mean_diff_ci(diff: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(diff)
    boots = np.array([np.mean(diff[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    boots.sort()
    return float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot) - 1])


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    evidence = pd.read_csv(config["clean_evidence_path"])
    eps = pd.read_csv(config["epsilon_star_results_path"])
    print(f"[stage9-analysis] loaded {len(evidence)} clean-evidence rows, {len(eps)} epsilon* rows")

    df = evidence.merge(eps[["pair_id", "arm", "status", "epsilon_star"]], on=["pair_id", "arm"], how="left")
    n_missing_eps = int(df["epsilon_star"].isna().sum())
    print(f"[stage9-analysis] merged; rows missing an epsilon* match: {n_missing_eps}")

    label_of = {"treatment": "high_cooccurrence", "control": "low_cooccurrence"}
    df["group"] = df["arm"].map(label_of)

    # --- Descriptive: High vs Low s_T ---
    descriptive = {}
    for arm in ["treatment", "control"]:
        sub = df[df["arm"] == arm]["s_T"]
        descriptive[label_of[arm]] = {
            "n": int(len(sub)),
            "mean": float(sub.mean()),
            "median": float(sub.median()),
            "sd": float(sub.std(ddof=1)),
            "clean_is_yes_rate": float(df[df["arm"] == arm]["clean_is_yes"].mean()),
        }
    print("[stage9-analysis] descriptive s_T by group:")
    for g, d in descriptive.items():
        print(f"  {g}: n={d['n']} mean={d['mean']:.3f} median={d['median']:.3f} sd={d['sd']:.3f} clean_is_yes_rate={d['clean_is_yes_rate']:.3f}")

    # --- A. Paired High vs Low comparison (matched-pair structure preserved) ---
    wide = df.pivot(index="pair_id", columns="arm", values="s_T")
    wide = wide.dropna(subset=["treatment", "control"])
    diff = (wide["treatment"] - wide["control"]).to_numpy()
    n_pairs = len(diff)

    wilcoxon_result = stats.wilcoxon(diff)
    ttest_result = stats.ttest_rel(wide["treatment"], wide["control"])
    dz = cohens_dz(diff)
    ci_lo, ci_hi = bootstrap_mean_diff_ci(diff, int(config["n_bootstrap"]), rng)

    paired_analysis = {
        "n_pairs": int(n_pairs),
        "mean_diff_treatment_minus_control": float(np.mean(diff)),
        "median_diff_treatment_minus_control": float(np.median(diff)),
        "bootstrap_ci_mean_diff": [ci_lo, ci_hi],
        "wilcoxon_signed_rank": {"statistic": float(wilcoxon_result.statistic), "p_value": float(wilcoxon_result.pvalue)},
        "paired_ttest": {"statistic": float(ttest_result.statistic), "p_value": float(ttest_result.pvalue)},
        "cohens_dz": dz,
    }
    print(f"[stage9-analysis] A. paired s_T (treatment-control): mean_diff={paired_analysis['mean_diff_treatment_minus_control']:.3f} "
          f"wilcoxon_p={wilcoxon_result.pvalue:.3g} paired_t_p={ttest_result.pvalue:.3g} dz={dz:.3f}")

    # --- B. Continuous relationship: co-occurrence score vs s_T (pooled, n=300) ---
    pearson_r, pearson_p = stats.pearsonr(df["cooc_score"], df["s_T"])
    spearman_r, spearman_p = stats.spearmanr(df["cooc_score"], df["s_T"])
    slope, intercept = np.polyfit(df["cooc_score"], df["s_T"], 1)
    continuous_relationship = {
        "n": int(len(df)),
        "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "ols_slope": float(slope), "ols_intercept": float(intercept),
    }
    print(f"[stage9-analysis] B. cooc_score vs s_T (pooled n={len(df)}): pearson_r={pearson_r:.3f} (p={pearson_p:.3g}) "
          f"spearman_r={spearman_r:.3f} (p={spearman_p:.3g})")

    # --- C. Relation with epsilon* (existing Stage 3 results; 0 censored samples in this data) ---
    eps_df = df.dropna(subset=["epsilon_star"])
    n_censored = int((eps_df["status"] == "censored").sum())
    sp_all_r, sp_all_p = stats.spearmanr(eps_df["s_T"], eps_df["epsilon_star"])

    flipped_only = eps_df[eps_df["status"] == "flipped"]
    sp_flip_r, sp_flip_p = stats.spearmanr(flipped_only["s_T"], flipped_only["epsilon_star"])

    epsilon_relationship = {
        "n_total_with_epsilon": int(len(eps_df)),
        "n_censored": n_censored,
        "n_already_yes": int((eps_df["status"] == "already_yes").sum()),
        "n_flipped": int(len(flipped_only)),
        "spearman_all_including_already_yes": {"r": float(sp_all_r), "p": float(sp_all_p)},
        "spearman_flipped_only_excluding_already_yes": {"r": float(sp_flip_r), "p": float(sp_flip_p)},
        "note": "already_yes rows have epsilon_star=0 by definition (model already says Yes at eps=0), "
                "so 'all' includes a mechanical component; 'flipped_only' tests the relationship among "
                "samples that did NOT already hallucinate at eps=0.",
    }
    print(f"[stage9-analysis] C. s_T vs epsilon* (n={len(eps_df)}, censored={n_censored}): "
          f"spearman_all_r={sp_all_r:.3f} (p={sp_all_p:.3g}) spearman_flipped_only_r={sp_flip_r:.3f} (p={sp_flip_p:.3g})")

    # --- Figure 1: High vs Low s_T, paired ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot([wide["control"], wide["treatment"]], labels=["Low co-occurrence", "High co-occurrence"], widths=0.5)
    for c, t in zip(wide["control"], wide["treatment"]):
        ax.plot([1, 2], [c, t], color="gray", alpha=0.15, linewidth=0.7, zorder=1)
    ax.scatter(np.full(len(wide), 1), wide["control"], alpha=0.4, s=12, zorder=2)
    ax.scatter(np.full(len(wide), 2), wide["treatment"], alpha=0.4, s=12, zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("s_T = logit(Yes) - logit(No)  [clean image]")
    ax.set_title(f"Clean target-positive evidence: High vs Low co-occurrence (n={n_pairs} pairs)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_high_vs_low_sT.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: co-occurrence score vs s_T ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for arm, marker, label in [("control", "o", "Low co-occurrence"), ("treatment", "^", "High co-occurrence")]:
        sub = df[df["arm"] == arm]
        ax.scatter(sub["cooc_score"], sub["s_T"], alpha=0.4, s=14, marker=marker, label=label)
    xs = np.linspace(df["cooc_score"].min(), df["cooc_score"].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.2, label=f"OLS fit (slope={slope:.2f})")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("co-occurrence score S(T, Y)  [mean PMI(T, y) over present objects y]")
    ax.set_ylabel("s_T = logit(Yes) - logit(No)  [clean image]")
    ax.set_title(f"Co-occurrence score vs. clean target-positive evidence (n={len(df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_score_vs_sT.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: s_T vs epsilon* ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(flipped_only["s_T"], flipped_only["epsilon_star"], alpha=0.4, s=14, label="flipped")
    already_yes = eps_df[eps_df["status"] == "already_yes"]
    ax.scatter(already_yes["s_T"], already_yes["epsilon_star"], alpha=0.4, s=14, color="red", marker="x", label="already_yes (eps*=0)")
    ax.set_xlabel("s_T = logit(Yes) - logit(No)  [clean image]")
    ax.set_ylabel("epsilon* (Stage 3, L-inf budget)")
    ax.set_title("Clean target-positive evidence vs. adversarial vulnerability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_sT_vs_epsilon_star.png", dpi=150)
    plt.close(fig)

    report = {
        "n_samples": int(len(df)),
        "n_pairs": int(n_pairs),
        "descriptive_sT_by_group": descriptive,
        "A_paired_high_vs_low": paired_analysis,
        "B_continuous_score_vs_sT": continuous_relationship,
        "C_sT_vs_epsilon_star": epsilon_relationship,
        "alpha": float(config["alpha"]),
    }
    with (output_dir / "stage9_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[stage9-analysis] report written to {output_dir / 'stage9_report.json'}")


if __name__ == "__main__":
    main()
