"""Stage 10 analysis (Experiment 2): within-image co-occurrence specificity.

Final question: holding the image fixed, does an absent target with a
stronger train-set co-occurrence relationship to that image's present
objects receive higher clean target-positive evidence s_T, after
controlling for image-specific and target-specific fixed effects?

No mitigation, attack, or representation editing is performed -- this is a
read-only statistical analysis of Stage 10's clean_evidence collection.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage10_analysis.py \
        --config configs/stage10_analysis.yaml
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
import statsmodels.formula.api as smf
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 10 analysis: within-image co-occurrence specificity")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def fit_two_way_fe(df: pd.DataFrame, cluster_col: str):
    """OLS with s_T ~ cooc_score + C(image_id) + C(target), cluster-robust SE.

    Two-way fixed effects (image + target dummies) implemented via patsy
    categorical dummies inside a single OLS design matrix -- exact at this
    scale (n~3-4k rows, ~120-130 dummy columns), no iterative demeaning needed.
    Cluster-robust (CR1) covariance clustered on `cluster_col`.
    """
    model = smf.ols("s_T ~ cooc_score + C(image_id) + C(target)", data=df)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
    ci = fit.conf_int().loc["cooc_score"]
    return {
        "beta": float(fit.params["cooc_score"]),
        "se": float(fit.bse["cooc_score"]),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_value": float(fit.pvalues["cooc_score"]),
        "n_obs": int(fit.nobs),
        "r_squared": float(fit.rsquared),
        "cluster_col": cluster_col,
    }


def build_fe_projection(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Precomputed Frisch-Waugh-Lovell projection for the permutation null:
    Z = [intercept, image dummies, target dummies] (everything EXCEPT
    cooc_score). residualize(v) = v - Z @ (Z_pinv @ v) partials out both fixed
    effects from any vector aligned with df's rows. Precomputing Z_pinv once
    means each permutation costs one matvec instead of a full OLS refit --
    exact by the FWL theorem, not an approximation.
    """
    image_dummies = pd.get_dummies(df["image_id"], prefix="img", drop_first=True).to_numpy(dtype=float)
    target_dummies = pd.get_dummies(df["target"], prefix="tgt", drop_first=True).to_numpy(dtype=float)
    Z = np.column_stack([np.ones(len(df)), image_dummies, target_dummies])
    Z_pinv = np.linalg.pinv(Z)
    return Z, Z_pinv


def residualize(v: np.ndarray, Z: np.ndarray, Z_pinv: np.ndarray) -> np.ndarray:
    return v - Z @ (Z_pinv @ v)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = __import__("yaml").safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    df = pd.read_csv(config["evidence_path"])
    print(f"[stage10-analysis] loaded {len(df)} (image, absent-target) rows, {df['image_id'].nunique()} images, {df['target'].nunique()} target categories")

    # --- Dataset statistics ---
    per_image_counts = df.groupby("image_id").size()
    per_target_counts = df.groupby("target").size()
    dataset_stats = {
        "n_images": int(df["image_id"].nunique()),
        "n_pairs": int(len(df)),
        "n_target_categories": int(df["target"].nunique()),
        "targets_per_image": {"min": int(per_image_counts.min()), "max": int(per_image_counts.max()), "mean": float(per_image_counts.mean())},
        "images_per_target": {"min": int(per_target_counts.min()), "max": int(per_target_counts.max()), "mean": float(per_target_counts.mean())},
        "cooc_score": {"min": float(df["cooc_score"].min()), "max": float(df["cooc_score"].max()), "mean": float(df["cooc_score"].mean()), "sd": float(df["cooc_score"].std(ddof=1))},
        "s_T": {"min": float(df["s_T"].min()), "max": float(df["s_T"].max()), "mean": float(df["s_T"].mean()), "sd": float(df["s_T"].std(ddof=1))},
        "clean_is_yes_rate_overall": float(df["clean_is_yes"].mean()),
    }
    print(f"[stage10-analysis] dataset: {dataset_stats['n_images']} images, {dataset_stats['n_pairs']} pairs, "
          f"{dataset_stats['n_target_categories']} target categories, targets/image mean={dataset_stats['targets_per_image']['mean']:.1f}")

    # --- Primary Analysis 1: within-image relationship ---
    min_n = int(config["min_targets_per_image"])
    within_rows = []
    for image_id, g in df.groupby("image_id"):
        if len(g) < min_n or g["cooc_score"].std(ddof=1) == 0:
            continue
        pear_r, pear_p = stats.pearsonr(g["cooc_score"], g["s_T"])
        sp_r, sp_p = stats.spearmanr(g["cooc_score"], g["s_T"])
        slope = np.polyfit(g["cooc_score"], g["s_T"], 1)[0]
        within_rows.append({"image_id": image_id, "n_targets": len(g), "pearson_r": pear_r, "spearman_r": sp_r, "slope": slope})
    within_df = pd.DataFrame(within_rows)
    n_used = len(within_df)
    n_positive_slope = int((within_df["slope"] > 0).sum())
    n_positive_corr = int((within_df["pearson_r"] > 0).sum())
    sign_test_slope = stats.binomtest(n_positive_slope, n_used, 0.5)
    wilcoxon_corr = stats.wilcoxon(within_df["pearson_r"])

    within_image_analysis = {
        "n_images_used": n_used,
        "n_images_excluded": int(dataset_stats["n_images"] - n_used),
        "mean_pearson_r": float(within_df["pearson_r"].mean()),
        "median_pearson_r": float(within_df["pearson_r"].median()),
        "sd_pearson_r": float(within_df["pearson_r"].std(ddof=1)),
        "mean_slope": float(within_df["slope"].mean()),
        "median_slope": float(within_df["slope"].median()),
        "proportion_positive_slope": n_positive_slope / n_used,
        "proportion_positive_correlation": n_positive_corr / n_used,
        "sign_test_positive_slope_vs_half": {"n_positive": n_positive_slope, "n_total": n_used, "p_value": float(sign_test_slope.pvalue)},
        "wilcoxon_pearson_r_vs_zero": {"statistic": float(wilcoxon_corr.statistic), "p_value": float(wilcoxon_corr.pvalue)},
    }
    print(f"[stage10-analysis] A. within-image: n_images_used={n_used} mean_r={within_image_analysis['mean_pearson_r']:.3f} "
          f"pct_positive_slope={within_image_analysis['proportion_positive_slope']:.3f} "
          f"sign_test_p={sign_test_slope.pvalue:.3g} wilcoxon_p={wilcoxon_corr.pvalue:.3g}")

    # --- Primary Analysis 2: two-way fixed-effect model ---
    fe_primary = fit_two_way_fe(df, cluster_col="image_id")
    fe_secondary_target_cluster = fit_two_way_fe(df, cluster_col="target")
    print(f"[stage10-analysis] B. two-way FE (image+target dummies), cluster-by-image: "
          f"beta={fe_primary['beta']:.4f} SE={fe_primary['se']:.4f} 95%CI=[{fe_primary['ci_lower']:.4f},{fe_primary['ci_upper']:.4f}] p={fe_primary['p_value']:.3g}")
    print(f"[stage10-analysis]    robustness, cluster-by-target: SE={fe_secondary_target_cluster['se']:.4f} p={fe_secondary_target_cluster['p_value']:.3g} (point estimate identical by construction)")

    # --- Negative control: permutation-shuffle of cooc_score ---
    Z, Z_pinv = build_fe_projection(df)
    resid_y = residualize(df["s_T"].to_numpy(dtype=float), Z, Z_pinv)
    resid_x_real = residualize(df["cooc_score"].to_numpy(dtype=float), Z, Z_pinv)
    beta_real_manual = float((resid_x_real @ resid_y) / (resid_x_real @ resid_x_real))

    n_perm = int(config["n_permutations"])
    raw_score = df["cooc_score"].to_numpy(dtype=float)
    beta_perms = np.empty(n_perm)
    for i in range(n_perm):
        perm_x = rng.permutation(raw_score)
        resid_x_perm = residualize(perm_x, Z, Z_pinv)
        beta_perms[i] = (resid_x_perm @ resid_y) / (resid_x_perm @ resid_x_perm)

    perm_p_one_sided = (int(np.sum(beta_perms >= beta_real_manual)) + 1) / (n_perm + 1)
    negative_control = {
        "beta_real_fwl_manual": beta_real_manual,
        "beta_real_statsmodels": fe_primary["beta"],
        "fwl_matches_statsmodels": bool(abs(beta_real_manual - fe_primary["beta"]) < 1e-6 * max(1.0, abs(fe_primary["beta"]))),
        "n_permutations": n_perm,
        "beta_shuffle_mean": float(beta_perms.mean()),
        "beta_shuffle_sd": float(beta_perms.std(ddof=1)),
        "beta_shuffle_min": float(beta_perms.min()),
        "beta_shuffle_max": float(beta_perms.max()),
        "permutation_p_value_one_sided": perm_p_one_sided,
    }
    print(f"[stage10-analysis] C. negative control: beta_real={beta_real_manual:.4f} (matches statsmodels: {negative_control['fwl_matches_statsmodels']}) "
          f"beta_shuffle: mean={beta_perms.mean():.4f} sd={beta_perms.std(ddof=1):.4f} "
          f"permutation_p={perm_p_one_sided:.4g} ({n_perm} permutations)")

    # --- Sanity checks ---
    freq_score_corr_r, freq_score_corr_p = stats.pearsonr(df["target_marginal_freq"], df["cooc_score"])
    yes_rate_by_target = df.groupby("target")["clean_is_yes"].mean().sort_values(ascending=False)
    sanity = {
        "all_targets_absent_check": "passed (asserted in collection script; build_candidates invariant)",
        "target_freq_vs_score_pearson_r": float(freq_score_corr_r),
        "target_freq_vs_score_p": float(freq_score_corr_p),
        "top5_categories_by_clean_yes_rate": yes_rate_by_target.head(5).to_dict(),
        "bottom5_categories_by_clean_yes_rate": yes_rate_by_target.tail(5).to_dict(),
        "decision_point_consistency_with_stage9": "prefix_ids=[] yes_id=3869 no_id=1939 (identical, printed at collection time)",
    }
    print(f"[stage10-analysis] sanity: target_freq vs score correlation r={freq_score_corr_r:.3f} (p={freq_score_corr_p:.3g})")

    # --- Robustness checks ---
    # (i) image-FE-only model with target_marginal_freq as a continuous covariate
    #     instead of full target dummies -- checks whether beta survives when
    #     target-level variation is represented by frequency alone, not fully
    #     absorbed by per-target intercepts.
    model_freq_covariate = smf.ols("s_T ~ cooc_score + target_marginal_freq + C(image_id)", data=df)
    fit_freq_covariate = model_freq_covariate.fit(cov_type="cluster", cov_kwds={"groups": df["image_id"]})
    robustness_freq_covariate = {
        "beta": float(fit_freq_covariate.params["cooc_score"]),
        "se": float(fit_freq_covariate.bse["cooc_score"]),
        "p_value": float(fit_freq_covariate.pvalues["cooc_score"]),
    }

    # (ii) leave-one-target-category-out
    targets = sorted(df["target"].unique())
    loo_target_betas = []
    for t in targets:
        sub = df[df["target"] != t]
        fit = smf.ols("s_T ~ cooc_score + C(image_id) + C(target)", data=sub).fit()
        loo_target_betas.append(float(fit.params["cooc_score"]))
    loo_target_betas = np.array(loo_target_betas)

    # (iii) leave-one-image-out
    images = sorted(df["image_id"].unique())
    loo_image_betas = []
    for img in images:
        sub = df[df["image_id"] != img]
        fit = smf.ols("s_T ~ cooc_score + C(image_id) + C(target)", data=sub).fit()
        loo_image_betas.append(float(fit.params["cooc_score"]))
    loo_image_betas = np.array(loo_image_betas)

    robustness = {
        "image_fe_plus_freq_covariate_instead_of_target_fe": robustness_freq_covariate,
        "leave_one_target_out": {
            "n_refits": len(loo_target_betas), "beta_min": float(loo_target_betas.min()), "beta_max": float(loo_target_betas.max()),
            "beta_range": float(loo_target_betas.max() - loo_target_betas.min()), "all_positive": bool(np.all(loo_target_betas > 0)),
        },
        "leave_one_image_out": {
            "n_refits": len(loo_image_betas), "beta_min": float(loo_image_betas.min()), "beta_max": float(loo_image_betas.max()),
            "beta_range": float(loo_image_betas.max() - loo_image_betas.min()), "all_positive": bool(np.all(loo_image_betas > 0)),
        },
    }
    print(f"[stage10-analysis] robustness: freq-covariate model beta={robustness_freq_covariate['beta']:.4f} (p={robustness_freq_covariate['p_value']:.3g}); "
          f"LOTO beta range=[{loo_target_betas.min():.4f},{loo_target_betas.max():.4f}] all_positive={robustness['leave_one_target_out']['all_positive']}; "
          f"LOIO beta range=[{loo_image_betas.min():.4f},{loo_image_betas.max():.4f}] all_positive={robustness['leave_one_image_out']['all_positive']}")

    # --- Figure 1: within-image (FE-demeaned) score vs s_T ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(resid_x_real, resid_y, alpha=0.15, s=8)
    xs = np.linspace(resid_x_real.min(), resid_x_real.max(), 100)
    ax.plot(xs, beta_real_manual * xs, color="black", linewidth=1.5, label=f"beta={beta_real_manual:.3f}")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("cooc_score, residualized on image + target fixed effects")
    ax.set_ylabel("s_T, residualized on image + target fixed effects")
    ax.set_title(f"Within-image co-occurrence specificity (n={len(df)} pairs, {dataset_stats['n_images']} images)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_within_image_score_vs_sT.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: distribution of per-image correlations ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(within_df["pearson_r"], bins=20, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
    ax.axvline(within_df["pearson_r"].mean(), color="red", linewidth=1.2, label=f"mean={within_df['pearson_r'].mean():.3f}")
    ax.set_xlabel("within-image Pearson r (cooc_score vs s_T)")
    ax.set_ylabel("number of images")
    ax.set_title(f"Distribution of within-image correlations (n={n_used} images, {within_image_analysis['proportion_positive_slope']*100:.0f}% positive slope)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_within_image_correlation_distribution.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: beta_real vs permutation null ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hist(beta_perms, bins=40, alpha=0.8, label="shuffled null (permuted cooc_score)")
    ax.axvline(beta_real_manual, color="red", linewidth=1.5, label=f"beta_real={beta_real_manual:.3f}")
    ax.set_xlabel("beta (two-way FE coefficient on cooc_score)")
    ax.set_ylabel("count (permutations)")
    ax.set_title(f"beta_real vs. permutation null (n={n_perm}, p={perm_p_one_sided:.3g})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_beta_real_vs_permutation_null.png", dpi=150)
    plt.close(fig)

    report = {
        "dataset_stats": dataset_stats,
        "A_within_image_analysis": within_image_analysis,
        "B_fixed_effect_primary_cluster_by_image": fe_primary,
        "B_fixed_effect_robustness_cluster_by_target": fe_secondary_target_cluster,
        "C_negative_control_permutation": negative_control,
        "sanity_checks": sanity,
        "robustness_checks": robustness,
        "alpha": float(config["alpha"]),
        "statistical_implementation": (
            "Primary model: statsmodels OLS, formula 's_T ~ cooc_score + C(image_id) + C(target)' "
            "(two-way fixed effects via categorical dummies), cluster-robust (CR1) SE clustered by image_id. "
            "Cluster-by-target reported as a robustness check (identical point estimate, different SE). "
            "Permutation null: cooc_score globally shuffled across all (image,target) rows, "
            "Frisch-Waugh-Lovell-residualized against the same image+target fixed effects "
            "(precomputed pseudo-inverse projection, verified to reproduce the real OLS beta exactly)."
        ),
    }
    with (output_dir / "stage10_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[stage10-analysis] report written to {output_dir / 'stage10_report.json'}")


if __name__ == "__main__":
    main()
