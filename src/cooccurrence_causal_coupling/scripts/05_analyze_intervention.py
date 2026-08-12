"""Experiment 4, Step 5: analyze the full intervention scan (Step 4's output).

For each (layer, lambda) with the REAL direction: beta_before/beta_after
(Experiment 2's exact FE model, refit on the test split), Delta_beta, mean/
median Delta_sT with bootstrap CI. Controls 1-5 compared against the same
metric where applicable. No mitigation/attack/editing performed -- read-only
analysis of Step 4's saved CSVs.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python 05_analyze_intervention.py \
        --config ../configs/05_analyze_intervention.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cooccurrence_causal_coupling.common import build_fe_projection, residualize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def fit_beta(df: pd.DataFrame, outcome_col: str) -> dict:
    alias = df.rename(columns={outcome_col: "y"})
    fit = smf.ols("y ~ cooc_score + C(image_id) + C(target)", data=alias).fit(
        cov_type="cluster", cov_kwds={"groups": alias["image_id"]}
    )
    ci = fit.conf_int().loc["cooc_score"]
    return {
        "beta": float(fit.params["cooc_score"]),
        "se": float(fit.bse["cooc_score"]),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_value": float(fit.pvalues["cooc_score"]),
        "n": int(fit.nobs),
    }


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(x)
    boots = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    boots.sort()
    return float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot) - 1])


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    figures_dir = Path(config["figures_dir"])
    robustness_dir = Path(config["robustness_dir"])
    for d in (output_dir, figures_dir, robustness_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    main_df = pd.read_csv(config["main_results_path"])
    main_df["delta_sT"] = main_df["s_after"] - main_df["s_before"]
    layers = sorted(main_df["layer"].unique())
    lambdas = sorted(main_df["lambda"].unique())
    print(f"[analyze] main results: {len(main_df)} rows, layers={layers}, lambdas={lambdas}")

    # --- beta_before: identical across (layer, lambda) since s_before is the cached, unpatched baseline ---
    beta_before = fit_beta(main_df.drop_duplicates(subset=["image_id", "target"]).assign(s_before=lambda d: d["s_before"]), "s_before")
    print(f"[analyze] beta_before (test split, n={beta_before['n']}): {beta_before['beta']:.4f} "
          f"SE={beta_before['se']:.4f} p={beta_before['p_value']:.3g}")

    per_cell_rows = []
    for L in layers:
        for lam in lambdas:
            cell = main_df[(main_df["layer"] == L) & (main_df["lambda"] == lam)]
            beta_after = fit_beta(cell, "s_after")
            delta_beta = beta_after["beta"] - beta_before["beta"]
            delta_sT = cell["delta_sT"].to_numpy()
            ci_lo, ci_hi = bootstrap_mean_ci(delta_sT, int(config["n_bootstrap"]), rng)
            per_cell_rows.append(
                {
                    "layer": int(L), "lambda": float(lam),
                    "beta_before": beta_before["beta"], "beta_after": beta_after["beta"],
                    "beta_after_se": beta_after["se"], "beta_after_p": beta_after["p_value"],
                    "delta_beta": delta_beta,
                    "mean_delta_sT": float(delta_sT.mean()), "median_delta_sT": float(np.median(delta_sT)),
                    "delta_sT_ci_lower": ci_lo, "delta_sT_ci_upper": ci_hi,
                    "n": len(cell),
                }
            )
            print(f"[analyze] L={L:2d} lambda={lam:.2f} beta_after={beta_after['beta']:+.4f} "
                  f"delta_beta={delta_beta:+.4f} mean_delta_sT={delta_sT.mean():+.4f}")

    per_cell_df = pd.DataFrame(per_cell_rows)
    per_cell_df.to_csv(output_dir / "beta_after_intervention.csv", index=False)

    # --- Control 2: low vs high co-occurrence (median split within test), at lambda=1.0 ---
    lam1 = float(config["lambda_control"])
    at_lam1 = main_df[main_df["lambda"] == lam1]
    control2_rows = []
    for L in layers:
        sub = at_lam1[at_lam1["layer"] == L]
        median_score = sub["cooc_score"].median()
        high = sub[sub["cooc_score"] >= median_score]["delta_sT"]
        low = sub[sub["cooc_score"] < median_score]["delta_sT"]
        control2_rows.append({"layer": int(L), "mean_delta_high_cooc": float(high.mean()), "mean_delta_low_cooc": float(low.mean()),
                               "n_high": len(high), "n_low": len(low), "selective": abs(high.mean()) > abs(low.mean())})
    control2_df = pd.DataFrame(control2_rows)
    control2_df.to_csv(Path(config["shuffled_direction_path"]).parent / "low_cooc_results.csv", index=False)
    print("[analyze] control2 (low vs high cooc):")
    print(control2_df.to_string(index=False))

    # --- Controls 3 & 4: random / shuffled direction Delta_beta at lambda_control ---
    random_df = pd.read_csv(config["random_direction_path"])
    random_df["delta_sT"] = random_df["s_after"] - random_df["s_before"]
    shuffled_df = pd.read_csv(config["shuffled_direction_path"])
    shuffled_df["delta_sT"] = shuffled_df["s_after"] - shuffled_df["s_before"]

    control_comparison_rows = []
    for L in layers:
        real_cell = per_cell_df[(per_cell_df["layer"] == L) & (per_cell_df["lambda"] == lam1)].iloc[0]
        rand_betas = []
        for seed_i in sorted(random_df["seed"].unique()):
            sub = random_df[(random_df["layer"] == L) & (random_df["seed"] == seed_i)]
            beta_r = fit_beta(sub, "s_after")
            rand_betas.append(beta_r["beta"] - beta_before["beta"])
        shuf_sub = shuffled_df[shuffled_df["layer"] == L]
        beta_shuf = fit_beta(shuf_sub, "s_after")
        delta_beta_shuf = beta_shuf["beta"] - beta_before["beta"]

        control_comparison_rows.append(
            {
                "layer": int(L), "delta_beta_real": float(real_cell["delta_beta"]),
                "delta_beta_random_mean": float(np.mean(rand_betas)), "delta_beta_random_sd": float(np.std(rand_betas, ddof=1)),
                "delta_beta_random_all": rand_betas,
                "delta_beta_shuffled": delta_beta_shuf,
                "real_more_negative_than_random": bool(real_cell["delta_beta"] < np.mean(rand_betas) - np.std(rand_betas, ddof=1)),
                "real_more_negative_than_shuffled": bool(real_cell["delta_beta"] < delta_beta_shuf),
            }
        )
    control_df = pd.DataFrame(control_comparison_rows)
    control_df.to_csv(robustness_dir / "cluster_target_results.csv", index=False)  # (naming kept per spec's suggested output tree)
    print("[analyze] controls 3&4 (random/shuffled) vs real, at lambda=1.0:")
    print(control_df[["layer", "delta_beta_real", "delta_beta_random_mean", "delta_beta_random_sd", "delta_beta_shuffled"]].to_string(index=False))

    # --- Control 1: genuine target preservation ---
    genuine_df = pd.read_csv(config["genuine_target_path"])
    genuine_df["delta_sT"] = genuine_df["s_after"] - genuine_df["s_before"]
    genuine_summary = genuine_df.groupby(["layer", "lambda"])["delta_sT"].mean().reset_index()
    genuine_summary.to_csv(Path(config["genuine_target_path"]).parent / "genuine_target_summary.csv", index=False)
    print("[analyze] control1 (genuine target) mean delta_sT by layer/lambda:")
    print(genuine_summary.to_string(index=False))

    # --- Control 5: general stability ---
    stability_df = pd.read_csv(config["stability_path"])
    stability_df["entropy_delta"] = stability_df["entropy_after"] - stability_df["entropy_before"]
    stability_df["caption_identical"] = stability_df["caption_before"] == stability_df["caption_after"]
    stability_summary = stability_df.groupby("layer").agg(
        pct_identical_caption=("caption_identical", "mean"), mean_entropy_delta=("entropy_delta", "mean")
    ).reset_index()
    print("[analyze] control5 (general stability):")
    print(stability_summary.to_string(index=False))

    # --- Figure 1: Delta_beta by layer (at lambda=1.0, real vs random vs shuffled) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(control_df["layer"], control_df["delta_beta_real"], marker="o", label="real direction", color="C0")
    ax.plot(control_df["layer"], control_df["delta_beta_shuffled"], marker="s", label="shuffled direction", color="C1")
    ax.errorbar(control_df["layer"], control_df["delta_beta_random_mean"], yerr=control_df["delta_beta_random_sd"],
                marker="^", label="random direction (mean +/- SD, 5 seeds)", color="C2", capsize=4)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xlabel("layer (hidden_states index)")
    ax.set_ylabel("Delta_beta = beta_after - beta_before")
    ax.set_title(f"Causal effect on beta by layer (lambda={lam1})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_delta_beta_by_layer.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: beta_before vs beta_after per layer (lambda=1.0) ---
    fig, ax = plt.subplots(figsize=(8, 6))
    cell1 = per_cell_df[per_cell_df["lambda"] == lam1]
    x = np.arange(len(cell1))
    width = 0.35
    ax.bar(x - width / 2, [beta_before["beta"]] * len(cell1), width, label="beta_before", color="C3")
    ax.bar(x + width / 2, cell1["beta_after"], width, label="beta_after", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in cell1["layer"]])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("beta (co-occurrence FE coefficient, test split)")
    ax.set_title(f"beta_before vs beta_after (lambda={lam1})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_beta_before_vs_after.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: selectivity (high-cooc absent vs low-cooc absent vs genuine target) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    x_labels, values, colors = [], [], []
    for L in layers:
        row = control2_df[control2_df["layer"] == L].iloc[0]
        g = genuine_summary[(genuine_summary["layer"] == L) & (genuine_summary["lambda"] == lam1)]
        g_val = float(g["delta_sT"].iloc[0]) if len(g) else np.nan
        x_labels += [f"L{L}\nhigh-cooc\nabsent", f"L{L}\nlow-cooc\nabsent", f"L{L}\ngenuine\npresent"]
        values += [row["mean_delta_high_cooc"], row["mean_delta_low_cooc"], g_val]
        colors += ["C0", "C1", "C2"]
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean Delta s_T")
    ax.set_title(f"Selectivity: high-cooc absent vs low-cooc absent vs genuine present (lambda={lam1})")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_selectivity.png", dpi=150)
    plt.close(fig)

    # --- Figure 5: Stage11 partial_r vs causal Delta_beta, aligned panels ---
    with open(config["stage11_report_path"], "r", encoding="utf-8") as f:
        stage11 = json.load(f)
    s11_layers = [r["layer"] for r in stage11["per_layer"]]
    s11_partial_r = [r["partial_r"] for r in stage11["per_layer"]]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(s11_layers, s11_partial_r, color="C4")
    axes[0].axhline(0, color="gray", linewidth=0.6, linestyle="--")
    axes[0].set_ylabel("Stage11 partial r\n(logit-lens, correlational)")
    axes[0].set_title("Representational localization (top) vs causal intervention (bottom)")
    axes[1].plot(control_df["layer"], control_df["delta_beta_real"], marker="o", color="C0")
    axes[1].axhline(0, color="gray", linewidth=0.6, linestyle="--")
    axes[1].set_ylabel("Delta_beta (causal, lambda=1.0)")
    axes[1].set_xlabel("layer")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_localization_vs_causal.png", dpi=150)
    plt.close(fig)

    report = {
        "beta_before": beta_before,
        "per_cell": per_cell_rows,
        "control1_genuine_target": genuine_summary.to_dict(orient="records"),
        "control2_low_vs_high_cooc": control2_df.to_dict(orient="records"),
        "control3_random_and_control4_shuffled": control_df.drop(columns=["delta_beta_random_all"]).to_dict(orient="records"),
        "control5_general_stability": stability_summary.to_dict(orient="records"),
    }
    with (output_dir / "experiment4_full_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[analyze] wrote {output_dir / 'experiment4_full_report.json'}")


if __name__ == "__main__":
    main()
