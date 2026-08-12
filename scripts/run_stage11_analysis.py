"""Stage 11 analysis (Experiment 3): at which LLM layer does the within-image
co-occurrence-specificity effect (Experiment 2's beta) emerge?

For every LLM decoder layer, refits Experiment 2's exact two-way fixed-effect
model (s_T_l ~ cooc_score + C(image_id) + C(target)) on that layer's
logit-lens readout, with a cluster-robust real fit (statsmodels) plus a
permutation-shuffle null computed ONCE (image+target fixed effects and the
cooc_score permutations are layer-independent) and reused across all layers.

No mitigation, attack, or representation editing -- read-only analysis of
Stage 11's logit-lens collection.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_analysis.py \
        --config configs/stage11_analysis.yaml
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
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 analysis: layer-wise localization")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def build_fe_projection(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    long_df = pd.read_csv(config["layerwise_evidence_path"])
    layers = sorted(long_df["layer"].unique())
    print(f"[stage11-analysis] loaded {len(long_df)} (row x layer) records, {len(layers)} layers: {layers}")

    wide = long_df.pivot(index=["image_id", "target", "cooc_score"], columns="layer", values="s_T_l").reset_index()
    n = len(wide)
    print(f"[stage11-analysis] wide frame: {n} (image, target) pairs x {len(layers)} layer columns")

    # --- One-time FE projection setup (layer-independent: same image+target design) ---
    Z, Z_pinv = build_fe_projection(wide)
    raw_score = wide["cooc_score"].to_numpy(dtype=float)
    resid_x_real = residualize(raw_score, Z, Z_pinv)

    n_perm = int(config["n_permutations"])
    resid_x_perms = np.empty((n_perm, n))
    for i in range(n_perm):
        resid_x_perms[i] = residualize(rng.permutation(raw_score), Z, Z_pinv)
    perm_denom = np.sum(resid_x_perms**2, axis=1)  # (n_perm,)
    real_denom = float(resid_x_real @ resid_x_real)

    # --- Per-layer real fit (statsmodels, cluster-robust by image) + permutation p-value (manual FWL, reused null) ---
    per_layer_rows = []
    for layer in layers:
        y = wide[layer].to_numpy(dtype=float)
        resid_y = residualize(y, Z, Z_pinv)
        beta_manual = float((resid_x_real @ resid_y) / real_denom)

        # statsmodels formulas can't reliably reference bare int column names, so
        # alias the layer's column to a fixed name before building the formula.
        alias_df = wide.rename(columns={layer: "y_layer"})
        fit = smf.ols("y_layer ~ cooc_score + C(image_id) + C(target)", data=alias_df).fit(cov_type="cluster", cov_kwds={"groups": alias_df["image_id"]})
        ci = fit.conf_int().loc["cooc_score"]

        beta_perms_l = (resid_x_perms @ resid_y) / perm_denom
        perm_p = (int(np.sum(beta_perms_l >= beta_manual)) + 1) / (n_perm + 1)

        # Scale-free effect size: within-image partial correlation (Pearson r
        # between the SAME residualized vectors used for beta). Necessary
        # because raw logit-lens magnitudes are not calibrated across layers --
        # the residual stream's norm balloons in mid-network layers well before
        # later layers + the final norm rescale it back down (visible directly
        # in resid_y's own SD below), so a layer-to-layer comparison of raw beta
        # is confounded by each layer's own output scale, not just effect
        # strength. Its permutation null is corr(resid_x_perm, resid_y), from
        # the SAME reused permutation draws.
        resid_y_sd = float(resid_y.std(ddof=1))
        if resid_y_sd < 1e-9:
            partial_r = float("nan")
            partial_r_perms = np.full(n_perm, np.nan)
        else:
            partial_r = float(np.corrcoef(resid_x_real, resid_y)[0, 1])
            partial_r_perms = (resid_x_perms @ resid_y) / (np.sqrt(perm_denom) * np.sqrt(float(resid_y @ resid_y)))
        partial_r_perm_p = float("nan") if np.isnan(partial_r) else (int(np.sum(partial_r_perms >= partial_r)) + 1) / (n_perm + 1)

        per_layer_rows.append(
            {
                "layer": int(layer),
                "beta": float(fit.params["cooc_score"]),
                "beta_fwl_check": beta_manual,
                "se_cluster_image": float(fit.bse["cooc_score"]),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "p_value": float(fit.pvalues["cooc_score"]),
                "permutation_p_value": perm_p,
                "beta_shuffle_null_mean": float(beta_perms_l.mean()),
                "beta_shuffle_null_sd": float(beta_perms_l.std(ddof=1)),
                "beta_shuffle_null_q025": float(np.quantile(beta_perms_l, 0.025)),
                "beta_shuffle_null_q975": float(np.quantile(beta_perms_l, 0.975)),
                "r_squared": float(fit.rsquared),
                "resid_y_sd": resid_y_sd,
                "partial_r": partial_r,
                "partial_r_permutation_p_value": partial_r_perm_p,
                "partial_r_null_q025": float(np.nanquantile(partial_r_perms, 0.025)),
                "partial_r_null_q975": float(np.nanquantile(partial_r_perms, 0.975)),
            }
        )
        print(f"[stage11-analysis] layer={layer:2d} beta={fit.params['cooc_score']:.4f} partial_r={partial_r: .4f} "
              f"(resid_y_sd={resid_y_sd:.3f}) p={fit.pvalues['cooc_score']:.3g} perm_p={perm_p:.4g} r_perm_p={partial_r_perm_p:.4g}")

    per_layer_df = pd.DataFrame(per_layer_rows)
    n_llm_layers = max(layers)
    final_layer_row = per_layer_df[per_layer_df["layer"] == n_llm_layers].iloc[0]
    print(f"[stage11-analysis] VALIDATION: final layer ({n_llm_layers}) beta={final_layer_row['beta']:.4f} "
          f"-- should match Stage 10's beta=0.4061 (same data, same model)")

    # --- Figure 1: beta_l vs layer, with cluster-robust CI and permutation null band ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(per_layer_df["layer"], per_layer_df["beta_shuffle_null_q025"], per_layer_df["beta_shuffle_null_q975"],
                     color="gray", alpha=0.3, label="shuffled null (95% band)")
    ax.plot(per_layer_df["layer"], per_layer_df["beta"], color="C0", marker="o", markersize=3, label="beta_l (real)")
    ax.fill_between(per_layer_df["layer"], per_layer_df["ci_lower"], per_layer_df["ci_upper"], color="C0", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("LLM decoder layer (0=embedding output, 32=true final layer)")
    ax.set_ylabel("beta_l  (within-image FE coefficient on cooc_score, RAW units)")
    ax.set_title("beta_l by layer -- NOT scale-comparable across layers (see fig1b)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_beta_by_layer.png", dpi=150)
    plt.close(fig)

    # --- Figure 1b: scale-free partial correlation by layer (the fair cross-layer comparison) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(per_layer_df["layer"], per_layer_df["partial_r_null_q025"], per_layer_df["partial_r_null_q975"],
                     color="gray", alpha=0.3, label="shuffled null (95% band)")
    ax.plot(per_layer_df["layer"], per_layer_df["partial_r"], color="C1", marker="o", markersize=3, label="within-image partial r")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("LLM decoder layer (0=embedding output, 32=true final layer)")
    ax.set_ylabel("partial r  (cooc_score vs s_T_l, residualized on image+target FE)")
    ax.set_title("Scale-free localization: within-image partial correlation by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig1b_partial_r_by_layer.png", dpi=150)
    plt.close(fig)

    # --- Figure 1c: each layer's own residual-stream scale (why raw beta is not comparable) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(per_layer_df["layer"], per_layer_df["resid_y_sd"], color="C2", marker="o", markersize=3)
    ax.set_xlabel("LLM decoder layer")
    ax.set_ylabel("SD of residualized s_T_l (logit-lens units)")
    ax.set_title("Logit-lens output scale by layer (uncalibrated until the final norm)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig1c_residual_scale_by_layer.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: small multiples of FE-demeaned scatter at example layers ---
    example_layers = [l for l in config["example_layers"] if l in layers]
    fig, axes = plt.subplots(1, len(example_layers), figsize=(5 * len(example_layers), 5), sharey=True)
    if len(example_layers) == 1:
        axes = [axes]
    for ax, layer in zip(axes, example_layers):
        resid_y = residualize(wide[layer].to_numpy(dtype=float), Z, Z_pinv)
        row = per_layer_df[per_layer_df["layer"] == layer].iloc[0]
        ax.scatter(resid_x_real, resid_y, alpha=0.15, s=8)
        xs = np.linspace(resid_x_real.min(), resid_x_real.max(), 100)
        ax.plot(xs, row["beta"] * xs, color="black", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"layer {layer}\nbeta={row['beta']:.3f} p={row['p_value']:.2g}")
        ax.set_xlabel("cooc_score (residualized)")
    axes[0].set_ylabel("s_T_l (residualized)")
    fig.suptitle("Within-image relationship at example layers (FE-demeaned)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_example_layers_scatter.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: permutation p-value by layer ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(per_layer_df["layer"], per_layer_df["permutation_p_value"], marker="o", markersize=3)
    ax.axhline(float(config["alpha"]), color="red", linewidth=1.0, linestyle="--", label=f"alpha={config['alpha']}")
    ax.set_yscale("log")
    ax.set_xlabel("LLM decoder layer")
    ax.set_ylabel("permutation p-value (log scale)")
    ax.set_title("Permutation-test significance of beta_l by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_permutation_pvalue_by_layer.png", dpi=150)
    plt.close(fig)

    report = {
        "n_pairs": int(n),
        "n_layers": len(layers),
        "layers": [int(l) for l in layers],
        "per_layer": per_layer_df.to_dict(orient="records"),
        "validation_final_layer_vs_stage10": {
            "final_layer_beta": float(final_layer_row["beta"]),
            "stage10_beta": 0.40608014226535166,
            "abs_diff": float(abs(final_layer_row["beta"] - 0.40608014226535166)),
        },
        "note_vision_projector_not_probed": (
            "Vision-tower and projector layers are architecturally blind to the "
            "text/target (image-only computation), so a fixed image yields an "
            "identical vision/projector hidden state regardless of target -- "
            "within-image beta there is structurally 0 by construction and is "
            "not reported here. Stage 5's between-group excess-AUC result "
            "(-0.023, 95% CI [-0.025,-0.020]) is the correct existing test for "
            "those layers and is not repeated."
        ),
        "statistical_implementation": (
            "Per layer: statsmodels OLS 'y_layer ~ cooc_score + C(image_id) + C(target)', "
            "cluster-robust (CR1) SE by image_id. Permutation null: cooc_score globally "
            "shuffled (2000 permutations, same permutation draws reused across all layers), "
            "FWL-residualized against the SAME image+target fixed effects used in the real fit."
        ),
    }
    with (output_dir / "stage11_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[stage11-analysis] report written to {output_dir / 'stage11_report.json'}")


if __name__ == "__main__":
    main()
