"""Figures 1-6, reading only already-saved CSVs/JSON (no model inference).
Same plotting convention as the prior pilot's analysis/make_figures.py
(matplotlib Agg, dpi=140; new call site, that file is not imported/modified).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python analysis/visualization.py --data-config configs/data.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

MODEL_LABELS = {"original": "Original", "clean_debias": "Clean Debias", "adv_debias": "Adv Debias", "adv_decomp_debias": "Adv+Decomp"}
MODEL_COLORS = {"original": "#999999", "clean_debias": "#1f6feb", "adv_debias": "#d62728", "adv_decomp_debias": "#2ca02c"}
MODELS = ["original", "clean_debias", "adv_debias", "adv_decomp_debias"]


def fig1_adversarial_exposure(out_dir: Path) -> None:
    rows = list(csv.DictReader((out_dir / "data" / "adversarial_forget_set.csv").open("r", encoding="utf-8")))
    clean = [float(r["clean_s_ball"]) for r in rows]
    adv = [float(r["adv_s_ball"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    for c, a in zip(clean, adv):
        ax.plot([0, 1], [c, a], color="#cccccc", linewidth=0.8, zorder=1)
    ax.scatter([0] * len(clean), clean, color="#999999", s=25, label="clean", zorder=2)
    ax.scatter([1] * len(adv), adv, color="#d62728", s=25, label="adversarial", zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["clean", "adversarial"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("s_ball = logit(Yes) - logit(No)")
    ax.set_title(f"Fig 1: Adversarial exposure, TRAIN G10 (bat+, ball-), n={len(rows)}\nepsilon={float(rows[0]['epsilon']):.4f} (16/255, reused fixed)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "figures" / "fig1_adversarial_exposure.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def fig2_explained_variance(out_dir: Path) -> None:
    rows = list(csv.DictReader((out_dir / "decomposition" / "explained_variance.csv").open("r", encoding="utf-8")))
    comps = [r["component"] for r in rows]
    evr = [float(r["explained_variance_ratio"]) for r in rows]
    cum = [float(r["cumulative_variance_ratio"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(comps))
    ax.bar(x, evr, color="#1f6feb", label="explained variance ratio")
    ax2 = ax.twinx()
    ax2.plot(x, cum, color="#d62728", marker="o", label="cumulative")
    ax.set_xticks(x)
    ax.set_xticklabels(comps, rotation=45, ha="right")
    ax.set_ylabel("Explained variance ratio")
    ax2.set_ylabel("Cumulative variance ratio")
    ax2.set_ylim(0, 1.05)
    ax.set_title("Fig 2: PCA explained variance (fit on TRAIN G10 DEV delta_h)")
    fig.tight_layout()
    path = out_dir / "figures" / "fig2_explained_variance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def fig3_component_selectivity(out_dir: Path) -> None:
    rows = list(csv.DictReader((out_dir / "decomposition" / "component_selectivity.csv").open("r", encoding="utf-8")))
    pca_rows = [r for r in rows if r["candidate_type"] == "pca_component"]
    pca_rows.sort(key=lambda r: int(r["candidate_id"].replace("PC", "")))

    labels = [r["candidate_id"] for r in pca_rows]
    spur = [float(r["mean_delta_spurious"]) for r in pca_rows]
    target = [float(r["mean_delta_target"]) for r in pca_rows]
    context = [float(r["mean_delta_context"]) for r in pca_rows]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, spur, width, label="Spurious Ball (Delta_spurious)", color="#d62728")
    ax.bar(x, target, width, label="Genuine Ball (Delta_target)", color="#1f6feb")
    ax.bar(x + width, context, width, label="Bat (Delta_context)", color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean effect on s (after - before), internal VAL split")
    ax.set_title("Fig 3: Component intervention effects (h' = h - lambda*proj_u(h), avg over lambda)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "figures" / "fig3_component_selectivity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def fig4_mean_vs_best_component(out_dir: Path) -> None:
    rows = list(csv.DictReader((out_dir / "decomposition" / "component_selectivity.csv").open("r", encoding="utf-8")))
    mean_row = next(r for r in rows if r["candidate_type"] == "mean_direction")
    decomposed = [r for r in rows if r["candidate_type"] in ("pca_component", "pls_component")]
    best_row = max(decomposed, key=lambda r: float(r["selectivity_min_over_lambda"]))
    best_random = max((r for r in rows if r["candidate_type"] == "random_direction"), key=lambda r: float(r["selectivity_min_over_lambda"]))

    candidates = [mean_row, best_random, best_row]
    labels = ["Mean direction", f"Best random\n({best_random['candidate_id']})", f"Best component\n({best_row['candidate_id']})"]
    metrics = ["mean_delta_spurious", "mean_delta_target", "mean_delta_context"]
    metric_labels = ["Spurious Ball", "Genuine Ball", "Bat"]

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = [float(c[metric]) for c in candidates]
        ax.bar(x + (i - 1) * width, vals, width, label=mlabel)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean effect on s (after - before), internal VAL split")
    ax.set_title("Fig 4: Mean direction vs. best random direction vs. best decomposed component")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "figures" / "fig4_mean_vs_best_component.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def fig5_coupling_by_method(out_dir: Path, stats: dict) -> None:
    B = [stats["coupling_B"][m] for m in MODELS]
    ci_lo = [stats["descriptive"][m]["coupling_B"]["ci_lower"] for m in MODELS]
    ci_hi = [stats["descriptive"][m]["coupling_B"]["ci_upper"] for m in MODELS]
    err_lo = [b - lo for b, lo in zip(B, ci_lo)]
    err_hi = [hi - b for b, hi in zip(B, ci_hi)]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MODELS))
    ax.bar(x, B, yerr=[err_lo, err_hi], color=[MODEL_COLORS[m] for m in MODELS], capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Coupling B = E[s_ball | G10 test] - E[s_ball | G00 test]\n(bootstrap 95% CI)")
    ax.set_title("Fig 5: Functional coupling on unseen CLEAN test images, all 4 models")
    fig.tight_layout()
    path = out_dir / "figures" / "fig5_coupling_by_method.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def fig6_method_selectivity(out_dir: Path, stats: dict) -> None:
    g10_s_ball = [stats["descriptive"][m]["g10_s_ball"]["mean"] for m in MODELS]
    ball_acc = [stats["ball_plus_acc"][m] for m in MODELS]
    bat_acc = [stats["bat_plus_acc"][m] for m in MODELS]
    labels = [MODEL_LABELS[m] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(MODELS))

    axes[0].bar(x, g10_s_ball, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("G10 suppression\n(s_ball, lower is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")

    axes[1].bar(x, ball_acc, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Genuine Ball+ retention\n(accuracy, higher is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    axes[2].bar(x, bat_acc, color=colors)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Bat+ retention\n(accuracy, higher is better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")

    fig.suptitle("Fig 6: Method selectivity -- spurious suppression vs. genuine retention, all 4 models")
    fig.tight_layout()
    path = out_dir / "figures" / "fig6_method_selectivity.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    args = parser.parse_args()
    with args.data_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    fig1_adversarial_exposure(out_dir)
    fig2_explained_variance(out_dir)
    fig3_component_selectivity(out_dir)
    fig4_mean_vs_best_component(out_dir)

    stats_path = out_dir / "evaluation" / "statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        fig5_coupling_by_method(out_dir, stats)
        fig6_method_selectivity(out_dir, stats)
    else:
        print(f"[visualization] {stats_path} not found yet -- skipping Fig 5/6 (run analysis/statistics.py first)")


if __name__ == "__main__":
    main()
