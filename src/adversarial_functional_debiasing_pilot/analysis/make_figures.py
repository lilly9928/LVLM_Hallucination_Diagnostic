"""Figures 1-3, reading only already-saved CSVs/JSON (no model inference).
Follows this repo's existing plotting convention (matplotlib Agg, dpi=140, see
scripts/make_exp5_figure.py).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python analysis/make_figures.py --config configs/pilot.yaml
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

MODEL_LABELS = {"original": "Original", "clean_debias": "Clean Debias", "adv_debias": "Adv Debias"}
MODEL_COLORS = {"original": "#999999", "clean_debias": "#1f6feb", "adv_debias": "#d62728"}


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
    ax.set_ylabel("s_ball = logit(Yes) - logit(No)\n(\"Is there a sports ball in the image?\")")
    ax.set_title(f"Fig 1: Adversarial exposure, TRAIN G10 (bat+, ball-), n={len(rows)}\nepsilon={float(rows[0]['epsilon']):.4f} (16/255, fixed)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "figures" / "fig1_adversarial_exposure.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def fig2_clean_test_coupling(out_dir: Path, stats: dict) -> None:
    models = ["original", "clean_debias", "adv_debias"]
    B = [stats["coupling_B"][m] for m in models]
    ci_lo = [stats["descriptive"][m]["coupling_B"]["ci_lower"] for m in models]
    ci_hi = [stats["descriptive"][m]["coupling_B"]["ci_upper"] for m in models]
    err_lo = [b - lo for b, lo in zip(B, ci_lo)]
    err_hi = [hi - b for b, hi in zip(B, ci_hi)]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]
    ax.bar(x, B, yerr=[err_lo, err_hi], color=colors, capsize=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Coupling B = E[s_ball | G10 test] - E[s_ball | G00 test]\n(bootstrap 95% CI)")
    ax.set_title("Fig 2: Functional coupling on unseen CLEAN test images")
    fig.tight_layout()
    path = out_dir / "figures" / "fig2_clean_test_coupling.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def fig3_selectivity(out_dir: Path, stats: dict) -> None:
    models = ["original", "clean_debias", "adv_debias"]
    g10_s_ball = [stats["descriptive"][m]["g10_s_ball"]["mean"] for m in models]
    ball_acc = [stats["ball_plus_acc"][m] for m in models]
    bat_acc = [stats["bat_plus_acc"][m] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(models))
    colors = [MODEL_COLORS[m] for m in models]
    labels = [MODEL_LABELS[m] for m in models]

    axes[0].bar(x, g10_s_ball, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Spurious G10 evidence\n(s_ball, lower is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")

    axes[1].bar(x, ball_acc, color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Genuine Ball+ accuracy\n(target retention, higher is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    axes[2].bar(x, bat_acc, color=colors)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Genuine Bat+ accuracy\n(context retention, higher is better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")

    fig.suptitle("Fig 3: Selectivity -- spurious suppression vs. genuine retention")
    fig.tight_layout()
    path = out_dir / "figures" / "fig3_selectivity.png"
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(config["output_dir"])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    stats = json.loads((out_dir / "evaluation" / "statistics.json").read_text())

    fig1_adversarial_exposure(out_dir)
    fig2_clean_test_coupling(out_dir, stats)
    fig3_selectivity(out_dir, stats)


if __name__ == "__main__":
    main()
