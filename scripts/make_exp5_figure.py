"""Quick standalone figure: Exp5 layerwise Delta_bat_to_ball vs Delta_sham."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])
    stats = json.loads((output_dir / "exp5_statistics.json").read_text())
    rows = stats["per_stage"]

    labels = [r["stage"] for r in rows]
    x = list(range(len(rows)))
    delta_bat = [r["delta_bat_mean"] for r in rows]
    delta_sham = [r["delta_sham_mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, delta_bat, marker="o", markersize=3, label="Delta_bat_to_ball (original - bat_removed)", color="#1f6feb")
    ax.plot(x, delta_sham, marker="o", markersize=3, label="Delta_sham (original - sham)", color="#999999")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvspan(-0.5, 1.5, color="#ffedd5", alpha=0.5, label="vision tower / projector")
    peak_idx = max(range(len(rows)), key=lambda i: delta_bat[i])
    ax.axvline(peak_idx, color="#d62728", linestyle="--", linewidth=1, label=f"peak: {labels[peak_idx]}")

    tick_positions = list(range(0, len(rows), 3))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[i] for i in tick_positions], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean e_ball shift (logit-lens units for LLM layers; probe decision_function for vision/projector)")
    ax.set_title('Exp5: where Bat-dependent Ball evidence emerges (G10 images, n=65)\ncontext="baseball bat" -> target="sports ball"')
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.subplots_adjust(left=0.09)

    fig_path = output_dir / "figures" / "exp5_layerwise_localization.png"
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
