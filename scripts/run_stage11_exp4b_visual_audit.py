"""Stage 11 Exp4B: mandatory visual audit of the Exp4 counterfactual
intervention, before interpreting the numeric Delta_bat_to_ball / Delta_sham
results. Renders original / bat-removed / sham triplets for representative
cases (largest positive delta, near-zero, negative delta, large sham effect).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp4b_visual_audit.py \
        --config configs/stage11_case_bat_ball.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp4B: counterfactual visual audit")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])
    image_dir = output_dir / "counterfactual_images"

    df = pd.read_csv(output_dir / "exp4_counterfactual.csv")
    bat = df[df.condition == "bat_removed"].sort_values("delta", ascending=False).reset_index(drop=True)
    sham = df[df.condition == "sham"].set_index("image_id")

    nonzero = bat[bat["delta"] != 0.0]
    selections = {
        "largest_positive_delta": bat.iloc[0]["image_id"],
        "second_largest_positive_delta": bat.iloc[1]["image_id"],
        "near_zero_delta": nonzero.iloc[(nonzero["delta"]).abs().argsort()[0]]["image_id"],
        "exact_zero_delta_artifact_example": bat[bat["delta"] == 0.0].iloc[0]["image_id"] if (bat["delta"] == 0.0).any() else None,
        "most_negative_delta": bat.iloc[-1]["image_id"],
        "second_most_negative_delta": bat.iloc[-2]["image_id"],
        "largest_sham_effect": sham["delta"].abs().idxmax(),
    }
    selections = {k: int(v) for k, v in selections.items() if v is not None}
    with (output_dir / "exp4b_selected_examples.json").open("w", encoding="utf-8") as f:
        json.dump(selections, f, indent=2)

    n_rows = len(selections)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row_i, (label, image_id) in enumerate(selections.items()):
        bat_row = bat[bat.image_id == image_id].iloc[0]
        sham_row = sham.loc[image_id]
        s_orig = bat_row["original_s_ball"]
        s_bat = bat_row["intervention_s_ball"]
        s_sham = sham_row["intervention_s_ball"]
        paths = [
            image_dir / f"{image_id}_original.png",
            image_dir / f"{image_id}_bat_removed.png",
            image_dir / f"{image_id}_sham.png",
        ]
        titles = [
            f"original\ns_ball={s_orig:.2f}",
            f"bat removed\ns_ball={s_bat:.2f}  (Δ={s_orig - s_bat:.2f})",
            f"sham\ns_ball={s_sham:.2f}  (Δ={s_orig - s_sham:.2f})",
        ]
        for col_i, (path, title) in enumerate(zip(paths, titles)):
            ax = axes[row_i, col_i]
            ax.imshow(Image.open(path))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row_i, 0].set_ylabel(f"{label}\nid={image_id}", fontsize=8, rotation=0, labelpad=40, ha="right")

    fig.suptitle('Exp4B visual audit: baseball-bat removal vs sham (target = "sports ball")', fontsize=11)
    fig.tight_layout(rect=[0.05, 0, 1, 0.97])
    fig_path = output_dir / "figures" / "exp4b_visual_audit.png"
    fig.savefig(fig_path, dpi=130)
    print(f"[exp4b] wrote {fig_path}")
    print(f"[exp4b] selections: {selections}")


if __name__ == "__main__":
    main()
