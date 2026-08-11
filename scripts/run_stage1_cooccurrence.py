"""Stage 1: compute COCO train2017 category co-occurrence statistics.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage1_cooccurrence.py \
        --config configs/stage1_cooccurrence.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.cooccurrence_stats import compute_cooccurrence_stats, get_top_bottom_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: COCO co-occurrence statistics")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def save_matrix_csv(path: Path, matrix: np.ndarray, names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + [f"{v:.6g}" for v in row])


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage1] loading COCO train2017 annotations: {config['train_annotation_path']}")
    index = load_coco_instances(config["train_annotation_path"])
    category_ids = index.category_ids
    names_by_id = index.category_names
    names = [names_by_id[cid] for cid in category_ids]

    print(f"[stage1] images={len(index.image_categories)} categories={len(category_ids)}")

    stats = compute_cooccurrence_stats(category_ids, index.image_categories)

    save_matrix_csv(output_dir / "cooccurrence_counts.csv", stats.joint_counts.astype(float), names)
    save_matrix_csv(output_dir / "cooccurrence_pmi.csv", stats.pmi, names)
    save_matrix_csv(output_dir / "cooccurrence_lift.csv", stats.lift, names)
    save_matrix_csv(output_dir / "cooccurrence_conditional.csv", stats.conditional, names)

    with (output_dir / "marginal_frequency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count_images", "p_marginal"])
        for cid, cname, count in zip(category_ids, names, stats.marginal_counts):
            writer.writerow([cname, int(count), count / stats.n_images])

    min_support = int(config.get("min_support_count", 10))
    top_k = int(config.get("top_k", 20))
    pairs = get_top_bottom_pairs(stats, names_by_id, min_support=min_support, top_k=top_k)

    with (output_dir / "top_bottom_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank_group", "category_a", "category_b", "joint_count", "pmi", "lift", "p_b_given_a", "p_a_given_b"]
        )
        for group in ("top", "bottom"):
            for r in pairs[group]:
                writer.writerow(
                    [
                        group,
                        r["category_a"],
                        r["category_b"],
                        r["joint_count"],
                        f"{r['pmi']:.4f}",
                        f"{r['lift']:.4f}",
                        f"{r['p_b_given_a']:.4f}",
                        f"{r['p_a_given_b']:.4f}",
                    ]
                )

    print(
        f"[stage1] total pairs={pairs['n_pairs_total']} "
        f"excluded_by_min_support(<{min_support})={pairs['n_pairs_excluded_by_min_support']}"
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(stats.pmi, cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title("COCO train2017 category co-occurrence PMI")
    fig.colorbar(im, ax=ax, label="PMI")
    fig.tight_layout()
    fig.savefig(plots_dir / "pmi_heatmap.png", dpi=150)
    plt.close(fig)

    freqs = stats.marginal_counts / stats.n_images
    order = np.argsort(-freqs)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), freqs[order])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("P(category present in image)")
    ax.set_title("COCO train2017 marginal frequency distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "marginal_frequency_hist.png", dpi=150)
    plt.close(fig)

    print(f"[stage1] outputs written to {output_dir}")


if __name__ == "__main__":
    main()
