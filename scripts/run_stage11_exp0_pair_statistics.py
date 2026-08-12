"""Stage 11 Exp0: pair sanity check for the fixed case-study pair
context="baseball bat", target="sports ball".

Reuses Stage 1's exact PMI table/definition (cooccurrence_stats.py) rather than
recomputing PMI, and Stage 1's own output CSVs directly where possible. Group
membership (G00/G10/G01/G11) is computed fresh over val2017 since Stage 1 only
stores pairwise train2017 statistics, not per-image val2017 group membership.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp0_pair_statistics.py \
        --config configs/stage11_case_bat_ball.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.cooccurrence_stats import compute_cooccurrence_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp0: pair statistics")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    context_name = config["context_category"]
    target_name = config["target_category"]

    print(f"[exp0] loading train2017 annotations: {config['train_annotation_path']}")
    train_index = load_coco_instances(config["train_annotation_path"])
    name_to_id = {c.name: c.id for c in train_index.categories}
    context_id = name_to_id[context_name]
    target_id = name_to_id[target_name]

    stats = compute_cooccurrence_stats(train_index.category_ids, train_index.image_categories)
    id_to_pos = {cid: i for i, cid in enumerate(train_index.category_ids)}
    a, b = id_to_pos[context_id], id_to_pos[target_id]

    n_bat = int(stats.marginal_counts[a])
    n_ball = int(stats.marginal_counts[b])
    n_bat_and_ball = int(stats.joint_counts[a, b])
    n_train_images = len(train_index.image_categories)
    p_ball = n_ball / n_train_images
    p_ball_given_bat = n_bat_and_ball / n_bat
    pmi = float(stats.pmi[a, b])
    lift = float(stats.lift[a, b])

    # Rank among all COCO pairs by PMI (descending), among pairs with any joint co-occurrence.
    import numpy as np

    finite_mask = np.isfinite(stats.pmi)
    flat_pmi = stats.pmi[finite_mask]
    rank = int((flat_pmi > pmi).sum()) + 1
    total_ranked = int(finite_mask.sum())

    print(f"[exp0] context={context_name} (id={context_id}), target={target_name} (id={target_id})")
    print(f"[exp0] N(bat)={n_bat} N(ball)={n_ball} N(bat&ball)={n_bat_and_ball}")
    print(f"[exp0] P(ball)={p_ball:.4f} P(ball|bat)={p_ball_given_bat:.4f} PMI={pmi:.4f} lift={lift:.4f}")
    print(f"[exp0] PMI rank: {rank} / {total_ranked} ordered (category,category) entries (train2017, finite PMI)")

    # --- Group membership on val2017 ---
    print(f"[exp0] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])
    val_name_to_id = {c.name: c.id for c in val_index.categories}
    val_context_id = val_name_to_id[context_name]
    val_target_id = val_name_to_id[target_name]
    names_by_id = val_index.category_names

    rows = []
    group_counts = {"G00": 0, "G10": 0, "G01": 0, "G11": 0}
    for image_id, cats in val_index.image_categories.items():
        bat_present = val_context_id in cats
        ball_present = val_target_id in cats
        group = ("G" + ("1" if bat_present else "0") + ("1" if ball_present else "0"))
        group_counts[group] += 1
        rows.append(
            {
                "image_id": image_id,
                "group": group,
                "bat_present": int(bat_present),
                "ball_present": int(ball_present),
                "n_present_categories": len(cats),
                "present_objects": "|".join(sorted(names_by_id[c] for c in cats)),
            }
        )
    rows.sort(key=lambda r: r["image_id"])

    membership_path = output_dir / "exp0_group_membership.csv"
    with membership_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp0] group counts: {group_counts} (total val2017 images={len(val_index.image_categories)})")
    print(f"[exp0] wrote group membership to {membership_path}")

    go_no_go = {
        "meaningful_cooccurrence": pmi > 0 and n_bat_and_ball >= 10,
        "sufficient_g10": group_counts["G10"] >= 30,
    }
    go = all(go_no_go.values())

    summary = {
        "context_category": context_name,
        "target_category": target_name,
        "context_category_id": context_id,
        "target_category_id": target_id,
        "train2017": {
            "n_images": n_train_images,
            "n_bat": n_bat,
            "n_ball": n_ball,
            "n_bat_and_ball": n_bat_and_ball,
            "p_ball": p_ball,
            "p_ball_given_bat": p_ball_given_bat,
            "pmi": pmi,
            "lift": lift,
            "pmi_rank": rank,
            "pmi_rank_total": total_ranked,
        },
        "val2017_groups": group_counts,
        "go_no_go_criteria": go_no_go,
        "decision": "GO" if go else "STOP",
    }
    stats_path = output_dir / "exp0_pair_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[exp0] wrote pair statistics to {stats_path}")
    print(f"[exp0] DECISION: {'GO' if go else 'STOP'} -- {go_no_go}")


if __name__ == "__main__":
    main()
