"""Stage 11 Exp1 (part A): build the matched G10-vs-G00 sample for the
phenomenon test, BEFORE looking at any epsilon* or s_ball result.

G10 (bat+, ball-) has only 65 val2017 images -- use all of them as treatment.
G00 (bat-, ball-) has 4766 -- far too many to attack, so an equal-sized control
sample is drawn via the same coarsened-exact-match mechanism Stage 2 uses
(matching.py::coarsened_exact_match), repurposed to per-image covariates since
context/target category is fixed for every unit here (unlike Stage 2's
cross-category candidates):
  freq     <- n_other_present (COCO categories present besides bat/ball itself;
              scene-complexity confound)
  area     <- CLIP image-text similarity to "a photo of a sports ball"
              (residual ball-like visual content, even though unlabeled)
  clip_sim <- CLIP image-text similarity to "a photo of a baseball bat"
              (residual bat-like visual content)
This fixes the sample before any attack is run -- no post-hoc matching.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp1_build_sample.py \
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

from cooc_diagnostic.clip_similarity import load_or_compute_similarity
from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.matching import MatchUnit, balance_table, coarsened_exact_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp1A: build matched G10/G00 sample")
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

    val_index = load_coco_instances(config["val_annotation_path"])
    name_to_id = {c.name: c.id for c in val_index.categories}
    context_id = name_to_id[context_name]
    target_id = name_to_id[target_name]
    image_dir = Path(config["val_image_dir"])

    g10_ids, g00_ids = [], []
    for image_id, cats in val_index.image_categories.items():
        bat = context_id in cats
        ball = target_id in cats
        if bat and not ball:
            g10_ids.append(image_id)
        elif not bat and not ball:
            g00_ids.append(image_id)
    print(f"[exp1a] G10 (treatment pool) = {len(g10_ids)}, G00 (control pool) = {len(g00_ids)}")

    pool_ids = g10_ids + g00_ids
    image_paths = {iid: str(image_dir / val_index.image_filenames[iid]) for iid in pool_ids}

    print(f"[exp1a] computing CLIP image-text similarity for {len(pool_ids)} images (cached)")
    clip_ids, sim_matrix = load_or_compute_similarity(
        cache_path=output_dir / "exp1_clip_sim_cache.npz",
        image_paths=image_paths,
        category_names=[target_name, context_name],
        model_id=config["clip_model_id"],
        device=config["clip_device"],
    )
    sim_by_image = {iid: sim_matrix[i] for i, iid in enumerate(clip_ids)}

    def n_other_present(image_id: int) -> int:
        cats = val_index.image_categories[image_id]
        return len(cats) - (1 if context_id in cats else 0)

    def make_unit(image_id: int) -> MatchUnit:
        clip_ball, clip_bat = sim_by_image[image_id]
        return MatchUnit(
            image_id=image_id,
            category_id=target_id,
            score=0.0,
            freq=float(n_other_present(image_id)),
            area=float(clip_ball),
            clip_sim=float(clip_bat),
        )

    treatment_units = [make_unit(iid) for iid in g10_ids]
    control_units = [make_unit(iid) for iid in g00_ids]

    match_result = coarsened_exact_match(
        treatment_units, control_units,
        freq_bins=int(config["freq_bins"]), area_bins=int(config["area_bins"]), clip_bins=int(config["clip_bins"]),
    )
    pairs = match_result["pairs"]
    print(f"[exp1a] matched {len(pairs)} / {len(treatment_units)} treatment units "
          f"(dropped {match_result['n_treatment_dropped_no_cell_overlap_or_excess']} with no control-cell overlap)")

    matched_treatment = [t for t, c, cell in pairs]
    matched_control = [c for t, c, cell in pairs]
    balance = balance_table(matched_treatment, matched_control)
    print("[exp1a] post-match covariate balance (SMD, |.|<0.25 ~ well-balanced):")
    for row in balance:
        print(f"  {row['covariate']:>14s}: treat={row['treat_mean']:.3f} control={row['control_mean']:.3f} smd={row['smd']:.3f}")

    rows = []
    names_by_id = val_index.category_names
    for pair_id, (t, c, cell) in enumerate(pairs):
        for arm, unit in [("treatment", t), ("control", c)]:
            cats = val_index.image_categories[unit.image_id]
            rows.append(
                {
                    "pair_id": pair_id,
                    "arm": arm,
                    "image_id": unit.image_id,
                    "group": "G10" if arm == "treatment" else "G00",
                    "category": target_name,
                    "n_other_present": unit.freq,
                    "clip_sim_ball": unit.area,
                    "clip_sim_bat": unit.clip_sim,
                    "match_cell": str(cell),
                    "present_objects": "|".join(sorted(names_by_id[cid] for cid in cats)),
                }
            )

    sample_path = output_dir / "exp1_sample_selection.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp1a] wrote {len(rows)} sample rows ({len(pairs)} pairs) to {sample_path}")

    with (output_dir / "exp1_sample_balance.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_treatment_pool": len(treatment_units),
                "n_control_pool": len(control_units),
                "n_matched_pairs": len(pairs),
                "n_treatment_dropped": match_result["n_treatment_dropped_no_cell_overlap_or_excess"],
                "balance_table": balance,
                "freq_bin_edges": match_result["freq_bin_edges"],
                "area_bin_edges": match_result["area_bin_edges"],
                "clip_bin_edges": match_result["clip_bin_edges"],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
