"""Stage 2: sample high-/low-co-occurrence absent-target candidates from COCO
val2017 and confound-match them on marginal frequency, average object area, and
CLIP image-text similarity.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage2_sampling_matching.py \
        --config configs/stage2_sampling_matching.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.clip_similarity import load_or_compute_similarity
from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.cooccurrence_stats import compute_cooccurrence_stats
from cooc_diagnostic.covariates import compute_category_average_area
from cooc_diagnostic.matching import MatchUnit, balance_table, coarsened_exact_match
from cooc_diagnostic.strata_sampling import build_candidates, split_strata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: sampling and confound matching")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def write_balance_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    min_support = int(config["min_support_count"])

    print(f"[stage2] loading train2017 annotations: {config['train_annotation_path']}")
    train_index = load_coco_instances(config["train_annotation_path"])
    train_stats = compute_cooccurrence_stats(train_index.category_ids, train_index.image_categories)
    category_index = {cid: i for i, cid in enumerate(train_stats.category_ids)}
    names_by_id = train_index.category_names

    freq_by_id = {
        cid: float(train_stats.marginal_counts[category_index[cid]]) / train_stats.n_images
        for cid in train_stats.category_ids
    }
    rare_threshold = float(config["rare_category_freq_threshold_pct"]) / 100.0
    eligible_category_ids = sorted(cid for cid, freq in freq_by_id.items() if freq >= rare_threshold)
    excluded = sorted(names_by_id[cid] for cid in train_stats.category_ids if cid not in eligible_category_ids)
    print(f"[stage2] eligible target categories: {len(eligible_category_ids)}/80 (excluded: {excluded})")

    area_by_id = compute_category_average_area(config["train_annotation_path"])

    print(f"[stage2] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])
    print(f"[stage2] val2017 images: {len(val_index.image_categories)}")

    candidates = build_candidates(
        val_index.image_categories,
        eligible_category_ids,
        category_index,
        train_stats.pmi,
        train_stats.joint_counts,
        min_support=min_support,
    )
    print(f"[stage2] candidates with >=1 usable present-object term: {len(candidates)}")

    treatment_c, control_c, split_info = split_strata(
        candidates, lower_pct=float(config["stratum_lower_pct"]), upper_pct=float(config["stratum_upper_pct"])
    )
    print(
        f"[stage2] strata split: treatment={split_info['n_treatment']} control={split_info['n_control']} "
        f"middle_dropped={split_info['n_middle_dropped']} (low_cut={split_info['low_cut']:.3f}, "
        f"high_cut={split_info['high_cut']:.3f})"
    )

    candidate_image_ids = sorted({c.image_id for c in treatment_c} | {c.image_id for c in control_c})
    image_paths = {
        img_id: str(Path(config["val_image_dir"]) / val_index.image_filenames[img_id]) for img_id in candidate_image_ids
    }
    eligible_names = [names_by_id[cid] for cid in eligible_category_ids]

    print(f"[stage2] computing CLIP image-text similarity for {len(image_paths)} images x {len(eligible_names)} categories")
    clip_image_ids, sim_matrix = load_or_compute_similarity(
        cache_path=output_dir / "clip_sim_cache.npz",
        image_paths=image_paths,
        category_names=eligible_names,
        model_id=config["clip_model_id"],
        device=config["clip_device"],
        batch_size=int(config["clip_batch_size"]),
    )
    image_row = {img_id: i for i, img_id in enumerate(clip_image_ids)}
    category_col = {cid: i for i, cid in enumerate(eligible_category_ids)}

    def to_match_unit(c) -> MatchUnit:
        return MatchUnit(
            image_id=c.image_id,
            category_id=c.category_id,
            score=c.score,
            freq=freq_by_id[c.category_id],
            area=area_by_id[c.category_id],
            clip_sim=float(sim_matrix[image_row[c.image_id], category_col[c.category_id]]),
        )

    treatment = [to_match_unit(c) for c in treatment_c]
    control = [to_match_unit(c) for c in control_c]

    before_balance = balance_table(treatment, control)
    write_balance_csv(output_dir / "balance_before_matching.csv", before_balance)
    print("[stage2] balance BEFORE matching:")
    for row in before_balance:
        print(f"  {row['covariate']:14s} smd={row['smd']:+.3f}  treat_mean={row['treat_mean']:.4f}  control_mean={row['control_mean']:.4f}")

    result = coarsened_exact_match(
        treatment,
        control,
        freq_bins=int(config["freq_bins"]),
        area_bins=int(config["area_bins"]),
        clip_bins=int(config["clip_bins"]),
    )
    matched_treatment = [p[0] for p in result["pairs"]]
    matched_control = [p[1] for p in result["pairs"]]

    after_balance = balance_table(matched_treatment, matched_control)
    write_balance_csv(output_dir / "balance_after_matching.csv", after_balance)
    print(f"[stage2] matched pairs: {result['n_matched_pairs']} (treatment pool={result['n_treatment_total']}, control pool={result['n_control_total']})")
    print("[stage2] balance AFTER matching:")
    for row in after_balance:
        print(f"  {row['covariate']:14s} smd={row['smd']:+.3f}  treat_mean={row['treat_mean']:.4f}  control_mean={row['control_mean']:.4f}")

    with (output_dir / "matched_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "image_id_treatment",
                "category_treatment",
                "score_treatment",
                "clip_sim_treatment",
                "image_id_control",
                "category_control",
                "score_control",
                "clip_sim_control",
                "freq_bin",
                "area_bin",
                "clip_bin",
            ]
        )
        for pair_id, (t, c, cell) in enumerate(result["pairs"]):
            writer.writerow(
                [
                    pair_id,
                    t.image_id,
                    names_by_id[t.category_id],
                    f"{t.score:.4f}",
                    f"{t.clip_sim:.4f}",
                    c.image_id,
                    names_by_id[c.category_id],
                    f"{c.score:.4f}",
                    f"{c.clip_sim:.4f}",
                    cell[0],
                    cell[1],
                    cell[2],
                ]
            )

    print(f"[stage2] outputs written to {output_dir}")


if __name__ == "__main__":
    main()
