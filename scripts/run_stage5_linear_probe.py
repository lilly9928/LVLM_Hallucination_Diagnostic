"""Stage 5: linear probe on frozen CLIP visual features, measuring excess AUC
over a present-object-set-only baseline.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage5_linear_probe.py \
        --config configs/stage5_linear_probe.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.clip_similarity import load_or_compute_image_embeddings
from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.cooccurrence_stats import compute_cooccurrence_stats
from cooc_diagnostic.linear_probe import ProbeExample, bootstrap_excess_auc_ci, build_feature_matrices, fit_probe, split_examples_by_image
from cooc_diagnostic.strata_sampling import build_candidates, split_strata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: linear probe on frozen CLIP features")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    min_support = int(config["min_support_count"])

    print(f"[stage5] loading train2017 annotations: {config['train_annotation_path']}")
    train_index = load_coco_instances(config["train_annotation_path"])
    train_stats = compute_cooccurrence_stats(train_index.category_ids, train_index.image_categories)
    category_index = {cid: i for i, cid in enumerate(train_stats.category_ids)}
    names_by_id = train_index.category_names

    freq_by_id = {
        cid: float(train_stats.marginal_counts[category_index[cid]]) / train_stats.n_images for cid in train_stats.category_ids
    }
    rare_threshold = float(config["rare_category_freq_threshold_pct"]) / 100.0
    eligible_category_ids = sorted(cid for cid, freq in freq_by_id.items() if freq >= rare_threshold)
    print(f"[stage5] eligible target categories: {len(eligible_category_ids)}/80")

    print(f"[stage5] loading val2017 annotations: {config['val_annotation_path']}")
    val_index = load_coco_instances(config["val_annotation_path"])

    candidates = build_candidates(
        val_index.image_categories, eligible_category_ids, category_index, train_stats.pmi, train_stats.joint_counts, min_support=min_support
    )
    print(f"[stage5] usable candidates: {len(candidates)}")

    treatment_c, control_c, split_info = split_strata(
        candidates, lower_pct=float(config["stratum_lower_pct"]), upper_pct=float(config["stratum_upper_pct"])
    )
    print(f"[stage5] strata: treatment={len(treatment_c)} control={len(control_c)} middle_dropped={split_info['n_middle_dropped']}")

    examples = [ProbeExample(image_id=c.image_id, category_id=c.category_id, label=1) for c in treatment_c] + [
        ProbeExample(image_id=c.image_id, category_id=c.category_id, label=0) for c in control_c
    ]

    image_ids_needed = sorted({ex.image_id for ex in examples})
    image_paths = {img_id: str(Path(config["val_image_dir"]) / val_index.image_filenames[img_id]) for img_id in image_ids_needed}
    print(f"[stage5] computing CLIP image embeddings for {len(image_paths)} images")
    clip_image_ids, embeddings = load_or_compute_image_embeddings(
        cache_path=output_dir / "clip_image_embeddings_cache.npz",
        image_paths=image_paths,
        model_id=config["clip_model_id"],
        device=config["clip_device"],
        batch_size=int(config["clip_batch_size"]),
    )
    image_clip_embeddings = {img_id: embeddings[i] for i, img_id in enumerate(clip_image_ids)}

    mats = build_feature_matrices(examples, eligible_category_ids, val_index.image_categories, image_clip_embeddings)
    print(f"[stage5] baseline_X shape={mats['baseline_X'].shape} full_X shape={mats['full_X'].shape}")

    rng = np.random.default_rng(int(config["seed"]))
    train_idx, test_idx = split_examples_by_image(examples, test_frac=float(config["test_frac"]), rng=rng)
    print(f"[stage5] train examples={len(train_idx)} test examples={len(test_idx)} (split by image)")

    y_train, y_test = mats["labels"][train_idx], mats["labels"][test_idx]

    print("[stage5] fitting baseline probe (one-hot(A) + multi-hot(Y))")
    baseline_clf = fit_probe(mats["baseline_X"][train_idx], y_train)
    baseline_probs = baseline_clf.predict_proba(mats["baseline_X"][test_idx])[:, 1]
    baseline_auc = float(roc_auc_score(y_test, baseline_probs))

    print("[stage5] fitting full probe (one-hot(A) + frozen CLIP image embedding)")
    full_clf = fit_probe(mats["full_X"][train_idx], y_train)
    full_probs = full_clf.predict_proba(mats["full_X"][test_idx])[:, 1]
    full_auc = float(roc_auc_score(y_test, full_probs))

    excess_auc = full_auc - baseline_auc
    print(f"[stage5] baseline AUC (Y-only) = {baseline_auc:.4f} (best C={baseline_clf.C_[0]:.4g})")
    print(f"[stage5] full AUC (CLIP feature) = {full_auc:.4f} (best C={full_clf.C_[0]:.4g})")
    print(f"[stage5] excess AUC = {excess_auc:.4f}")

    ci = bootstrap_excess_auc_ci(y_test, baseline_probs, full_probs, n_boot=int(config["n_bootstrap"]), rng=rng)
    print(f"[stage5] excess AUC 95% bootstrap CI = [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] (n_boot_used={ci['n_boot_used']})")

    report = {
        "n_eligible_categories": len(eligible_category_ids),
        "n_candidates_total": len(candidates),
        "strata_split_info": split_info,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "baseline_auc": baseline_auc,
        "baseline_best_C": float(baseline_clf.C_[0]),
        "full_auc": full_auc,
        "full_best_C": float(full_clf.C_[0]),
        "excess_auc": excess_auc,
        "excess_auc_bootstrap_ci": ci,
    }
    with (output_dir / "stage5_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[stage5] report written to {output_dir / 'stage5_report.json'}")


if __name__ == "__main__":
    main()
