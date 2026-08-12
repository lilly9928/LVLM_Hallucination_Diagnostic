"""Experiment 4, Step 2: estimate the co-occurrence direction d_L at each
candidate layer (audit Sec.4), TRAIN split only. Also produces:
  - 5 norm-matched random directions per layer (control 3)
  - 1 shuffled-cooc-score direction per layer (control 4)
All directions share the same reference point (train-split mean activation
at that layer) so every control uses an identical patch mechanism -- only
the direction itself differs.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python 02_estimate_directions.py \
        --config ../configs/02_estimate_directions.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cooccurrence_causal_coupling.common import CANDIDATE_LAYERS, build_fe_projection, fe_regression_direction, residualize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    meta = pd.read_csv(config["meta_path"])
    hidden = np.load(config["hidden_states_path"])
    train_mask = (meta["split"] == "train").to_numpy()
    train_meta = meta[train_mask].reset_index(drop=True)
    print(f"[direction] train rows: {len(train_meta)} (of {len(meta)} train+val)")

    Z, Z_pinv = build_fe_projection(train_meta)
    score_resid = residualize(train_meta["cooc_score"].to_numpy(dtype=float), Z, Z_pinv)

    shuffled_score = rng.permutation(train_meta["cooc_score"].to_numpy(dtype=float))
    shuffled_score_resid = residualize(shuffled_score, Z, Z_pinv)

    directions_payload = {}
    metadata_rows = []
    for L in CANDIDATE_LAYERS:
        H = hidden[f"layer_{L}"][train_mask]  # (n_train, 4096)
        reference = H.mean(axis=0)

        H_resid = residualize(H, Z, Z_pinv)
        d_real = fe_regression_direction(score_resid, H_resid)
        d_shuffled = fe_regression_direction(shuffled_score_resid, H_resid)

        real_norm = float(np.linalg.norm(d_real))
        d_randoms = []
        for seed_i in range(int(config["n_random_directions"])):
            rand_rng = np.random.default_rng(int(config["seed"]) * 1000 + seed_i)
            v = rand_rng.normal(size=H.shape[1])
            v = v / np.linalg.norm(v) * real_norm
            d_randoms.append(v)

        directions_payload[f"real_{L}"] = d_real
        directions_payload[f"shuffled_{L}"] = d_shuffled
        directions_payload[f"reference_{L}"] = reference
        for seed_i, v in enumerate(d_randoms):
            directions_payload[f"random{seed_i}_{L}"] = v

        metadata_rows.append(
            {
                "layer": L,
                "real_direction_norm": real_norm,
                "shuffled_direction_norm": float(np.linalg.norm(d_shuffled)),
                "reference_norm": float(np.linalg.norm(reference)),
                "n_train_pairs": int(len(train_meta)),
                "score_resid_var": float(score_resid @ score_resid / len(score_resid)),
            }
        )
        print(f"[direction] layer={L:2d} ||d_real||={real_norm:.4f} ||d_shuffled||={metadata_rows[-1]['shuffled_direction_norm']:.4f} "
              f"||reference||={metadata_rows[-1]['reference_norm']:.4f}")

    npz_path = output_dir / "directions.npz"
    np.savez_compressed(npz_path, **directions_payload)
    meta_out_path = output_dir / "direction_metadata.json"
    with meta_out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate_layers": CANDIDATE_LAYERS,
                "n_random_directions": int(config["n_random_directions"]),
                "n_shuffled_directions": 1,
                "per_layer": metadata_rows,
                "estimator": (
                    "d_L = (score_resid^T H_resid) / (score_resid^T score_resid), "
                    "score and H both residualized against image+target FE dummies "
                    "(train split only). reference_L = raw (non-residualized) train-split "
                    "mean activation at layer L. See audit Sec.4 for the full derivation."
                ),
            },
            f,
            indent=2,
        )
    print(f"[direction] wrote {npz_path} and {meta_out_path}")


if __name__ == "__main__":
    main()
