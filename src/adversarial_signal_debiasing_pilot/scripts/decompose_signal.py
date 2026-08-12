"""Part IV (+ optional Part VIII): PCA/SVD decomposition of the Layer-19
adversarial shift, fit on the DEV portion of TRAIN G10 only (never on VAL or
on the final CLEAN TEST split -- see Part VII's component-selection
discipline in configs/data.yaml's dev_val_split).

D = [delta_h_1, ..., delta_h_N_dev]^T ,  D = U Sigma V^T

K=10 components requested (configs/decomposition.yaml), reduced to rank(D)
if smaller. PC1 is NOT assumed spurious merely because it explains the most
variance -- that judgment is made downstream by evaluate_components.py's
functional selectivity test, not here.

Optionally also fits a PLS decomposition (X=delta_h, y=delta_s_ball, both DEV
only) if invoked with --with-pls; this is Part VIII, run only if the PCA
components tested by evaluate_components.py show poor selectivity.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/decompose_signal.py \
        --data-config configs/data.yaml --decomp-config configs/decomposition.yaml [--with-pls]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--decomp-config", type=Path, required=True)
    parser.add_argument("--with-pls", action="store_true", help="Also fit PLS (Part VIII, only if PCA selectivity is weak)")
    args = parser.parse_args()
    with args.data_config.open("r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with args.decomp_config.open("r", encoding="utf-8") as f:
        decomp_config = yaml.safe_load(f)

    out_dir = Path(data_config["output_dir"])
    cache_dir = out_dir / "cached_activations"
    decomp_dir = out_dir / "decomposition"
    decomp_dir.mkdir(parents=True, exist_ok=True)

    cached = torch.load(cache_dir / "layer19_delta_h.pt")
    delta_h = cached["delta_h"].numpy()  # (N, hidden_dim)
    image_ids = cached["image_ids"]

    dev_val_rows = list(csv.DictReader((out_dir / "data" / "train_dev_val_split.csv").open("r", encoding="utf-8")))
    dev_ids = {int(r["image_id"]) for r in dev_val_rows if r["role"] == "G10_forget" and r["dev_val"] == "dev"}
    val_ids = {int(r["image_id"]) for r in dev_val_rows if r["role"] == "G10_forget" and r["dev_val"] == "val"}

    dev_mask = np.array([iid in dev_ids for iid in image_ids])
    val_mask = np.array([iid in val_ids for iid in image_ids])
    assert dev_mask.sum() + val_mask.sum() == len(image_ids), "every G10 train image must be dev or val, exclusively"

    D_dev = delta_h[dev_mask]
    print(f"[decompose] D_dev shape: {D_dev.shape} (fit only on DEV, n_val_held_out={val_mask.sum()})")

    k_requested = int(decomp_config["pca"]["k"])
    rank = min(D_dev.shape[0] - 1, D_dev.shape[1])  # PCA centers the data, so max useful rank = n_samples - 1
    k = min(k_requested, rank)
    if k < k_requested:
        print(f"[decompose] rank(D_dev)={rank} < requested K={k_requested}; using K={k}")

    pca = PCA(n_components=k, random_state=int(data_config["seed"]))
    scores_dev = pca.fit_transform(D_dev)  # (n_dev, k)
    components = pca.components_  # (k, hidden_dim)
    explained_variance_ratio = pca.explained_variance_ratio_

    torch.save(
        {
            "components": torch.from_numpy(components).float(),  # (k, hidden_dim), each row is a unit vector u_k
            "mean": torch.from_numpy(pca.mean_).float(),
            "k": k,
            "fit_on": "TRAIN_G10_dev",
            "n_dev": int(dev_mask.sum()),
        },
        decomp_dir / "pca_components.pt",
    )

    # Project ALL train G10 (dev+val) onto the dev-fit components, for inspection.
    scores_all = pca.transform(delta_h)
    with (decomp_dir / "pca_scores.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "dev_val"] + [f"pc{i+1}" for i in range(k)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, iid in enumerate(image_ids):
            dv = "dev" if dev_mask[i] else "val"
            row = {"image_id": iid, "dev_val": dv}
            row.update({f"pc{j+1}": float(scores_all[i, j]) for j in range(k)})
            writer.writerow(row)

    with (decomp_dir / "explained_variance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "explained_variance_ratio", "cumulative_variance_ratio"])
        writer.writeheader()
        cumulative = 0.0
        for i, evr in enumerate(explained_variance_ratio):
            cumulative += float(evr)
            writer.writerow({"component": f"PC{i+1}", "explained_variance_ratio": float(evr), "cumulative_variance_ratio": cumulative})

    print(f"[decompose] K={k} components, explained variance ratio: {[round(float(v), 4) for v in explained_variance_ratio]}")
    print(f"[decompose] cumulative variance (first {k}): {float(explained_variance_ratio.sum()):.4f}")
    print(f"[decompose] wrote {decomp_dir / 'pca_components.pt'}, pca_scores.csv, explained_variance.csv")

    if args.with_pls:
        from sklearn.cross_decomposition import PLSRegression

        metadata_rows = {int(r["image_id"]): r for r in csv.DictReader((cache_dir / "layer19_metadata.csv").open("r", encoding="utf-8"))}
        y_dev = np.array([float(metadata_rows[iid]["delta_s_ball"]) for iid, m in zip(image_ids, dev_mask) if m])

        n_pls = int(decomp_config["pls"]["n_components"])
        pls = PLSRegression(n_components=n_pls)
        pls.fit(D_dev, y_dev)
        # PLS loading/weight vectors are not orthonormal by construction; normalize
        # to unit vectors so the intervention test's proj_u(h) is well-defined.
        pls_directions = pls.x_weights_.T  # (n_pls, hidden_dim)
        pls_directions = pls_directions / np.linalg.norm(pls_directions, axis=1, keepdims=True)

        torch.save(
            {
                "components": torch.from_numpy(pls_directions).float(),
                "k": n_pls,
                "fit_on": "TRAIN_G10_dev",
                "target": "delta_s_ball",
            },
            decomp_dir / "pls_components.pt",
        )
        print(f"[decompose] PLS: fit {n_pls} components on {len(y_dev)} DEV images, wrote {decomp_dir / 'pls_components.pt'}")


if __name__ == "__main__":
    main()
