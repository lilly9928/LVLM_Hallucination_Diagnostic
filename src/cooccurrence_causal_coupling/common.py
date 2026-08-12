"""Shared utilities for Experiment 4 (causal coupling via activation patching).

Reuses cooc_diagnostic.llava_runtime for the model/prompt/decision-point
machinery (same as Stage 9/10/11) and re-implements only the FE-projection
and hook logic that Experiment 4 newly needs -- documented in
outputs/cooccurrence_causal_coupling/audit/repository_audit.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

CANDIDATE_LAYERS = [3, 8, 11, 12, 13, 16, 20, 24, 28]


def image_level_split(image_ids: list[int], seed: int = 42, n_train: int = 30, n_val: int = 10) -> dict[int, str]:
    """Deterministic image-level train/val/test split (never split by row --
    see audit doc Sec.5: this prevents a target-level leakage where the same
    image contributes to both direction estimation and causal evaluation."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(sorted(image_ids))
    assignment = {}
    for i, image_id in enumerate(shuffled):
        if i < n_train:
            assignment[int(image_id)] = "train"
        elif i < n_train + n_val:
            assignment[int(image_id)] = "val"
        else:
            assignment[int(image_id)] = "test"
    return assignment


def build_fe_projection(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Same FE design as Stage10/11: image + target dummies + intercept.
    Z_pinv precomputed once; residualize() below reuses it for any
    same-length vector OR matrix (n x d) aligned with df's row order."""
    image_dummies = pd.get_dummies(df["image_id"], prefix="img", drop_first=True).to_numpy(dtype=float)
    target_dummies = pd.get_dummies(df["target"], prefix="tgt", drop_first=True).to_numpy(dtype=float)
    Z = np.column_stack([np.ones(len(df)), image_dummies, target_dummies])
    Z_pinv = np.linalg.pinv(Z)
    return Z, Z_pinv


def residualize(V: np.ndarray, Z: np.ndarray, Z_pinv: np.ndarray) -> np.ndarray:
    """V: (n,) or (n, d). Returns V with the FE design Z projected out."""
    return V - Z @ (Z_pinv @ V)


def fe_regression_direction(score_resid: np.ndarray, H_resid: np.ndarray) -> np.ndarray:
    """Vector-valued FWL slope: d = (score_resid^T H_resid) / (score_resid^T score_resid).
    H_resid is (n, hidden_dim); returns (hidden_dim,). This is the exact
    multivariate analogue of Stage10/11's scalar `beta` -- see audit Sec.4."""
    denom = float(score_resid @ score_resid)
    return (score_resid @ H_resid) / denom


def get_decoder_layers(model):
    """model.language_model.layers -- verified working against the
    installed transformers version (see audit Sec.2); do NOT use
    model.language_model.model.layers (confirmed broken in this environment)."""
    return model.language_model.layers


class ResidualProjectionHook:
    """Mean-referenced projection-removal hook (audit Sec.4):
        h' = h - lambda * ((h - reference) . d_hat) * d_hat
    applied ONLY at the last sequence position (decision position), on the
    OUTPUT of language_model.layers[layer_idx - 1] (== hidden_states[layer_idx]
    in Stage 11's indexing convention). Skips cached single-token decode steps
    (seq_len == 1), matching Method_DSTR's ResidualIntervention precedent.
    """

    def __init__(self, model, layer_idx: int, direction: torch.Tensor, reference: torch.Tensor):
        self.layer_idx = layer_idx
        self.direction_hat = direction / direction.norm()
        self.reference = reference
        self.lam = 0.0
        self.armed = False
        self.last_projection_before = None
        self.last_projection_after = None
        module = get_decoder_layers(model)[layer_idx - 1]
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.armed:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] <= 1:
            return output
        d = self.direction_hat.to(hidden.dtype).to(hidden.device)
        ref = self.reference.to(hidden.dtype).to(hidden.device)
        modified = hidden.clone()
        current = modified[:, -1, :]
        centered = current - ref
        coeff = (centered * d).sum(dim=-1, keepdim=True)
        self.last_projection_before = float(coeff.mean().item())
        current_after = current - self.lam * coeff * d
        coeff_after = ((current_after - ref) * d).sum(dim=-1, keepdim=True)
        self.last_projection_after = float(coeff_after.mean().item())
        modified[:, -1, :] = current_after
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def remove(self):
        self.handle.remove()


@dataclass(frozen=True)
class DirectionMetadata:
    layer: int
    direction_norm: float
    reference_norm: float
    n_train_pairs: int
