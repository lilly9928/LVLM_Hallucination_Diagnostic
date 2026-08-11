"""Stage 5: does the frozen CLIP visual feature carry information about which
absent categories are high-co-occurrence, BEYOND what the present-object set Y
already determines?

Two linear probes are trained on the identical (image, candidate category A)
task -- predict whether this pair falls in Stage 2's high-co-occurrence
(treatment) stratum vs. low-co-occurrence (control) stratum -- differing only
in what they see:
  - baseline: multi-hot(Y) + one-hot(A)        (symbolic present-object info)
  - full:     CLIP image embedding + one-hot(A) (raw visual feature)

Both models are given one-hot(A): "which category is being asked about" is the
task's query/specification, not predictive information under comparison -- the
comparison is symbolic-Y vs. raw-pixels for the SAME query. Since Z is a
(nonlinear, set-size-dependent) function of (A, Y) via mean-PMI, a plain linear
baseline over one-hot(A)+multi-hot(Y) cannot perfectly recover it (no A x Y
interaction terms), leaving genuine room for excess AUC to be informative
rather than tautologically zero or one.

Train/test is split by image (never by example), so no image's pixels or
category composition leaks between train and test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ProbeExample:
    image_id: int
    category_id: int
    label: int  # 1 = treatment (high co-occurrence absent), 0 = control


def build_feature_matrices(
    examples: list[ProbeExample],
    category_ids: list[int],
    image_present_categories: dict[int, set[int]],
    image_clip_embeddings: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    cat_index = {cid: i for i, cid in enumerate(category_ids)}
    k = len(category_ids)
    n = len(examples)
    clip_dim = next(iter(image_clip_embeddings.values())).shape[0]

    onehot_a = np.zeros((n, k), dtype=np.float32)
    multihot_y = np.zeros((n, k), dtype=np.float32)
    clip_feat = np.zeros((n, clip_dim), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    for i, ex in enumerate(examples):
        onehot_a[i, cat_index[ex.category_id]] = 1.0
        for y in image_present_categories[ex.image_id]:
            if y in cat_index:
                multihot_y[i, cat_index[y]] = 1.0
        clip_feat[i] = image_clip_embeddings[ex.image_id]
        labels[i] = ex.label

    return {
        "baseline_X": np.concatenate([onehot_a, multihot_y], axis=1),
        "full_X": np.concatenate([onehot_a, clip_feat], axis=1),
        "labels": labels,
    }


def split_examples_by_image(examples: list[ProbeExample], test_frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_indices, test_indices) into `examples`, split by image_id
    so no image contributes to both sides."""
    image_ids = sorted({ex.image_id for ex in examples})
    rng.shuffle(image_ids)
    n_test = int(round(len(image_ids) * test_frac))
    test_images = set(image_ids[:n_test])

    example_image_ids = np.array([ex.image_id for ex in examples])
    is_test = np.isin(example_image_ids, list(test_images))
    return np.where(~is_test)[0], np.where(is_test)[0]


def fit_probe(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegressionCV:
    clf = LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, scoring="roc_auc", n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def bootstrap_excess_auc_ci(y_test: np.ndarray, baseline_probs: np.ndarray, full_probs: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    n = len(y_test)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_test[idx])) < 2:
            continue
        auc_baseline = roc_auc_score(y_test[idx], baseline_probs[idx])
        auc_full = roc_auc_score(y_test[idx], full_probs[idx])
        diffs.append(auc_full - auc_baseline)
    diffs.sort()
    n_used = len(diffs)
    return {
        "n_boot_used": n_used,
        "ci_lower": diffs[int(0.025 * n_used)],
        "ci_upper": diffs[min(int(0.975 * n_used), n_used - 1)],
    }
