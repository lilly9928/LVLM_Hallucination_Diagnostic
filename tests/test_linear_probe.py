import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.linear_probe import (
    ProbeExample,
    bootstrap_excess_auc_ci,
    build_feature_matrices,
    fit_probe,
    split_examples_by_image,
)

CATEGORY_IDS = [1, 2, 3, 4]


class TestBuildFeatureMatrices(unittest.TestCase):
    def test_onehot_multihot_and_clip_feature_shapes_and_values(self) -> None:
        examples = [
            ProbeExample(image_id=100, category_id=1, label=1),
            ProbeExample(image_id=101, category_id=2, label=0),
        ]
        present = {100: {2, 3}, 101: {1}}
        clip_embeddings = {100: np.array([1.0, 0.0]), 101: np.array([0.0, 1.0])}

        result = build_feature_matrices(examples, CATEGORY_IDS, present, clip_embeddings)
        baseline_X, full_X, labels = result["baseline_X"], result["full_X"], result["labels"]

        self.assertEqual(baseline_X.shape, (2, 8))  # 4 onehot(A) + 4 multihot(Y)
        self.assertEqual(full_X.shape, (2, 6))  # 4 onehot(A) + 2 clip dims
        np.testing.assert_array_equal(labels, [1, 0])

        # example 0: A=category_id 1 -> index 0; present={2,3} -> indices 1,2
        np.testing.assert_array_equal(baseline_X[0, :4], [1, 0, 0, 0])
        np.testing.assert_array_equal(baseline_X[0, 4:], [0, 1, 1, 0])
        np.testing.assert_array_equal(full_X[0, :4], [1, 0, 0, 0])
        np.testing.assert_array_equal(full_X[0, 4:], [1.0, 0.0])


class TestSplitExamplesByImage(unittest.TestCase):
    def test_no_image_appears_on_both_sides(self) -> None:
        examples = [ProbeExample(image_id=img_id, category_id=1, label=0) for img_id in range(20) for _ in range(3)]
        train_idx, test_idx = split_examples_by_image(examples, test_frac=0.3, rng=np.random.default_rng(0))
        train_images = {examples[i].image_id for i in train_idx}
        test_images = {examples[i].image_id for i in test_idx}
        self.assertEqual(train_images & test_images, set())
        self.assertEqual(len(train_idx) + len(test_idx), len(examples))

    def test_test_fraction_is_approximately_respected_at_image_level(self) -> None:
        examples = [ProbeExample(image_id=img_id, category_id=1, label=0) for img_id in range(100)]
        train_idx, test_idx = split_examples_by_image(examples, test_frac=0.2, rng=np.random.default_rng(0))
        self.assertEqual(len(test_idx), 20)
        self.assertEqual(len(train_idx), 80)


class TestBootstrapExcessAucCi(unittest.TestCase):
    def test_confidently_positive_when_full_model_is_clearly_better(self) -> None:
        rng = np.random.default_rng(0)
        n = 500
        y = rng.integers(0, 2, size=n)
        full_probs = np.where(y == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
        baseline_probs = rng.uniform(0.0, 1.0, n)  # chance-level
        result = bootstrap_excess_auc_ci(y, baseline_probs, full_probs, n_boot=1000, rng=rng)
        self.assertGreater(result["ci_lower"], 0.0)


class TestProbeRecoversExcessSignal(unittest.TestCase):
    def test_full_model_beats_baseline_when_only_clip_feature_carries_signal(self) -> None:
        rng = np.random.default_rng(0)
        n_images = 200
        labels = rng.integers(0, 2, size=n_images)
        # clip_feat cleanly separates by label; multihot(Y) is pure independent noise.
        examples, present, clip_emb = [], {}, {}
        for img_id in range(n_images):
            examples.append(ProbeExample(image_id=img_id, category_id=1, label=int(labels[img_id])))
            present[img_id] = set(rng.choice(CATEGORY_IDS, size=2, replace=False))  # unrelated to label
            base = np.array([5.0, 0.0]) if labels[img_id] == 1 else np.array([-5.0, 0.0])
            clip_emb[img_id] = base + rng.normal(scale=0.5, size=2)

        mats = build_feature_matrices(examples, CATEGORY_IDS, present, clip_emb)
        train_idx, test_idx = split_examples_by_image(examples, test_frac=0.3, rng=np.random.default_rng(1))

        from sklearn.metrics import roc_auc_score

        baseline_clf = fit_probe(mats["baseline_X"][train_idx], mats["labels"][train_idx])
        full_clf = fit_probe(mats["full_X"][train_idx], mats["labels"][train_idx])
        y_test = mats["labels"][test_idx]
        baseline_auc = roc_auc_score(y_test, baseline_clf.predict_proba(mats["baseline_X"][test_idx])[:, 1])
        full_auc = roc_auc_score(y_test, full_clf.predict_proba(mats["full_X"][test_idx])[:, 1])

        self.assertLess(baseline_auc, 0.65)  # Y is noise -> near chance
        self.assertGreater(full_auc, 0.95)  # CLIP feature cleanly separates the classes


if __name__ == "__main__":
    unittest.main()
