import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.cooccurrence_stats import compute_cooccurrence_stats, get_top_bottom_pairs

# Category ids: A=1, B=2, C=3
# Images: {A,B}, {A,B}, {A}, {B,C}, {} (no objects)
CATEGORY_IDS = [1, 2, 3]
CATEGORY_NAMES = {1: "A", 2: "B", 3: "C"}
IMAGE_CATEGORIES = {
    1: {1, 2},
    2: {1, 2},
    3: {1},
    4: {2, 3},
    5: set(),
}


class TestCooccurrenceStats(unittest.TestCase):
    def setUp(self) -> None:
        self.stats = compute_cooccurrence_stats(CATEGORY_IDS, IMAGE_CATEGORIES)
        self.idx = {cid: i for i, cid in enumerate(CATEGORY_IDS)}

    def test_marginal_counts(self) -> None:
        self.assertEqual(self.stats.n_images, 5)
        self.assertEqual(int(self.stats.marginal_counts[self.idx[1]]), 3)  # A
        self.assertEqual(int(self.stats.marginal_counts[self.idx[2]]), 3)  # B
        self.assertEqual(int(self.stats.marginal_counts[self.idx[3]]), 1)  # C

    def test_joint_counts_symmetric(self) -> None:
        a, b, c = self.idx[1], self.idx[2], self.idx[3]
        self.assertEqual(int(self.stats.joint_counts[a, b]), 2)
        self.assertEqual(int(self.stats.joint_counts[b, a]), 2)
        self.assertEqual(int(self.stats.joint_counts[a, c]), 0)
        self.assertEqual(int(self.stats.joint_counts[b, c]), 1)

    def test_pmi_matches_manual_computation(self) -> None:
        a, b, c = self.idx[1], self.idx[2], self.idx[3]
        # P(A)=0.6, P(B)=0.6, P(C)=0.2, P(A,B)=0.4, P(B,C)=0.2
        expected_pmi_ab = math.log(0.4 / (0.6 * 0.6))
        expected_pmi_bc = math.log(0.2 / (0.6 * 0.2))
        self.assertAlmostEqual(float(self.stats.pmi[a, b]), expected_pmi_ab, places=6)
        self.assertAlmostEqual(float(self.stats.pmi[b, c]), expected_pmi_bc, places=6)
        # symmetry
        self.assertAlmostEqual(float(self.stats.pmi[a, b]), float(self.stats.pmi[b, a]), places=6)

    def test_zero_joint_count_is_negative_infinity_pmi_and_zero_lift(self) -> None:
        a, c = self.idx[1], self.idx[3]
        self.assertTrue(math.isinf(float(self.stats.pmi[a, c])))
        self.assertLess(float(self.stats.pmi[a, c]), 0)
        self.assertEqual(float(self.stats.lift[a, c]), 0.0)

    def test_lift_equals_exp_pmi_when_finite(self) -> None:
        a, b = self.idx[1], self.idx[2]
        self.assertAlmostEqual(float(self.stats.lift[a, b]), math.exp(float(self.stats.pmi[a, b])), places=6)

    def test_conditional_is_asymmetric(self) -> None:
        b, c = self.idx[2], self.idx[3]
        # P(C|B) = joint(B,C)/marginal(B) = 1/3
        self.assertAlmostEqual(float(self.stats.conditional[b, c]), 1 / 3, places=6)
        # P(B|C) = joint(B,C)/marginal(C) = 1/1
        self.assertAlmostEqual(float(self.stats.conditional[c, b]), 1.0, places=6)

    def test_diagonal_is_nan(self) -> None:
        for i in range(len(CATEGORY_IDS)):
            self.assertTrue(math.isnan(float(self.stats.pmi[i, i])))
            self.assertTrue(math.isnan(float(self.stats.conditional[i, i])))

    def test_min_support_excludes_zero_and_low_count_pairs(self) -> None:
        result = get_top_bottom_pairs(self.stats, CATEGORY_NAMES, min_support=2, top_k=10)
        # Only A-B has joint_count >= 2; B-C (count=1) and A-C (count=0) excluded.
        eligible_pairs = {(r["category_a"], r["category_b"]) for r in result["top"]}
        self.assertEqual(eligible_pairs, {("A", "B")})
        self.assertEqual(result["n_pairs_total"], 3)
        self.assertEqual(result["n_pairs_excluded_by_min_support"], 2)

    def test_ranking_orders_by_pmi_descending_then_ascending(self) -> None:
        result = get_top_bottom_pairs(self.stats, CATEGORY_NAMES, min_support=1, top_k=10)
        # Eligible: A-B (pmi~0.105), B-C (pmi~0.511); A-C excluded (joint_count=0).
        top_pairs = [(r["category_a"], r["category_b"]) for r in result["top"]]
        bottom_pairs = [(r["category_a"], r["category_b"]) for r in result["bottom"]]
        self.assertEqual(top_pairs, [("B", "C"), ("A", "B")])
        self.assertEqual(bottom_pairs, [("A", "B"), ("B", "C")])
        self.assertEqual(result["n_pairs_excluded_by_min_support"], 1)


if __name__ == "__main__":
    unittest.main()
