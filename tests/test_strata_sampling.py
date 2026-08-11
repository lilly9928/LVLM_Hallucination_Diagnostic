import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.strata_sampling import Candidate, build_candidates, split_strata

# Categories: A=1, B=2, C=3, D=4 -> matrix index 0,1,2,3
CATEGORY_INDEX = {1: 0, 2: 1, 3: 2, 4: 3}

JOINT_COUNTS = np.array(
    [
        [0, 50, 20, 8],
        [50, 0, 5, 8],
        [20, 5, 0, 15],
        [8, 8, 15, 0],
    ]
)

PMI = np.array(
    [
        [np.nan, 0.0, 1.0, 2.0],
        [0.0, np.nan, 9.0, 3.0],
        [1.0, 9.0, np.nan, 4.0],
        [2.0, 3.0, 4.0, np.nan],
    ]
)

IMAGE_CATEGORIES = {
    100: {1, 2},  # present A, B
    101: {3},  # present C only
}


class TestBuildCandidates(unittest.TestCase):
    def test_excludes_present_categories_and_low_support_terms(self) -> None:
        candidates = build_candidates(
            IMAGE_CATEGORIES, [1, 2, 3, 4], CATEGORY_INDEX, PMI, JOINT_COUNTS, min_support=10
        )
        by_key = {(c.image_id, c.category_id): c for c in candidates}

        # image100: A,B present -> only C,D are candidates. D has no present term
        # meeting min_support (counts 8 and 8), so it must be dropped entirely.
        self.assertIn((100, 3), by_key)
        self.assertNotIn((100, 4), by_key)
        self.assertAlmostEqual(by_key[(100, 3)].score, 1.0, places=6)
        self.assertEqual(by_key[(100, 3)].n_pmi_terms_used, 1)

        # image101: C present -> A,B,D are candidates. B's only term has count=5 (<10)
        # so it must be dropped entirely; A and D each have one qualifying term.
        self.assertIn((101, 1), by_key)
        self.assertNotIn((101, 2), by_key)
        self.assertIn((101, 4), by_key)
        self.assertAlmostEqual(by_key[(101, 1)].score, 1.0, places=6)
        self.assertAlmostEqual(by_key[(101, 4)].score, 4.0, places=6)

        self.assertEqual(len(candidates), 3)

    def test_skips_images_with_empty_present_set(self) -> None:
        candidates = build_candidates(
            {200: set()}, [1, 2, 3, 4], CATEGORY_INDEX, PMI, JOINT_COUNTS, min_support=10
        )
        self.assertEqual(candidates, [])


class TestSplitStrata(unittest.TestCase):
    def test_top_and_bottom_thirds_are_correctly_separated(self) -> None:
        candidates = [Candidate(image_id=i, category_id=1, score=float(i), n_present=1, n_pmi_terms_used=1) for i in range(1, 101)]
        treatment, control, info = split_strata(candidates, lower_pct=33.0, upper_pct=67.0)

        self.assertTrue(all(c.score >= info["high_cut"] for c in treatment))
        self.assertTrue(all(c.score <= info["low_cut"] for c in control))
        self.assertGreater(info["n_middle_dropped"], 0)
        self.assertEqual(info["n_middle_dropped"], len(candidates) - len(treatment) - len(control))


if __name__ == "__main__":
    unittest.main()
