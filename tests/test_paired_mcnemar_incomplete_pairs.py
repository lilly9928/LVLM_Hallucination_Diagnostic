import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.survival_analysis import paired_mcnemar_test


class TestPairedMcNemarIncompletePairs(unittest.TestCase):
    def test_drops_pairs_missing_one_arm_instead_of_crashing(self) -> None:
        df = pd.DataFrame(
            {
                "pair_id": [0, 0, 1, 1, 2],  # pair 2 only has a "treatment" row -- incomplete
                "arm": ["treatment", "control", "treatment", "control", "treatment"],
                "flag": [True, False, False, True, True],
            }
        )
        result = paired_mcnemar_test(df, "flag")
        self.assertEqual(result["n_pairs_incomplete_dropped"], 1)
        self.assertEqual(result["treatment_only_flag"] + result["control_only_flag"] + result["both_flag"] + result["both_not"], 2)

    def test_matches_prior_behavior_when_all_pairs_are_complete(self) -> None:
        df = pd.DataFrame(
            {
                "pair_id": [0, 0, 1, 1],
                "arm": ["treatment", "control", "treatment", "control"],
                "flag": [True, False, True, False],
            }
        )
        result = paired_mcnemar_test(df, "flag")
        self.assertEqual(result["n_pairs_incomplete_dropped"], 0)
        self.assertEqual(result["treatment_only_flag"], 2)
        self.assertEqual(result["control_only_flag"], 0)


if __name__ == "__main__":
    unittest.main()
