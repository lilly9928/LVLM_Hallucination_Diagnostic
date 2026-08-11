import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.matching import MatchUnit, balance_table, coarsened_exact_match


def _make_units(rng, n: int, freq_alpha: float, freq_beta: float) -> list[MatchUnit]:
    freqs = rng.beta(freq_alpha, freq_beta, n)
    areas = rng.uniform(0, 1, n)
    clip_sims = rng.uniform(0, 1, n)
    return [
        MatchUnit(
            image_id=i,
            category_id=0,
            score=float(freqs[i]),
            freq=float(freqs[i]),
            area=float(areas[i]),
            clip_sim=float(clip_sims[i]),
        )
        for i in range(n)
    ]


class TestCoarsenedExactMatching(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        # Deliberate confound: treatment skews toward high marginal_freq, control
        # toward low -- CEM must trim to the overlapping region to fix this.
        self.treatment = _make_units(rng, 300, freq_alpha=2, freq_beta=1)
        self.control = _make_units(rng, 300, freq_alpha=1, freq_beta=2)

    def test_matching_reduces_frequency_imbalance(self) -> None:
        before = {row["covariate"]: row for row in balance_table(self.treatment, self.control)}

        result = coarsened_exact_match(self.treatment, self.control, freq_bins=4, area_bins=4, clip_bins=5)
        matched_treatment = [p[0] for p in result["pairs"]]
        matched_control = [p[1] for p in result["pairs"]]
        after = {row["covariate"]: row for row in balance_table(matched_treatment, matched_control)}

        self.assertGreater(result["n_matched_pairs"], 0)
        self.assertLess(abs(after["marginal_freq"]["smd"]), abs(before["marginal_freq"]["smd"]))
        self.assertLess(abs(after["marginal_freq"]["smd"]), 0.25)

    def test_no_unit_is_reused_across_pairs(self) -> None:
        result = coarsened_exact_match(self.treatment, self.control, freq_bins=4, area_bins=4, clip_bins=5)
        treat_ids = [id(p[0]) for p in result["pairs"]]
        control_ids = [id(p[1]) for p in result["pairs"]]
        self.assertEqual(len(treat_ids), len(set(treat_ids)))
        self.assertEqual(len(control_ids), len(set(control_ids)))


if __name__ == "__main__":
    unittest.main()
