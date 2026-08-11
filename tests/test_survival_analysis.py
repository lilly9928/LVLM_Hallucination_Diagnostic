import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.survival_analysis import (
    build_survival_frame,
    holm_correction,
    mcnemar_already_yes_test,
    paired_bootstrap_median_diff,
    stratified_cox_test,
    weibull_aft_time_ratio,
)

EPSILON_MAX = 32 / 255


def _rows(n_pairs, treat_status_fn, control_status_fn, treat_eps_fn, control_eps_fn):
    rows = []
    for i in range(n_pairs):
        rows.append({"pair_id": i, "arm": "treatment", "status": treat_status_fn(i), "epsilon_star": treat_eps_fn(i)})
        rows.append({"pair_id": i, "arm": "control", "status": control_status_fn(i), "epsilon_star": control_eps_fn(i)})
    return rows


class TestBuildSurvivalFrame(unittest.TestCase):
    def test_status_to_duration_and_event_mapping(self) -> None:
        rows = [
            {"pair_id": 1, "arm": "treatment", "status": "flipped", "epsilon_star": 0.01},
            {"pair_id": 1, "arm": "control", "status": "already_yes", "epsilon_star": 0.0},
            {"pair_id": 2, "arm": "treatment", "status": "censored", "epsilon_star": None},
            {"pair_id": 2, "arm": "control", "status": "flipped", "epsilon_star": 0.02},
        ]
        df = build_survival_frame(rows, epsilon_max=EPSILON_MAX)
        flipped_row = df[(df.pair_id == 1) & (df.arm == "treatment")].iloc[0]
        self.assertAlmostEqual(flipped_row["duration"], 0.01)
        self.assertEqual(flipped_row["event_observed"], 1)
        self.assertFalse(flipped_row["already_yes"])

        already_yes_row = df[(df.pair_id == 1) & (df.arm == "control")].iloc[0]
        self.assertAlmostEqual(already_yes_row["duration"], 0.0)
        self.assertEqual(already_yes_row["event_observed"], 1)
        self.assertTrue(already_yes_row["already_yes"])

        censored_row = df[(df.pair_id == 2) & (df.arm == "treatment")].iloc[0]
        self.assertAlmostEqual(censored_row["duration"], EPSILON_MAX)
        self.assertEqual(censored_row["event_observed"], 0)


class TestHolmCorrection(unittest.TestCase):
    def test_matches_hand_computation(self) -> None:
        adjusted = holm_correction({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(adjusted["a"], 0.02)
        self.assertAlmostEqual(adjusted["b"], 0.04)

    def test_enforces_monotonicity_across_ranks(self) -> None:
        # raw Holm value for 'b' (rank 2, m=3) is 1*0.06=0.06, which is LESS than
        # 'a's adjusted value (0.10) -- Holm must clip it up to stay monotonic.
        adjusted = holm_correction({"a": 0.05, "b": 0.06, "c": 0.001})
        self.assertAlmostEqual(adjusted["c"], 0.003)
        self.assertAlmostEqual(adjusted["a"], 0.10)
        self.assertAlmostEqual(adjusted["b"], 0.10)  # clipped up, not 0.06

    def test_never_exceeds_one(self) -> None:
        adjusted = holm_correction({"a": 0.9, "b": 0.8})
        self.assertLessEqual(adjusted["a"], 1.0)
        self.assertLessEqual(adjusted["b"], 1.0)


class TestMcNemarAlreadyYes(unittest.TestCase):
    def test_symmetric_discordance_gives_high_p_value(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "pair_id": [0, 0, 1, 1, 2, 2, 3, 3],
                "arm": ["treatment", "control"] * 4,
                "already_yes": [True, False, False, True, True, False, False, True],
            }
        )
        result = mcnemar_already_yes_test(df)
        self.assertEqual(result["treatment_only_already_yes"], 2)
        self.assertEqual(result["control_only_already_yes"], 2)
        self.assertGreater(result["p_value"], 0.9)

    def test_strong_asymmetric_discordance_gives_low_p_value(self) -> None:
        import pandas as pd

        n = 20
        pair_ids = list(range(n))
        rows = []
        for i in pair_ids:
            # treatment always already_yes, control never -- maximally discordant.
            rows.append({"pair_id": i, "arm": "treatment", "already_yes": True})
            rows.append({"pair_id": i, "arm": "control", "already_yes": False})
        df = pd.DataFrame(rows)
        result = mcnemar_already_yes_test(df)
        self.assertEqual(result["treatment_only_already_yes"], n)
        self.assertEqual(result["control_only_already_yes"], 0)
        self.assertLess(result["p_value"], 0.001)


class TestStratifiedCoxSmoke(unittest.TestCase):
    def test_recovers_a_clear_group_effect(self) -> None:
        rng = np.random.default_rng(0)
        n_pairs = 100
        # Treatment durations systematically smaller (scale 0.05) than control (scale 0.2).
        treat_eps = rng.exponential(scale=0.05, size=n_pairs)
        control_eps = rng.exponential(scale=0.2, size=n_pairs)
        rows = _rows(
            n_pairs,
            treat_status_fn=lambda i: "flipped",
            control_status_fn=lambda i: "flipped",
            treat_eps_fn=lambda i: float(treat_eps[i]),
            control_eps_fn=lambda i: float(control_eps[i]),
        )
        df = build_survival_frame(rows, epsilon_max=EPSILON_MAX)
        result = stratified_cox_test(df)
        # Treatment flips at smaller epsilon -> higher hazard -> HR > 1.
        self.assertGreater(result["hazard_ratio"], 1.0)
        self.assertLess(result["p_value"], 0.01)
        self.assertGreater(result["ci_lower"], 1.0)


class TestWeibullAftTimeRatio(unittest.TestCase):
    def test_handles_zero_durations_via_epsilon_floor_shift(self) -> None:
        # Half the pairs are already_yes (duration=0) -- must not raise, and
        # the fitted direction should still favor treatment (smaller epsilon).
        rng = np.random.default_rng(0)
        n_pairs = 60
        rows = []
        for i in range(n_pairs):
            treat_already_yes = i < 20
            rows.append(
                {
                    "pair_id": i,
                    "arm": "treatment",
                    "status": "already_yes" if treat_already_yes else "flipped",
                    "epsilon_star": 0.0 if treat_already_yes else float(rng.exponential(0.05)),
                }
            )
            rows.append({"pair_id": i, "arm": "control", "status": "flipped", "epsilon_star": float(rng.exponential(0.2))})
        df = build_survival_frame(rows, epsilon_max=EPSILON_MAX)
        result = weibull_aft_time_ratio(df, epsilon_floor=1e-5)
        self.assertLess(result["time_ratio"], 1.0)  # treatment epsilon* scaled down relative to control
        self.assertLess(result["p_value"], 0.05)


class TestPairedBootstrapMedianDiff(unittest.TestCase):
    def test_excludes_pairs_with_a_censored_side(self) -> None:
        rows = _rows(
            10,
            treat_status_fn=lambda i: "flipped",
            control_status_fn=lambda i: "censored" if i < 3 else "flipped",
            treat_eps_fn=lambda i: 0.01,
            control_eps_fn=lambda i: (EPSILON_MAX if i < 3 else 0.05),
        )
        df = build_survival_frame(rows, epsilon_max=EPSILON_MAX)
        result = paired_bootstrap_median_diff(df, n_boot=200, rng=np.random.default_rng(0))
        self.assertEqual(result["n_pairs_excluded_censored"], 3)
        self.assertEqual(result["n_pairs_used"], 7)

    def test_observed_diff_matches_manual_medians(self) -> None:
        treat_vals = [0.01, 0.02, 0.03, 0.04, 0.05]
        control_vals = [0.05, 0.06, 0.07, 0.08, 0.09]
        rows = _rows(
            5,
            treat_status_fn=lambda i: "flipped",
            control_status_fn=lambda i: "flipped",
            treat_eps_fn=lambda i: treat_vals[i],
            control_eps_fn=lambda i: control_vals[i],
        )
        df = build_survival_frame(rows, epsilon_max=EPSILON_MAX)
        result = paired_bootstrap_median_diff(df, n_boot=500, rng=np.random.default_rng(0))
        self.assertAlmostEqual(result["observed_median_diff"], 0.03 - 0.07, places=6)
        self.assertLess(result["ci_upper"], 0)  # treatment consistently smaller -> whole CI negative


if __name__ == "__main__":
    unittest.main()
