import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.sanity_checks import (
    compare_attack_vs_random_control,
    summarize_attack_success_rate,
    summarize_present_object_baseline,
    summarize_random_noise_control,
)


class TestSanityCheckSummaries(unittest.TestCase):
    def test_present_object_baseline_passes_when_yes_rate_is_high(self) -> None:
        records = [{"is_yes": True}] * 9 + [{"is_yes": False}] * 1
        result = summarize_present_object_baseline(records, min_yes_rate=0.8)
        self.assertAlmostEqual(result["yes_rate"], 0.9)
        self.assertTrue(result["passed"])

    def test_present_object_baseline_fails_when_yes_rate_is_low(self) -> None:
        records = [{"is_yes": True}] * 5 + [{"is_yes": False}] * 5
        result = summarize_present_object_baseline(records, min_yes_rate=0.8)
        self.assertAlmostEqual(result["yes_rate"], 0.5)
        self.assertFalse(result["passed"])

    def test_attack_success_rate_thresholding(self) -> None:
        records = [{"flipped": True}] * 96 + [{"flipped": False}] * 4
        result = summarize_attack_success_rate(records, epsilon_label="16/255", min_success_rate=0.95)
        self.assertAlmostEqual(result["success_rate"], 0.96)
        self.assertTrue(result["passed"])

    def test_random_noise_control_fails_when_flip_rate_too_high(self) -> None:
        records = [{"flipped": True}] * 40 + [{"flipped": False}] * 60
        result = summarize_random_noise_control(records, epsilon_label="16/255", max_flip_rate=0.3)
        self.assertAlmostEqual(result["flip_rate"], 0.4)
        self.assertFalse(result["passed"])

    def test_compare_attack_vs_random_control_requires_a_large_gap(self) -> None:
        attack = summarize_attack_success_rate([{"flipped": True}] * 98 + [{"flipped": False}] * 2, "16/255")
        random_low = summarize_random_noise_control([{"flipped": True}] * 5 + [{"flipped": False}] * 95, "16/255")
        comparison = compare_attack_vs_random_control(attack, random_low, min_gap=0.3)
        self.assertAlmostEqual(comparison["gap"], 0.93)
        self.assertTrue(comparison["passed"])

        random_high = summarize_random_noise_control([{"flipped": True}] * 90 + [{"flipped": False}] * 10, "16/255")
        comparison_bad = compare_attack_vs_random_control(attack, random_high, min_gap=0.3)
        self.assertFalse(comparison_bad["passed"])


if __name__ == "__main__":
    unittest.main()
