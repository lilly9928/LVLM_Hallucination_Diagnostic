import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.epsilon_star import find_epsilon_star

EPSILON_MAX = 32 / 255
EPS0 = EPSILON_MAX / 8192


def step_attack(true_threshold: float):
    def attack_at_epsilon(epsilon: float) -> dict:
        return {"flipped": epsilon >= true_threshold, "response_text": "Yes" if epsilon >= true_threshold else "No"}

    return attack_at_epsilon


class TestFindEpsilonStar(unittest.TestCase):
    def test_already_yes_short_circuits_with_zero_attack_calls(self) -> None:
        result = find_epsilon_star(
            attack_at_epsilon=step_attack(0.05),
            baseline_response={"flipped": True, "response_text": "Yes"},
            epsilon_max=EPSILON_MAX,
            eps0=EPS0,
        )
        self.assertEqual(result.status, "already_yes")
        self.assertEqual(result.epsilon_star, 0.0)
        self.assertEqual(result.n_attack_calls, 0)

    def test_never_flips_is_recorded_as_censored_not_dropped(self) -> None:
        result = find_epsilon_star(
            attack_at_epsilon=step_attack(true_threshold=100.0),  # unreachable within epsilon_max
            baseline_response={"flipped": False, "response_text": "No"},
            epsilon_max=EPSILON_MAX,
            eps0=EPS0,
        )
        self.assertEqual(result.status, "censored")
        self.assertIsNone(result.epsilon_star)
        # epsilon_max itself must have been tried before declaring censored, not just the last doubling step.
        self.assertTrue(any(t["epsilon"] == EPSILON_MAX for t in result.trace))

    def test_finds_a_threshold_far_below_the_starting_grid_point(self) -> None:
        # This is exactly the pilot's failure mode: a fixed grid at 1/255 would
        # see every one of these flip at the coarsest point tried and report
        # them as indistinguishable. The true threshold here is ~1/255 / 200.
        true_threshold = (1 / 255) / 200
        result = find_epsilon_star(
            attack_at_epsilon=step_attack(true_threshold),
            baseline_response={"flipped": False, "response_text": "No"},
            epsilon_max=EPSILON_MAX,
            eps0=EPS0,
            relative_tolerance=0.1,
        )
        self.assertEqual(result.status, "flipped")
        self.assertGreaterEqual(result.epsilon_star, true_threshold)
        self.assertLessEqual(result.epsilon_star, true_threshold * 1.1 + 1e-12)

    def test_finds_a_threshold_near_epsilon_max(self) -> None:
        true_threshold = EPSILON_MAX * 0.9
        result = find_epsilon_star(
            attack_at_epsilon=step_attack(true_threshold),
            baseline_response={"flipped": False, "response_text": "No"},
            epsilon_max=EPSILON_MAX,
            eps0=EPS0,
            relative_tolerance=0.1,
        )
        self.assertEqual(result.status, "flipped")
        self.assertGreaterEqual(result.epsilon_star, true_threshold)
        self.assertLessEqual(result.epsilon_star, true_threshold * 1.1 + 1e-9)

    def test_epsilon_star_is_always_within_relative_tolerance_of_true_threshold(self) -> None:
        for true_threshold in [1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1]:
            result = find_epsilon_star(
                attack_at_epsilon=step_attack(true_threshold),
                baseline_response={"flipped": False, "response_text": "No"},
                epsilon_max=EPSILON_MAX,
                eps0=EPS0,
                relative_tolerance=0.1,
            )
            self.assertEqual(result.status, "flipped", msg=f"threshold={true_threshold}")
            self.assertGreaterEqual(result.epsilon_star, true_threshold, msg=f"threshold={true_threshold}")
            self.assertLessEqual(
                result.epsilon_star, true_threshold * 1.1 + 1e-9, msg=f"threshold={true_threshold}"
            )

    def test_every_confirmed_flip_in_trace_is_at_or_above_true_threshold(self) -> None:
        true_threshold = 0.0123
        result = find_epsilon_star(
            attack_at_epsilon=step_attack(true_threshold),
            baseline_response={"flipped": False, "response_text": "No"},
            epsilon_max=EPSILON_MAX,
            eps0=EPS0,
        )
        for row in result.trace:
            if row["epsilon"] > 0 and row["flipped"]:
                self.assertGreaterEqual(row["epsilon"], true_threshold)


if __name__ == "__main__":
    unittest.main()
