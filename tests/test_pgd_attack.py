import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.pgd_attack import pgd_attack, pgd_attack_with_restarts

TARGET = torch.tensor([0.9, 0.1, 0.5])  # margin is maximized as image01 -> TARGET


def margin_fn(image01: torch.Tensor) -> torch.Tensor:
    # Negative squared distance to TARGET: 0 at the target, increasingly negative
    # away from it. This is > 0 only exactly at the target itself, so a large
    # enough epsilon ball around a starting point far from TARGET should NOT flip
    # (margin cannot reach exactly 0), while a starting point already close should.
    return -((image01 - TARGET) ** 2).sum()


class TestPGDAttack(unittest.TestCase):
    def test_never_leaves_the_zero_one_pixel_range(self) -> None:
        image01 = torch.tensor([0.05, 0.95, 0.5])
        result = pgd_attack(image01, epsilon=0.5, n_steps=20, margin_fn=margin_fn)
        self.assertTrue(torch.all(result.best_image >= 0.0))
        self.assertTrue(torch.all(result.best_image <= 1.0))

    def test_never_leaves_the_epsilon_ball_around_the_original_image(self) -> None:
        image01 = torch.tensor([0.5, 0.5, 0.5])
        epsilon = 0.1
        result = pgd_attack(image01, epsilon=epsilon, n_steps=20, margin_fn=margin_fn)
        self.assertTrue(torch.all((result.best_image - image01).abs() <= epsilon + 1e-6))

    def test_margin_improves_over_the_unperturbed_image(self) -> None:
        image01 = torch.tensor([0.5, 0.5, 0.5])
        initial_margin = float(margin_fn(image01))
        result = pgd_attack(image01, epsilon=0.3, n_steps=30, margin_fn=margin_fn)
        self.assertGreater(result.best_margin, initial_margin)

    def test_epsilon_zero_runs_no_steps_and_reports_clean_margin(self) -> None:
        image01 = torch.tensor([0.5, 0.5, 0.5])
        result = pgd_attack(image01, epsilon=0.0, n_steps=20, margin_fn=margin_fn)
        self.assertEqual(result.n_steps_used, 0)
        self.assertAlmostEqual(result.best_margin, float(margin_fn(image01)), places=6)

    def test_restarts_can_escape_a_zero_gradient_saddle_point(self) -> None:
        # A saddle centered exactly at the clean image: gradient is exactly zero
        # there, so a zero-init-only search (n_restarts=1) can never move and must
        # fail to flip -- while random-init restarts can land off the saddle and
        # climb to margin > 0. This is the scenario restarts exist to guard against.
        def saddle_margin_fn(image01: torch.Tensor) -> torch.Tensor:
            dx = image01[0] - 0.5
            dy = image01[1] - 0.5
            return dx * dy * 20.0

        original = torch.tensor([0.5, 0.5])

        torch.manual_seed(0)
        single = pgd_attack_with_restarts(original.clone(), epsilon=0.1, n_steps=10, margin_fn=saddle_margin_fn, n_restarts=1)
        self.assertFalse(single.flipped_by_margin)
        self.assertEqual(single.best_margin, 0.0)

        torch.manual_seed(0)
        multi = pgd_attack_with_restarts(original.clone(), epsilon=0.1, n_steps=10, margin_fn=saddle_margin_fn, n_restarts=5)
        self.assertTrue(multi.flipped_by_margin)


if __name__ == "__main__":
    unittest.main()
