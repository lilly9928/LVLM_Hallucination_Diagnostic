import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.random_attack import random_perturbations


class TestRandomPerturbations(unittest.TestCase):
    def test_stays_within_pixel_range_and_epsilon_ball(self) -> None:
        image01 = torch.tensor([0.02, 0.5, 0.98])
        epsilon = 0.1
        images = random_perturbations(image01, epsilon=epsilon, n_trials=20)
        self.assertEqual(len(images), 20)
        for img in images:
            self.assertTrue(torch.all(img >= 0.0))
            self.assertTrue(torch.all(img <= 1.0))
            self.assertTrue(torch.all((img - image01).abs() <= epsilon + 1e-6))

    def test_trials_are_independent_draws(self) -> None:
        torch.manual_seed(0)
        image01 = torch.full((100,), 0.5)
        images = random_perturbations(image01, epsilon=0.2, n_trials=5)
        # Independent draws should not all be identical.
        self.assertFalse(all(torch.allclose(images[0], img) for img in images[1:]))


if __name__ == "__main__":
    unittest.main()
