import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.stratified_subsample import stratified_subsample


def _make_rows(cell_sizes: dict[tuple, int]) -> list[dict]:
    rows = []
    for cell, size in cell_sizes.items():
        for i in range(size):
            rows.append({"cell_a": cell[0], "cell_b": cell[1], "idx": i})
    return rows


class TestStratifiedSubsample(unittest.TestCase):
    def test_returns_exact_target_count(self) -> None:
        rows = _make_rows({(0, 0): 40, (0, 1): 30, (1, 0): 20, (1, 1): 10})
        result = stratified_subsample(rows, ["cell_a", "cell_b"], target_n=20, rng=random.Random(0))
        self.assertEqual(len(result), 20)

    def test_preserves_cell_proportions_approximately(self) -> None:
        rows = _make_rows({(0, 0): 400, (0, 1): 300, (1, 0): 200, (1, 1): 100})
        result = stratified_subsample(rows, ["cell_a", "cell_b"], target_n=100, rng=random.Random(0))
        counts = Counter((r["cell_a"], r["cell_b"]) for r in result)
        # Original proportions: 40%/30%/20%/10% -> expect close to 40/30/20/10 out of 100.
        self.assertEqual(counts[(0, 0)], 40)
        self.assertEqual(counts[(0, 1)], 30)
        self.assertEqual(counts[(1, 0)], 20)
        self.assertEqual(counts[(1, 1)], 10)

    def test_never_exceeds_a_cells_available_rows(self) -> None:
        rows = _make_rows({(0, 0): 3, (0, 1): 3, (1, 0): 3, (1, 1): 3})
        result = stratified_subsample(rows, ["cell_a", "cell_b"], target_n=11, rng=random.Random(0))
        self.assertEqual(len(result), 11)
        counts = Counter((r["cell_a"], r["cell_b"]) for r in result)
        for c in counts.values():
            self.assertLessEqual(c, 3)

    def test_returns_all_rows_when_target_exceeds_total(self) -> None:
        rows = _make_rows({(0, 0): 5, (0, 1): 5})
        result = stratified_subsample(rows, ["cell_a", "cell_b"], target_n=1000, rng=random.Random(0))
        self.assertEqual(len(result), 10)

    def test_no_row_is_duplicated(self) -> None:
        rows = _make_rows({(0, 0): 50, (0, 1): 50})
        result = stratified_subsample(rows, ["cell_a", "cell_b"], target_n=30, rng=random.Random(1))
        ids = [(r["cell_a"], r["cell_b"], r["idx"]) for r in result]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
