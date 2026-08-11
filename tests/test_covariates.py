import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.covariates import compute_category_average_area


class TestCategoryAverageArea(unittest.TestCase):
    def test_normalizes_by_image_area_and_averages_per_instance(self) -> None:
        payload = {
            "images": [
                {"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"},  # area=10000
                {"id": 2, "width": 200, "height": 100, "file_name": "b.jpg"},  # area=20000
            ],
            "annotations": [
                {"image_id": 1, "category_id": 1, "area": 1000.0},  # 0.10
                {"image_id": 2, "category_id": 1, "area": 4000.0},  # 0.20
                {"image_id": 1, "category_id": 2, "area": 5000.0},  # 0.50
            ],
            "categories": [{"id": 1, "name": "cat1"}, {"id": 2, "name": "cat2"}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name

        result = compute_category_average_area(path)

        self.assertAlmostEqual(result[1], (0.10 + 0.20) / 2, places=6)
        self.assertAlmostEqual(result[2], 0.50, places=6)


if __name__ == "__main__":
    unittest.main()
