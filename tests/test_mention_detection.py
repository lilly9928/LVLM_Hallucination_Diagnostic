import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.mention_detection import category_mention_variants, text_mentions_category


class TestCategoryMentionVariants(unittest.TestCase):
    def test_regular_plural(self) -> None:
        self.assertEqual(category_mention_variants("bench"), ["bench", "benches"])

    def test_multi_word_category_pluralizes_last_word_only(self) -> None:
        self.assertEqual(category_mention_variants("wine glass"), ["wine glass", "wine glasses"])
        self.assertEqual(category_mention_variants("hot dog"), ["hot dog", "hot dogs"])

    def test_irregular_plurals(self) -> None:
        self.assertIn("knives", category_mention_variants("knife"))
        self.assertIn("mice", category_mention_variants("mouse"))
        self.assertIn("people", category_mention_variants("person"))

    def test_does_not_mangle_double_f_words(self) -> None:
        # a naive "-fe/-f -> -ves" heuristic would wrongly produce "giraves".
        self.assertIn("giraffes", category_mention_variants("giraffe"))
        self.assertNotIn("giraves", category_mention_variants("giraffe"))


class TestTextMentionsCategory(unittest.TestCase):
    def test_matches_singular_and_plural_case_insensitively(self) -> None:
        self.assertTrue(text_mentions_category("There is a Bench in the park.", "bench"))
        self.assertTrue(text_mentions_category("Several benches line the path.", "bench"))
        self.assertFalse(text_mentions_category("A person sits on a chair.", "bench"))

    def test_respects_word_boundaries_no_substring_false_positive(self) -> None:
        # "bus" should not match inside "business" or "busy".
        self.assertFalse(text_mentions_category("This is a busy business district.", "bus"))
        self.assertTrue(text_mentions_category("A bus drives by.", "bus"))

    def test_multi_word_category_matched_as_phrase(self) -> None:
        self.assertTrue(text_mentions_category("A traffic light is visible at the corner.", "traffic light"))
        self.assertFalse(text_mentions_category("The traffic was heavy near the light pole.", "traffic light"))


if __name__ == "__main__":
    unittest.main()
