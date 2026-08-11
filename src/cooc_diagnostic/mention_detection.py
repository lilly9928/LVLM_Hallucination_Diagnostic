"""Word-boundary category-mention detection for open-ended caption text.

Deliberately simple (exact name + naive plural, case-insensitive word-boundary
match) -- this is a lightweight transfer check, not a new NLP subsystem. COCO
category names are common, unambiguous English nouns/noun phrases (e.g. "bench",
"traffic light"), so false positives from substring collisions are unlikely;
this is not attempting synonym coverage.
"""

from __future__ import annotations

import re

# The only irregular plurals among COCO's 80 category names; a generic
# "-f/-fe -> -ves" rule would incorrectly turn "giraffe" into "giraves", so
# irregulars are special-cased explicitly instead of guessed heuristically.
_IRREGULAR_PLURALS = {
    "person": "people",
    "knife": "knives",
    "mouse": "mice",
}


def _pluralize(word: str) -> str:
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def category_mention_variants(category_name: str) -> list[str]:
    words = category_name.split(" ")
    last_word_plural = _pluralize(words[-1])
    plural_name = " ".join(words[:-1] + [last_word_plural])
    variants = {category_name.lower(), plural_name.lower()}
    return sorted(variants)


def text_mentions_category(text: str, category_name: str) -> bool:
    text_lower = text.lower()
    for variant in category_mention_variants(category_name):
        if re.search(rf"\b{re.escape(variant)}\b", text_lower):
            return True
    return False
