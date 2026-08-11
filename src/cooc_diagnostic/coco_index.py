"""COCO instances_*.json loading with deduplicated per-image category presence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str


@dataclass
class CocoInstancesIndex:
    """Per-image category presence built from a COCO instances_*.json file.

    Presence is deduplicated per image: an image with 3 instances of "person"
    contributes a single membership to the "person" present-set, since
    co-occurrence statistics describe joint presence/absence, not instance counts.
    """

    categories: list[CocoCategory]  # sorted by category id
    image_categories: dict[int, set[int]]  # image_id -> present category ids (deduped)
    image_filenames: dict[int, str]  # image_id -> file_name

    @property
    def category_ids(self) -> list[int]:
        return [c.id for c in self.categories]

    @property
    def category_names(self) -> dict[int, str]:
        return {c.id: c.name for c in self.categories}


def load_coco_instances(annotation_path: str | Path) -> CocoInstancesIndex:
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    categories = sorted(
        (CocoCategory(id=int(c["id"]), name=str(c["name"])) for c in payload["categories"]),
        key=lambda c: c.id,
    )

    image_filenames = {int(im["id"]): str(im["file_name"]) for im in payload["images"]}
    image_categories: dict[int, set[int]] = {img_id: set() for img_id in image_filenames}

    for ann in payload["annotations"]:
        img_id = int(ann["image_id"])
        cat_id = int(ann["category_id"])
        image_categories.setdefault(img_id, set()).add(cat_id)

    return CocoInstancesIndex(
        categories=categories,
        image_categories=image_categories,
        image_filenames=image_filenames,
    )
