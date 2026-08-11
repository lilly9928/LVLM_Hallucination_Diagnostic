"""Category-level average object area (normalized by image area), from COCO train2017.

This is a confound-matching covariate for Stage 2: two categories can have the
same marginal frequency yet be systematically different in typical object size
(e.g. "toothbrush" vs "bed"), which could independently affect attack difficulty.

Area is averaged per *instance*, not deduplicated per image like presence in
coco_index.py -- an image with three small "apple" instances should pull the
average apple size down, since size is a property of the object, not of the image.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def compute_category_average_area(annotation_path: str | Path) -> dict[int, float]:
    """Returns category_id -> mean(instance_segmentation_area / image_area)."""
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    image_area = {int(im["id"]): float(im["width"]) * float(im["height"]) for im in payload["images"]}

    normalized_areas: dict[int, list[float]] = defaultdict(list)
    for ann in payload["annotations"]:
        img_area = image_area.get(int(ann["image_id"]))
        if not img_area:
            continue
        normalized_areas[int(ann["category_id"])].append(float(ann["area"]) / img_area)

    return {cat_id: sum(areas) / len(areas) for cat_id, areas in normalized_areas.items()}
