"""Exp4 counterfactual masking: gray-fill object-removal and a translated-mirror
sham control, generalized from the validated gray-fill technique already used
in extract_features_counterfactual.py::mask_dog_regions (polygon + RLE + bbox
fallback, fill_color=(114,114,114), reused by GenerationBias and BiasMitigation).

Sham definition (fixed here, before Exp4 is run, per the no-post-hoc-matching
rule): horizontally mirror the trigger-object mask within the image, then, if
it overlaps the trigger's own region or any OTHER annotated object in the same
image, translate it vertically in fixed steps until a non-overlapping
placement is found (or the search is exhausted, in which case the
lowest-overlap placement tried is used and flagged, never silently dropped).
This is a simplified alternative to Method_DSTR's appearance-matched
background-patch sham (src/dstr/data/removal.py::find_background_sham_mask) --
not texture-matched, but same fill mechanism, same area, a genuinely different
image region -- documented as a limitation, not claimed as appearance-matched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FILL_COLOR: tuple[int, int, int] = (114, 114, 114)
VERTICAL_SHIFT_FRACTIONS = [0.0, 0.15, -0.15, 0.30, -0.30, 0.45, -0.45]


@dataclass
class RawCocoAnnotations:
    image_annotations: dict[int, list[dict]]  # image_id -> all annotation dicts (any category)


def load_raw_annotations(annotation_path: str | Path) -> RawCocoAnnotations:
    with Path(annotation_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    image_annotations: dict[int, list[dict]] = {}
    for ann in payload["annotations"]:
        image_annotations.setdefault(int(ann["image_id"]), []).append(ann)
    return RawCocoAnnotations(image_annotations=image_annotations)


def _annotations_mask(annotations: list[dict], img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=bool)
    for ann in annotations:
        seg = ann.get("segmentation", [])
        if not seg:
            x, y, w, h = [int(round(v)) for v in ann["bbox"]]
            mask[max(y, 0) : min(y + h, img_h), max(x, 0) : min(x + w, img_w)] = True
            continue
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) < 6:
                    continue
                draw_img = Image.new("L", (img_w, img_h), 0)
                ImageDraw.Draw(draw_img).polygon(
                    [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=1
                )
                mask |= np.array(draw_img, dtype=bool)
        elif isinstance(seg, dict):
            try:
                from pycocotools import mask as coco_mask

                rle = coco_mask.frPyObjects(seg, img_h, img_w) if isinstance(seg.get("counts"), list) else seg
                mask |= coco_mask.decode(rle).squeeze().astype(bool)
            except ImportError:
                x, y, w, h = [int(round(v)) for v in ann["bbox"]]
                mask[max(y, 0) : min(y + h, img_h), max(x, 0) : min(x + w, img_w)] = True
    return mask


def _fill(image: Image.Image, mask: np.ndarray, fill_color=FILL_COLOR) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    arr[mask] = fill_color
    return Image.fromarray(arr)


@dataclass
class InterventionResult:
    image: Image.Image
    mask: np.ndarray
    mask_area_px: int
    metadata: dict


def apply_bat_removal(image: Image.Image, bat_annotations: list[dict]) -> InterventionResult:
    w, h = image.size
    mask = _annotations_mask(bat_annotations, h, w)
    return InterventionResult(
        image=_fill(image, mask),
        mask=mask,
        mask_area_px=int(mask.sum()),
        metadata={"n_bat_instances": len(bat_annotations)},
    )


def apply_sham_removal(image: Image.Image, bat_mask: np.ndarray, other_annotations: list[dict]) -> InterventionResult:
    w, h = image.size
    other_mask = _annotations_mask(other_annotations, h, w)
    forbidden = other_mask | bat_mask
    mirrored = np.fliplr(bat_mask)

    best_mask = mirrored
    best_overlap = int((mirrored & forbidden).sum())
    best_shift = 0.0
    resolved = best_overlap == 0

    if not resolved:
        for frac in VERTICAL_SHIFT_FRACTIONS[1:]:
            shift = int(round(frac * h))
            shifted = np.roll(mirrored, shift, axis=0)
            if shift > 0:
                shifted[:shift, :] = False
            elif shift < 0:
                shifted[shift:, :] = False
            overlap = int((shifted & forbidden).sum())
            if overlap < best_overlap:
                best_mask, best_overlap, best_shift = shifted, overlap, frac
            if overlap == 0:
                best_mask, best_overlap, best_shift = shifted, 0, frac
                resolved = True
                break

    return InterventionResult(
        image=_fill(image, best_mask),
        mask=best_mask,
        mask_area_px=int(best_mask.sum()),
        metadata={
            "sham_vertical_shift_frac": best_shift,
            "sham_overlap_residual_px": best_overlap,
            "sham_overlap_resolved": resolved,
        },
    )
