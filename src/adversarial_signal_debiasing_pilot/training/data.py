"""Training pools for Model C (Adv + Decomp Debias).

Adapted from the prior adversarial_functional_debiasing_pilot's
training/data.py::load_pools (new code, that file is not imported or
modified -- it returns a single image path per forget example selected by
`variant`, but this pilot's L_spur term needs BOTH the clean and adversarial
image for every forget pair simultaneously, so the pool shape differs).
Retain pools are unchanged in spirit: clean images only, one role each.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cooc_diagnostic.coco_index import CocoInstancesIndex


@dataclass(frozen=True)
class ForgetPair:
    image_id: int
    clean_image_path: str
    adv_image_path: str


@dataclass(frozen=True)
class RetainExample:
    image_id: int
    image_path: str
    question_category: str
    answer: str  # always "Yes" for both retain roles in this pilot


def _resolve_path(image_dir: Path, val_index: CocoInstancesIndex, image_id: int) -> str:
    return str(image_dir / val_index.image_filenames[image_id])


def load_decomp_pools(
    train_split_path: Path,
    adversarial_forget_set_path: Path,
    val_index: CocoInstancesIndex,
    val_image_dir: Path,
    target_category: str,
    context_category: str,
) -> dict:
    train_rows = list(csv.DictReader(train_split_path.open("r", encoding="utf-8")))
    forget_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "G10_forget"})
    retain_target_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "GT_target_retain"})
    retain_context_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "GC_context_retain"})

    adv_rows = {int(r["image_id"]): r for r in csv.DictReader(adversarial_forget_set_path.open("r", encoding="utf-8"))}

    forget_pairs = [
        ForgetPair(image_id, adv_rows[image_id]["clean_image_path"], adv_rows[image_id]["adv_image_path"])
        for image_id in forget_ids
    ]
    retain_target = [
        RetainExample(image_id, _resolve_path(val_image_dir, val_index, image_id), target_category, "Yes")
        for image_id in retain_target_ids
    ]
    retain_context = [
        RetainExample(image_id, _resolve_path(val_image_dir, val_index, image_id), context_category, "Yes")
        for image_id in retain_context_ids
    ]
    return {"forget_pairs": forget_pairs, "retain_target": retain_target, "retain_context": retain_context}
