"""Builds the three fixed training pools (forget / retain-target / retain-context)
for both debiasing variants. Clean vs Adv Debias differ ONLY in which image path
is used for the forget pool -- retain pools are always clean, identical between
variants (see task spec "Retain Training Data").
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cooc_diagnostic.coco_index import CocoInstancesIndex


@dataclass(frozen=True)
class Example:
    image_id: int
    image_path: str
    question_category: str  # "sports ball" | "baseball bat"
    answer: str  # "Yes" | "No"
    role: str  # "forget" | "retain_target" | "retain_context"


def _resolve_path(image_dir: Path, val_index: CocoInstancesIndex, image_id: int) -> str:
    return str(image_dir / val_index.image_filenames[image_id])


def load_pools(
    train_split_path: Path,
    adversarial_forget_set_path: Path,
    val_index: CocoInstancesIndex,
    val_image_dir: Path,
    target_category: str,
    context_category: str,
    variant: str,  # "clean" | "adv"
) -> dict[str, list[Example]]:
    assert variant in ("clean", "adv")

    train_rows = list(csv.DictReader(train_split_path.open("r", encoding="utf-8")))
    forget_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "G10_forget"})
    retain_target_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "GT_target_retain"})
    retain_context_ids = sorted({int(r["image_id"]) for r in train_rows if r["role"] == "GC_context_retain"})

    adv_rows = {int(r["image_id"]): r for r in csv.DictReader(adversarial_forget_set_path.open("r", encoding="utf-8"))}

    forget: list[Example] = []
    for image_id in forget_ids:
        row = adv_rows[image_id]
        path = row["adv_image_path"] if variant == "adv" else row["clean_image_path"]
        forget.append(Example(image_id, path, target_category, "No", "forget"))

    retain_target = [
        Example(image_id, _resolve_path(val_image_dir, val_index, image_id), target_category, "Yes", "retain_target")
        for image_id in retain_target_ids
    ]
    retain_context = [
        Example(image_id, _resolve_path(val_image_dir, val_index, image_id), context_category, "Yes", "retain_context")
        for image_id in retain_context_ids
    ]

    return {"forget": forget, "retain_target": retain_target, "retain_context": retain_context}
