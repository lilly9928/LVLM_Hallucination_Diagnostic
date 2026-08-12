"""LoRA setup, same conventions as adversarial_functional_debiasing_pilot/
training/train_lora_debias.py (new code adapted from that pattern; that file
is not imported or modified)."""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model


def resolve_target_modules(model, suffixes: list[str]) -> list[str]:
    """Full dotted names of language_model Linear layers ending in one of
    `suffixes`, excluding the vision tower / projector."""
    names = []
    for name, module in model.named_modules():
        if "vision_tower" in name or "multi_modal_projector" in name:
            continue
        if "language_model" not in name:
            continue
        if isinstance(module, torch.nn.Linear) and any(name.endswith(s) for s in suffixes):
            names.append(name)
    if not names:
        raise RuntimeError(f"No language_model Linear layers matched suffixes {suffixes}; inspect model.named_modules().")
    return names


def build_peft_model(model, lora_cfg: dict):
    target_modules = resolve_target_modules(model, lora_cfg["target_modules"])
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    return peft_model, target_modules
