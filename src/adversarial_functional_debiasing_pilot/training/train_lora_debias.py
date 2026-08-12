"""Step 2/3: minimal LoRA SFT for ONE debiasing variant (clean or adv).

Ordinary supervised cross-entropy (labels = -100 over the prompt, real token ids
over the answer span), no custom unlearning objective. Forget/retain-target/
retain-context examples are mixed 1:1:1 per optimizer step (fixed ratio, not
tuned post hoc). The ONLY difference between the two variants this script can
produce is which image path backs the forget examples (clean vs adversarial) --
everything else (retain data, hyperparameters, seed, code path) is identical.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python training/train_lora_debias.py \
        --config configs/pilot.yaml --variant clean --device cuda:0
    /opt/anaconda3/envs/py3_11/bin/python training/train_lora_debias.py \
        --config configs/pilot.yaml --variant adv --device cuda:2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../CooccurrenceHallucinationDiagnostic/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../adversarial_functional_debiasing_pilot

from cooc_diagnostic.coco_index import load_coco_instances  # noqa: E402
from cooc_diagnostic.llava_runtime import (  # noqa: E402
    build_inputs,
    detect_yes_no_decision_point,
    load_model,
    normalize,
)
from training.data import load_pools  # noqa: E402


def resolve_target_modules(model, suffixes: list[str]) -> list[str]:
    """Full dotted names of language_model Linear layers ending in one of
    `suffixes`, explicitly excluding the vision tower (CLIP attention layers
    share the same q_proj/v_proj naming, which must never be targeted)."""
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


def build_example_ids(processor, decision_point, category: str, answer: str, image: Image.Image, device: str):
    input_ids, image01 = build_inputs(processor, category, image, device)
    answer_token_id = decision_point.yes_token_id if answer == "Yes" else decision_point.no_token_id
    if decision_point.prefix_ids:
        prefix = torch.tensor([decision_point.prefix_ids], device=device, dtype=input_ids.dtype)
    else:
        prefix = torch.zeros((1, 0), device=device, dtype=input_ids.dtype)
    answer_tensor = torch.tensor([[answer_token_id]], device=device, dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, prefix, answer_tensor], dim=1)
    labels = torch.cat(
        [
            torch.full_like(input_ids, -100),
            prefix.clone(),
            answer_tensor.clone(),
        ],
        dim=1,
    )
    return full_ids, labels, image01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=["clean", "adv"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = args.device
    seed = int(config["seed"])
    out_dir = Path(config["output_dir"])
    variant_dir = out_dir / f"{args.variant}_debias"
    ckpt_dir = out_dir / "checkpoints" / f"{args.variant}_debias"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)

    print(f"[train:{args.variant}] loading base model on {device}")
    model, processor = load_model(config["model_id"], device)

    val_index = load_coco_instances(config["val_annotation_path"])
    val_image_dir = Path(config["val_image_dir"])

    pools = load_pools(
        train_split_path=out_dir / "data" / "train_split.csv",
        adversarial_forget_set_path=out_dir / "data" / "adversarial_forget_set.csv",
        val_index=val_index,
        val_image_dir=val_image_dir,
        target_category=config["target_category"],
        context_category=config["context_category"],
        variant=args.variant,
    )
    n_forget, n_rt, n_rc = len(pools["forget"]), len(pools["retain_target"]), len(pools["retain_context"])
    print(f"[train:{args.variant}] pools: forget={n_forget} retain_target={n_rt} retain_context={n_rc}")

    probe_image = Image.open(pools["forget"][0].image_path).convert("RGB") if args.variant == "clean" else Image.open(
        pools["retain_context"][0].image_path
    ).convert("RGB")
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[train:{args.variant}] decision point: yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    lora_cfg = config["lora"]
    target_modules = resolve_target_modules(model, lora_cfg["target_modules"])
    print(f"[train:{args.variant}] LoRA target modules ({len(target_modules)}): {target_modules[:2]} ... {target_modules[-2:]}")

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"[train:{args.variant}] trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    optimizer = torch.optim.AdamW((p for p in peft_model.parameters() if p.requires_grad), lr=float(lora_cfg["learning_rate"]))

    num_epochs = int(lora_cfg["num_epochs"])
    grad_accum = int(lora_cfg["gradient_accumulation_steps"])

    log_rows = []
    global_step = 0
    peft_model.train()

    for epoch in range(num_epochs):
        rng = random.Random(seed + epoch)
        forget_shuf = pools["forget"][:]
        rt_shuf = pools["retain_target"][:]
        rc_shuf = pools["retain_context"][:]
        rng.shuffle(forget_shuf)
        rng.shuffle(rt_shuf)
        rng.shuffle(rc_shuf)

        n_steps = min(len(forget_shuf), len(rt_shuf), len(rc_shuf))
        optimizer.zero_grad()
        for step in range(n_steps):
            triplet = [
                (forget_shuf[step], "forget"),
                (rt_shuf[step], "retain_target"),
                (rc_shuf[step], "retain_context"),
            ]
            losses = {}
            for example, role in triplet:
                image = Image.open(example.image_path).convert("RGB")
                full_ids, labels, image01 = build_example_ids(
                    processor, decision_point, example.question_category, example.answer, image, device
                )
                pixel_values = normalize(processor, image01).to(peft_model.dtype)
                outputs = peft_model(input_ids=full_ids, pixel_values=pixel_values, labels=labels)
                losses[role] = outputs.loss
            total_loss = losses["forget"] + losses["retain_target"] + losses["retain_context"]
            (total_loss / grad_accum).backward()

            global_step += 1
            if global_step % grad_accum == 0 or step == n_steps - 1:
                optimizer.step()
                optimizer.zero_grad()

            log_rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "loss_forget": float(losses["forget"].detach()),
                    "loss_retain_target": float(losses["retain_target"].detach()),
                    "loss_retain_context": float(losses["retain_context"].detach()),
                    "loss_total": float(total_loss.detach()),
                }
            )
            if (step + 1) % 10 == 0:
                print(
                    f"[train:{args.variant}] epoch {epoch} step {step + 1}/{n_steps} "
                    f"loss_forget={log_rows[-1]['loss_forget']:.4f} "
                    f"loss_rt={log_rows[-1]['loss_retain_target']:.4f} "
                    f"loss_rc={log_rows[-1]['loss_retain_context']:.4f}"
                )
        torch.cuda.empty_cache()

    log_path = variant_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train:{args.variant}] wrote {log_path}")

    peft_model.save_pretrained(str(ckpt_dir))
    print(f"[train:{args.variant}] saved LoRA adapter to {ckpt_dir}")

    training_config = {
        "variant": args.variant,
        "model_id": config["model_id"],
        "seed": seed,
        "lora": lora_cfg,
        "target_modules_resolved_count": len(target_modules),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100 * trainable_params / total_params,
        "n_forget": n_forget,
        "n_retain_target": n_rt,
        "n_retain_context": n_rc,
        "n_epochs": num_epochs,
        "gradient_accumulation_steps": grad_accum,
        "n_optimizer_steps": global_step // grad_accum,
    }
    cfg_path = variant_dir / "training_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
    print(f"[train:{args.variant}] wrote {cfg_path}")
    print(f"[train:{args.variant}] final losses: forget={log_rows[-1]['loss_forget']:.4f} "
          f"retain_target={log_rows[-1]['loss_retain_target']:.4f} retain_context={log_rows[-1]['loss_retain_context']:.4f}")


if __name__ == "__main__":
    main()
