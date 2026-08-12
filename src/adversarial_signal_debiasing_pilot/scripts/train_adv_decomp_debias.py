"""Part IX/X: train Model C (Adv + Decomp Debias) -- the only genuinely new
LoRA training run in this pilot (Models A/B are reused verbatim, see
configs/training_clean.yaml / training_adv.yaml and README "What We Reused").

Selects the best component u* from decomposition/component_selectivity.csv
(produced by evaluate_components.py on the internal VAL split -- never on
CLEAN TEST), then trains with the loss specified in configs/
training_adv_decomp.yaml (loss_mode: spur_only by default, the Part IX
"preferred first prototype").

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/train_adv_decomp_debias.py \
        --data-config configs/data.yaml --decomp-config configs/decomposition.yaml \
        --train-config configs/training_adv_decomp.yaml [--device cuda:0]
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../CooccurrenceHallucinationDiagnostic/src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # .../adversarial_signal_debiasing_pilot

from cooc_diagnostic.coco_index import load_coco_instances  # noqa: E402
from cooc_diagnostic.llava_runtime import detect_yes_no_decision_point, load_model  # noqa: E402
from training.data import load_decomp_pools  # noqa: E402
from training.lora_utils import build_peft_model  # noqa: E402
from training.trainer import run_training  # noqa: E402


def select_best_component(out_dir: Path) -> dict:
    rows = list(csv.DictReader((out_dir / "decomposition" / "component_selectivity.csv").open("r", encoding="utf-8")))
    decomposed = [r for r in rows if r["candidate_type"] in ("pca_component", "pls_component")]
    if not decomposed:
        raise RuntimeError("No pca_component/pls_component rows found in component_selectivity.csv")
    decomposed.sort(key=lambda r: float(r["selectivity_min_over_lambda"]), reverse=True)
    best = decomposed[0]
    mean_row = next((r for r in rows if r["candidate_type"] == "mean_direction"), None)
    best_random = max((r for r in rows if r["candidate_type"] == "random_direction"), key=lambda r: float(r["selectivity_min_over_lambda"]), default=None)
    beats_mean = mean_row is not None and float(best["selectivity_min_over_lambda"]) > float(mean_row["selectivity_min_over_lambda"])
    beats_random = best_random is not None and float(best["selectivity_min_over_lambda"]) > float(best_random["selectivity_min_over_lambda"])
    return {
        "candidate_type": best["candidate_type"],
        "candidate_id": best["candidate_id"],
        "selectivity_min_over_lambda": float(best["selectivity_min_over_lambda"]),
        "beats_mean_direction": beats_mean,
        "beats_best_random_direction": beats_random,
    }


def load_component_vector(out_dir: Path, candidate_type: str, candidate_id: str) -> torch.Tensor:
    if candidate_type == "pca_component":
        data = torch.load(out_dir / "decomposition" / "pca_components.pt")
        idx = int(candidate_id.replace("PC", "")) - 1
        return data["components"][idx]
    elif candidate_type == "pls_component":
        data = torch.load(out_dir / "decomposition" / "pls_components.pt")
        idx = int(candidate_id.replace("PLS", "")) - 1
        return data["components"][idx]
    raise ValueError(candidate_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--decomp-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    with args.data_config.open("r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with args.decomp_config.open("r", encoding="utf-8") as f:
        decomp_config = yaml.safe_load(f)
    with args.train_config.open("r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    device = args.device
    seed = int(data_config["seed"])
    out_dir = Path(data_config["output_dir"])
    layer_idx = int(train_config.get("layer_idx", decomp_config["layer_idx"]))

    variant_dir = out_dir / "adv_decomp_debias"
    ckpt_dir = out_dir / "checkpoints" / "adv_decomp_debias"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    random.seed(seed)

    selected = select_best_component(out_dir)
    print(f"[train:adv_decomp] selected component: {selected}")
    u_star = load_component_vector(out_dir, selected["candidate_type"], selected["candidate_id"]).to(device)

    print(f"[train:adv_decomp] loading base model {data_config['model_id']} on {device}")
    model, processor = load_model(data_config["model_id"], device)

    val_index = load_coco_instances(data_config["val_annotation_path"])
    val_image_dir = Path(data_config["val_image_dir"])

    pools = load_decomp_pools(
        train_split_path=out_dir / "data" / "train_split.csv",
        adversarial_forget_set_path=out_dir / "data" / "adversarial_forget_set.csv",
        val_index=val_index,
        val_image_dir=val_image_dir,
        target_category=data_config["target_category"],
        context_category=data_config["context_category"],
    )
    n_forget, n_rt, n_rc = len(pools["forget_pairs"]), len(pools["retain_target"]), len(pools["retain_context"])
    print(f"[train:adv_decomp] pools: forget_pairs={n_forget} retain_target={n_rt} retain_context={n_rc}")

    probe_image = Image.open(pools["forget_pairs"][0].clean_image_path).convert("RGB")
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[train:adv_decomp] decision point: yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    lora_cfg = train_config["lora"]
    peft_model, target_modules = build_peft_model(model, lora_cfg)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"[train:adv_decomp] trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

    loss_mode = train_config.get("loss_mode", "spur_only")
    log_rows = run_training(
        peft_model=peft_model,
        processor=processor,
        decision_point=decision_point,
        pools=pools,
        u_star=u_star,
        layer_idx=layer_idx,
        target_category=data_config["target_category"],
        context_category=data_config["context_category"],
        loss_mode=loss_mode,
        lambda_spur=float(train_config["lambda_spur"]),
        lambda_target_retain=float(train_config["lambda_target_retain"]),
        lambda_context_retain=float(train_config["lambda_context_retain"]),
        num_epochs=int(lora_cfg["num_epochs"]),
        grad_accum=int(lora_cfg["gradient_accumulation_steps"]),
        lr=float(lora_cfg["learning_rate"]),
        seed=seed,
        device=device,
    )

    log_path = variant_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train:adv_decomp] wrote {log_path}")

    peft_model.save_pretrained(str(ckpt_dir))
    print(f"[train:adv_decomp] saved LoRA adapter to {ckpt_dir}")

    training_config = {
        "variant": "adv_decomp",
        "model_id": data_config["model_id"],
        "seed": seed,
        "layer_idx": layer_idx,
        "selected_component": selected,
        "loss_mode": loss_mode,
        "lambda_spur": float(train_config["lambda_spur"]),
        "lambda_target_retain": float(train_config["lambda_target_retain"]),
        "lambda_context_retain": float(train_config["lambda_context_retain"]),
        "lora": lora_cfg,
        "target_modules_resolved_count": len(target_modules),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100 * trainable_params / total_params,
        "n_forget_pairs": n_forget,
        "n_retain_target": n_rt,
        "n_retain_context": n_rc,
        "n_epochs": int(lora_cfg["num_epochs"]),
        "gradient_accumulation_steps": int(lora_cfg["gradient_accumulation_steps"]),
        "n_optimizer_steps": len(log_rows) // int(lora_cfg["gradient_accumulation_steps"]),
    }
    cfg_path = variant_dir / "training_config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)
    print(f"[train:adv_decomp] wrote {cfg_path}")
    print(f"[train:adv_decomp] final losses: {log_rows[-1]}")


if __name__ == "__main__":
    main()
