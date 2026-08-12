"""Part IX/X: Model C training loop (Adv + Decomp Debias).

Same LoRA hyperparameters, same retain pools, and the same per-step training
shape (forget : retain_target : retain_context mixed every step) as the
existing Clean/Adv Debias runs (adversarial_functional_debiasing_pilot/
training/train_lora_debias.py) -- the only thing this loop changes is the
forget-example loss:

  spur_only         : L_total = L_spur + lambda_T*L_target_retain + lambda_C*L_context_retain
  spur_plus_adv_no  : L_total = L_adv_no + lambda_spur*L_spur + lambda_T*L_target_retain + lambda_C*L_context_retain

L_spur requires BOTH the clean and adversarial image per forget example (to
form delta_h_theta = h_theta(x_adv) - h_theta(x)), so this loop's forget step
does two forward passes (clean, adv) instead of one.
"""

from __future__ import annotations

import random

import torch
from PIL import Image

from cooc_diagnostic.llava_runtime import build_inputs
from training.data import load_decomp_pools
from training.losses import differentiable_hidden_at_layer, spurious_projection_loss, supervised_ce_loss


def run_training(
    peft_model,
    processor,
    decision_point,
    pools: dict,
    u_star: torch.Tensor,
    layer_idx: int,
    target_category: str,
    context_category: str,
    loss_mode: str,
    lambda_spur: float,
    lambda_target_retain: float,
    lambda_context_retain: float,
    num_epochs: int,
    grad_accum: int,
    lr: float,
    seed: int,
    device: str,
    log_every: int = 10,
) -> list[dict]:
    optimizer = torch.optim.AdamW((p for p in peft_model.parameters() if p.requires_grad), lr=lr)

    forget_pairs = pools["forget_pairs"]
    retain_target = pools["retain_target"]
    retain_context = pools["retain_context"]

    log_rows = []
    global_step = 0
    peft_model.train()

    for epoch in range(num_epochs):
        rng = random.Random(seed + epoch)
        fp_shuf, rt_shuf, rc_shuf = forget_pairs[:], retain_target[:], retain_context[:]
        rng.shuffle(fp_shuf)
        rng.shuffle(rt_shuf)
        rng.shuffle(rc_shuf)
        n_steps = min(len(fp_shuf), len(rt_shuf), len(rc_shuf))

        optimizer.zero_grad()
        for step in range(n_steps):
            pair = fp_shuf[step]
            clean_image = Image.open(pair.clean_image_path).convert("RGB")
            adv_image = Image.open(pair.adv_image_path).convert("RGB")
            input_ids_c, image01_c = build_inputs(processor, target_category, clean_image, device)
            input_ids_a, image01_a = build_inputs(processor, target_category, adv_image, device)

            h_clean = differentiable_hidden_at_layer(peft_model, processor, decision_point, input_ids_c, image01_c, layer_idx)
            h_adv = differentiable_hidden_at_layer(peft_model, processor, decision_point, input_ids_a, image01_a, layer_idx)
            l_spur = spurious_projection_loss(h_clean, h_adv, u_star)

            if loss_mode == "spur_plus_adv_no":
                l_adv_no = supervised_ce_loss(peft_model, processor, decision_point, target_category, "No", adv_image, device)
                forget_loss = l_adv_no + lambda_spur * l_spur
            else:
                l_adv_no = None
                forget_loss = l_spur

            rt_example = rt_shuf[step]
            rt_image = Image.open(rt_example.image_path).convert("RGB")
            l_target_retain = supervised_ce_loss(peft_model, processor, decision_point, rt_example.question_category, rt_example.answer, rt_image, device)

            rc_example = rc_shuf[step]
            rc_image = Image.open(rc_example.image_path).convert("RGB")
            l_context_retain = supervised_ce_loss(peft_model, processor, decision_point, rc_example.question_category, rc_example.answer, rc_image, device)

            total_loss = forget_loss + lambda_target_retain * l_target_retain + lambda_context_retain * l_context_retain
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
                    "loss_spur": float(l_spur.detach()),
                    "loss_adv_no": float(l_adv_no.detach()) if l_adv_no is not None else None,
                    "loss_target_retain": float(l_target_retain.detach()),
                    "loss_context_retain": float(l_context_retain.detach()),
                    "loss_total": float(total_loss.detach()),
                }
            )
            if (step + 1) % log_every == 0:
                print(
                    f"[train:adv_decomp] epoch {epoch} step {step + 1}/{n_steps} "
                    f"loss_spur={log_rows[-1]['loss_spur']:.4f} "
                    f"loss_rt={log_rows[-1]['loss_target_retain']:.4f} "
                    f"loss_rc={log_rows[-1]['loss_context_retain']:.4f} "
                    f"loss_total={log_rows[-1]['loss_total']:.4f}"
                )
        torch.cuda.empty_cache()

    return log_rows
