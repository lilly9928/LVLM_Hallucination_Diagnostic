"""Part IX: the spurious-component loss, plus the two retain objectives.

L_spur = ||Proj_{u*}(delta_h_theta)||^2 ,  delta_h_theta = h_theta(x_adv) - h_theta(x)

Since u* is unit-norm, ||Proj_u(v)||^2 = (v . u)^2 -- a single dot product,
squared. Retain losses reuse the plain supervised-CE convention already used
by train_lora_debias.py (labels=-100 over the prompt, real token id over the
single-token "Yes" answer) -- same idea, new call site, since this pilot's
training loop shape (forget PAIRS, not single forget examples) differs from
that script's.
"""

from __future__ import annotations

import torch

from cooc_diagnostic.llava_runtime import normalize


def differentiable_hidden_at_layer(model, processor, decision_point, input_ids, image01, layer_idx: int) -> torch.Tensor:
    """Same hidden-state readout convention as extract_layer19_shifts.py's
    get_hidden_at_layer, but WITHOUT torch.no_grad() -- gradients must flow
    into the LoRA parameters through this call."""
    if decision_point.prefix_ids:
        prefix = torch.tensor([decision_point.prefix_ids], device=input_ids.device, dtype=input_ids.dtype)
        full_ids = torch.cat([input_ids, prefix], dim=1)
    else:
        full_ids = input_ids
    pixel_values = normalize(processor, image01).to(model.dtype)
    out = model(input_ids=full_ids, pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    return out.hidden_states[layer_idx][0, -1, :]


def spurious_projection_loss(h_clean: torch.Tensor, h_adv: torch.Tensor, u_star: torch.Tensor) -> torch.Tensor:
    delta_h_theta = h_adv - h_clean
    u = u_star.to(delta_h_theta.dtype).to(delta_h_theta.device)
    coeff = torch.dot(delta_h_theta, u)
    return coeff ** 2


def build_labeled_ids(processor, decision_point, category: str, answer: str, image, device: str):
    from cooc_diagnostic.llava_runtime import build_inputs

    input_ids, image01 = build_inputs(processor, category, image, device)
    answer_token_id = decision_point.yes_token_id if answer == "Yes" else decision_point.no_token_id
    if decision_point.prefix_ids:
        prefix = torch.tensor([decision_point.prefix_ids], device=device, dtype=input_ids.dtype)
    else:
        prefix = torch.zeros((1, 0), device=device, dtype=input_ids.dtype)
    answer_tensor = torch.tensor([[answer_token_id]], device=device, dtype=input_ids.dtype)
    full_ids = torch.cat([input_ids, prefix, answer_tensor], dim=1)
    labels = torch.cat([torch.full_like(input_ids, -100), prefix.clone(), answer_tensor.clone()], dim=1)
    return full_ids, labels, image01


def supervised_ce_loss(model, processor, decision_point, category: str, answer: str, image, device: str) -> torch.Tensor:
    full_ids, labels, image01 = build_labeled_ids(processor, decision_point, category, answer, image, device)
    pixel_values = normalize(processor, image01).to(model.dtype)
    outputs = model(input_ids=full_ids, pixel_values=pixel_values, labels=labels)
    return outputs.loss
