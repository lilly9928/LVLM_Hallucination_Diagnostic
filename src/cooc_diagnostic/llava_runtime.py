"""LLaVA-1.5 runtime wrapper: loading, prompt construction, preprocessing-order
handling, and Yes/No decision-logit extraction for the attack and epsilon* search.

Preprocessing order (verified against llava-hf's CLIPImageProcessor config:
do_resize(shortest_edge=336) -> do_center_crop(336x336) -> do_rescale(1/255) ->
do_normalize(CLIP mean/std)): the perturbation in this experiment is added AFTER
resize+crop+rescale (i.e. to the [0,1] tensor on the exact 336x336 grid the vision
tower consumes) and BEFORE normalization. `preprocess_to_unit_range` stops right
after rescale (do_normalize=False); `normalize` applies CLIP mean/std as the very
last step, inside the differentiable forward pass used by the attack -- so
perturbation and gradient computation never touch normalized values directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

QUESTION_TEMPLATE = "Is there a {category} in the image?"
OPEN_ENDED_CAPTION_PROMPT = "Describe this image in detail."
# VizWiz/VQA-style short-answer instruction (same "answer in a word or short
# phrase" convention used across VQA evaluation of LVLMs) -- Stage 7 uses
# this instead of OPEN_ENDED_CAPTION_PROMPT so the model's first generated
# token is the content answer itself, not a templated caption opener.
SHORT_ANSWER_VQA_PROMPT = "What is in this image? Answer the question using a single word or phrase."


@dataclass(frozen=True)
class YesNoDecisionPoint:
    """Where the Yes/No decision actually happens in the generated sequence.

    LLaVA's chat template ends in "ASSISTANT:" with no trailing space, and the
    tokenizer represents " Yes"/" No" as a leading-space token shared by both
    answers, followed by a token that actually diverges (verified empirically via
    `detect_yes_no_decision_point`, never assumed). `prefix_ids` is that shared
    prefix, teacher-forced before reading the decision logit; `yes_token_id` /
    `no_token_id` are the first tokens where the two answers diverge.
    """

    prefix_ids: list[int]
    yes_token_id: int
    no_token_id: int


def load_model(model_id: str, device: str, dtype: torch.dtype = torch.float16):
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def build_prompt_from_text(processor, question_text: str) -> str:
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_text}]}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def build_prompt(processor, category: str) -> str:
    return build_prompt_from_text(processor, QUESTION_TEMPLATE.format(category=category))


def preprocess_to_unit_range(processor, image: Image.Image, device: str) -> torch.Tensor:
    """Resize -> center-crop -> rescale to [0,1]; do_normalize is explicitly
    disabled so the returned tensor is exactly the pre-normalization pixel grid."""
    out = processor.image_processor(images=image, do_normalize=False, return_tensors="pt")
    return out["pixel_values"].to(device)


def normalize(processor, image01: torch.Tensor) -> torch.Tensor:
    """CLIP mean/std normalization -- the last step before the vision tower,
    applied here (and only here) so it stays inside the differentiable graph."""
    mean = torch.tensor(processor.image_processor.image_mean, device=image01.device, dtype=image01.dtype).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device=image01.device, dtype=image01.dtype).view(1, 3, 1, 1)
    return (image01 - mean) / std


def denormalize(processor, pixel_values: torch.Tensor) -> torch.Tensor:
    """Inverse of `normalize`; used to recover the [0,1] pixel grid from a
    processor-produced `pixel_values` tensor without re-running PIL preprocessing."""
    mean = torch.tensor(processor.image_processor.image_mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_processor.image_std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
    return torch.clamp(pixel_values * std + mean, 0.0, 1.0)


def build_inputs_from_text(processor, question_text: str, image: Image.Image, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (input_ids, image01) for one (image, question) pair.

    input_ids MUST come from calling the processor with the image attached: the
    single "<image>" placeholder in the chat-template text gets expanded into one
    token per vision-tower patch, and that expansion count depends on the vision
    config, not just the text -- calling the tokenizer on text alone (bypassing
    the processor) leaves a single placeholder token and the model errors on a
    feature/token-count mismatch. image01 is recovered by denormalizing the same
    call's pixel_values, so it is pixel-identical to what a direct do_normalize=False
    preprocessing call would give (verified in tests), without preprocessing twice.
    """
    prompt = build_prompt_from_text(processor, question_text)
    encoded = processor(text=prompt, images=image, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    image01 = denormalize(processor, encoded["pixel_values"]).to(device)
    return input_ids, image01


def build_inputs(processor, category: str, image: Image.Image, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return build_inputs_from_text(processor, QUESTION_TEMPLATE.format(category=category), image, device)


def detect_yes_no_decision_point(model, processor, image: Image.Image, device: str, top_k: int = 20) -> YesNoDecisionPoint:
    """Empirically determines the shared prefix and diverging Yes/No token ids
    from one real greedy generation, rather than assuming a fixed position --
    a documented, easy-to-get-wrong step for this experiment."""
    input_ids, image01 = build_inputs(processor, "cat", image, device)

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            pixel_values=normalize(processor, image01).to(model.dtype),
            max_new_tokens=4,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

    new_tokens = gen.sequences[0, input_ids.shape[1]:].tolist()
    tok = processor.tokenizer
    decoded_pieces = [tok.decode([t]) for t in new_tokens]

    decision_step = None
    for i, piece in enumerate(decoded_pieces):
        if piece.strip().lower() in ("yes", "no"):
            decision_step = i
            break
    if decision_step is None:
        raise RuntimeError(f"Could not locate a Yes/No token in the greedy continuation: {decoded_pieces!r}")

    prefix_ids = new_tokens[:decision_step]
    decision_logits = gen.scores[decision_step][0]
    topk = torch.topk(decision_logits, top_k)

    yes_token_id = None
    no_token_id = None
    for tok_id in topk.indices.tolist():
        text = tok.decode([tok_id]).strip().lower()
        if text == "yes" and yes_token_id is None:
            yes_token_id = tok_id
        elif text == "no" and no_token_id is None:
            no_token_id = tok_id
    if yes_token_id is None or no_token_id is None:
        raise RuntimeError(
            f"Yes/No token id not found in top-{top_k} decision-step logits "
            f"(yes={yes_token_id}, no={no_token_id}); inspect manually and raise top_k if needed."
        )
    return YesNoDecisionPoint(prefix_ids=prefix_ids, yes_token_id=yes_token_id, no_token_id=no_token_id)


def yes_no_margin(model, processor, decision_point: YesNoDecisionPoint, input_ids: torch.Tensor, image01: torch.Tensor) -> torch.Tensor:
    """logit(yes) - logit(no) at the decision position, differentiable w.r.t. image01.

    `input_ids` is the tokenized prompt (ending in "ASSISTANT:"); the decision
    point's shared prefix is teacher-forced onto it before reading the logit.
    """
    if decision_point.prefix_ids:
        prefix = torch.tensor([decision_point.prefix_ids], device=input_ids.device, dtype=input_ids.dtype)
        full_ids = torch.cat([input_ids, prefix], dim=1)
    else:
        full_ids = input_ids

    pixel_values = normalize(processor, image01).to(model.dtype)
    outputs = model(input_ids=full_ids, pixel_values=pixel_values)
    logits = outputs.logits[0, -1, :]
    return logits[decision_point.yes_token_id] - logits[decision_point.no_token_id]


def yes_no_logits(model, processor, decision_point: YesNoDecisionPoint, input_ids: torch.Tensor, image01: torch.Tensor) -> tuple[float, float]:
    """Non-differentiable clean-image readout of (logit(yes), logit(no)) at the
    decision position -- same teacher-forced position as `yes_no_margin`, but
    returns both raw logits (not just their difference) so the margin s_T =
    yes_logit - no_logit can be reconstructed and reported alongside its two
    components, e.g. for Stage 9's clean (epsilon=0, no-attack) evidence readout.
    """
    with torch.no_grad():
        if decision_point.prefix_ids:
            prefix = torch.tensor([decision_point.prefix_ids], device=input_ids.device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, prefix], dim=1)
        else:
            full_ids = input_ids
        pixel_values = normalize(processor, image01).to(model.dtype)
        outputs = model(input_ids=full_ids, pixel_values=pixel_values)
        logits = outputs.logits[0, -1, :]
        return float(logits[decision_point.yes_token_id]), float(logits[decision_point.no_token_id])


def layerwise_logit_lens(model, processor, decision_point: YesNoDecisionPoint, input_ids: torch.Tensor, image01: torch.Tensor) -> dict[int, tuple[float, float]]:
    """Logit lens (Nostalgebraist): apply the model's OWN final RMSNorm + lm_head
    to every intermediate LLM decoder layer's hidden state at the decision
    position, instead of only the true final layer -- unsupervised (no probe
    fitting), and exact at the true final layer (layer index == number of LLM
    decoder layers) since that reproduces the actual computation `yes_no_logits`
    reads off. Only meaningful for LLM decoder layers: LLaVA's vision tower and
    projector never see the text/target at all, so their hidden states carry no
    target-specific signal to localize (same image -> identical vision/projector
    hidden state regardless of which target is being asked about).

    Returns {layer_index (0=embedding output, 1..N=after each decoder layer):
    (yes_logit, no_logit)}.
    """
    with torch.no_grad():
        if decision_point.prefix_ids:
            prefix = torch.tensor([decision_point.prefix_ids], device=input_ids.device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, prefix], dim=1)
        else:
            full_ids = input_ids
        pixel_values = normalize(processor, image01).to(model.dtype)
        outputs = model(input_ids=full_ids, pixel_values=pixel_values, output_hidden_states=True)

        final_norm = model.model.language_model.norm
        lm_head_weight = model.lm_head.weight
        yes_vec = lm_head_weight[decision_point.yes_token_id]
        no_vec = lm_head_weight[decision_point.no_token_id]

        # HF's output_hidden_states convention (verified empirically against
        # outputs.logits, not assumed): entries 0..N-1 are the RAW pre-final-norm
        # residual stream (embedding output, then each decoder layer's output) --
        # final_norm must be applied to these for a valid logit-lens readout.
        # Entry N (the last one) is already post-final-norm (it IS what produced
        # outputs.logits), so applying final_norm to it again would double-normalize
        # and silently corrupt every "final layer" reading -- caught by Stage 11's
        # own validation check (logit-lens layer N must reproduce Stage 10's real s_T).
        n_entries = len(outputs.hidden_states)
        results: dict[int, tuple[float, float]] = {}
        for layer_idx, h in enumerate(outputs.hidden_states):
            last_token_hidden = h[0, -1, :]
            if layer_idx == n_entries - 1:
                readout = last_token_hidden
            else:
                readout = final_norm(last_token_hidden.unsqueeze(0)).squeeze(0)
            yes_logit = float(torch.dot(readout.to(yes_vec.dtype), yes_vec))
            no_logit = float(torch.dot(readout.to(no_vec.dtype), no_vec))
            results[layer_idx] = (yes_logit, no_logit)
        return results


def generate_greedy_answer(model, processor, input_ids: torch.Tensor, image01: torch.Tensor, max_new_tokens: int = 5) -> str:
    """Authoritative flip check: real greedy decoding (never sampling), decoded text.

    Unlike `yes_no_margin` (a differentiable proxy used only to drive PGD), this
    is what actually determines whether a sample counts as flipped -- it reflects
    the model's real argmax choice over the full vocabulary, not just Yes-vs-No.
    """
    pixel_values = normalize(processor, image01).to(model.dtype)
    with torch.no_grad():
        gen = model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = gen[0, input_ids.shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def is_yes_response(response_text: str) -> bool:
    return response_text.strip().lower().startswith("yes")
