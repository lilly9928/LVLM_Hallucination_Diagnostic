"""Stage 7: targeted keyword attack on short-answer VQA (VizWiz-style
"answer in a word or short phrase" format), not open-ended captioning.

History: the first two designs targeted LLaVA's free-form "Describe this
image in detail." caption -- a single forced-continuation loss, then a
Show-and-Fool-style multi-position keyword loss (Chen et al., "Attacking
Visual Language Grounding with Adversarial Examples: A Case Study on Neural
Image Captioning"). Both hit the same wall empirically: LLaVA's open-ended
captions open with a near-fixed template ("The image features/shows/is..."),
so no small set of early positions reliably carries the target token, and
positions further in require the free decode's own greedy path to
coincidentally match a forced conditioning prefix -- which frequently doesn't
hold. Pilot testing showed PGD could always drive a multi-position margin to
0 (proxy "success") while only ~47-60% of those attacks actually produced the
category in a real free-decode caption.

Switching the READOUT from free captioning to a short-answer VQA prompt (see
llava_runtime.SHORT_ANSWER_VQA_PROMPT) removes the problem at its root: with
"answer in a word or short phrase," the model's very first generated token
IS the content answer, not a templated scene-setting phrase. This makes the
margin structurally identical to Stage 3's yes_no_margin -- a single fixed
position, no forced continuation, no periodic re-generation, no position
search -- just generalized from a binary (yes vs no) contrast to an
open-vocabulary (target vs every other token) contrast.
"""

from __future__ import annotations

import torch

from cooc_diagnostic.llava_runtime import generate_greedy_answer, normalize
from cooc_diagnostic.mention_detection import text_mentions_category


def category_first_token_id(processor, category: str) -> int:
    """First subword token of ' {category}'. COCO has several multi-word
    categories (traffic light, teddy bear, ...); targeting only the first
    token is the same single-token simplification the Yes/No decision point
    already relies on. This can only under-count attack success relative to
    the authoritative check (which requires the full phrase via
    text_mentions_category), never over-count it.

    Verified empirically (same discipline as detect_yes_no_decision_point):
    tokenizing f" {category}" IN ISOLATION prepends a standalone SentencePiece
    space-marker token that decodes to "" (e.g. "snowboard" -> ["", "snow",
    "board"]) -- an isolated-string tokenization artifact that does NOT occur
    when the same text is embedded after real preceding context (confirmed by
    tokenizing "ASSISTANT: {category}" and comparing). Skipping empty-decode
    tokens recovers the real first content token in both cases.
    """
    tok = processor.tokenizer
    ids = tok(f" {category}", add_special_tokens=False)["input_ids"]
    for token_id in ids:
        if tok.decode([token_id]).strip():
            return token_id
    raise RuntimeError(f"no non-empty-decode token found for category {category!r}: {ids}")


def short_answer_margin(model, processor, input_ids: torch.Tensor, category_token_id: int, image01: torch.Tensor) -> torch.Tensor:
    """logit(category_token) - max(logit of every other token), read at the
    position right after input_ids (no concatenation, no inserted text) --
    differentiable wrt image01 via llava_runtime.normalize. Structurally
    identical to yes_no_margin, generalized from a fixed yes/no pair to an
    open-vocabulary target.
    """
    pixel_values = normalize(processor, image01).to(model.dtype)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    logits = outputs.logits[0, -1, :]
    target_logit = logits[category_token_id]
    other = logits.clone()
    other[category_token_id] = -float("inf")
    return target_logit - other.max()


def evaluate_short_answer_response(
    model, processor, input_ids: torch.Tensor, category: str, image01: torch.Tensor, max_new_tokens: int = 10
) -> dict:
    """Authoritative flip check: real unconstrained greedy generation (never
    the single logit read used by short_answer_margin) checked against the
    whole answer text via mention_detection.text_mentions_category.
    """
    text = generate_greedy_answer(model, processor, input_ids, image01, max_new_tokens=max_new_tokens)
    mentioned = text_mentions_category(text, category)
    return {"response_text": text, "mentioned": mentioned, "flipped": mentioned}
