"""Stage 11 Exp6: candidate causal coupling test, motivated directly by Exp5.

Exp5 (logit-lens, all 33 LLM readout points) shows the Bat-dependent Ball
signal is small/noisy through layers 0-13, then rises sharply and peaks at
LLM decoder layer 19 (of 32) -- delta_bat_to_ball=1.493 vs delta_sham=-0.030
at that layer, the clearest separation in the whole profile. Layer 19 is
therefore the candidate mediator M (chosen from Exp5's own profile, not
because it is easy to edit).

Intervention (least destructive available: a single-layer, single-direction
subtraction, not a full ablation of the layer or a trained edit):
  1. From the SAME 65 G10 images used throughout, compute the mean hidden-state
     shift v_19 = mean_i[ h_19(original_i) - h_19(bat_removed_i) ] at the
     decision (last prompt token) position -- i.e. exactly the direction Exp5's
     Delta_bat_to_ball already measures the logit-lens projection of, but here
     as a full hidden vector, not a scalar.
  2. A forward hook on language_model.layers[18] (whose output IS
     hidden_states[19]) subtracts v_19 from every position's hidden state
     during generation/scoring, for the ORIGINAL (unmasked) images -- testing
     whether removing this specific internal component, without touching the
     input pixels at all, causally reduces unsupported ball evidence.

Controls (mandatory, per task brief):
  1. Genuine ball evidence (G01/G11) should be largely preserved.
  2. Bat recognition ("Is there a baseball bat in the image?") should be
     largely preserved.
  3. General output should not collapse (qualitative caption check).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp6_causal_intervention.py \
        --config configs/stage11_case_bat_ball.yaml --layer 19 [--device cuda:2]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cooc_diagnostic.coco_index import load_coco_instances
from cooc_diagnostic.llava_runtime import (
    OPEN_ENDED_CAPTION_PROMPT,
    build_inputs,
    detect_yes_no_decision_point,
    generate_greedy_answer,
    is_yes_response,
    load_model,
    yes_no_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp6: candidate causal coupling")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--layer", type=int, required=True, help="Hidden-states index L (candidate mediator, from Exp5)")
    return parser.parse_args()


class DirectionAblationHook:
    """Subtracts a fixed direction vector from a decoder layer's output at every
    position, while the hook is armed. `layer_idx` is a hidden_states index
    (1..num_layers); the corresponding module is language_model.layers[layer_idx - 1]."""

    def __init__(self, model, layer_idx: int, direction: torch.Tensor):
        self.direction = direction
        self.armed = False
        module = model.language_model.layers[layer_idx - 1]
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.armed:
            return output
        return output - self.direction.to(output.dtype).to(output.device)

    def remove(self):
        self.handle.remove()


def get_hidden_at_layer(model, processor, decision_point, input_ids, image01, layer_idx, device) -> torch.Tensor:
    with torch.no_grad():
        if decision_point.prefix_ids:
            prefix = torch.tensor([decision_point.prefix_ids], device=device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, prefix], dim=1)
        else:
            full_ids = input_ids
        from cooc_diagnostic.llava_runtime import normalize

        pixel_values = normalize(processor, image01).to(model.dtype)
        out = model(input_ids=full_ids, pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        return out.hidden_states[layer_idx][0, -1, :].float().cpu()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    device = args.device or config["device"]
    output_dir = Path(config["output_dir"])
    context_name = config["context_category"]
    target_name = config["target_category"]
    layer_idx = args.layer

    val_index = load_coco_instances(config["val_annotation_path"])
    name_to_id = {c.name: c.id for c in val_index.categories}
    context_id, target_id = name_to_id[context_name], name_to_id[target_name]
    image_dir = Path(config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    g10_ids = sorted(iid for iid, cats in val_index.image_categories.items() if context_id in cats and target_id not in cats)

    print(f"[exp6] loading model {config['model_id']} on {device}")
    model, processor = load_model(config["model_id"], device)
    decision_point = detect_yes_no_decision_point(model, processor, load_image(g10_ids[0]), device)

    # --- Step 1: compute mediator direction v_L from G10 original vs bat-removed ---
    counterfactual_dir = output_dir / "counterfactual_images"
    diffs = []
    for image_id in g10_ids:
        orig_path = counterfactual_dir / f"{image_id}_original.png"
        bat_path = counterfactual_dir / f"{image_id}_bat_removed.png"
        if not (orig_path.exists() and bat_path.exists()):
            continue
        input_ids_o, image01_o = build_inputs(processor, target_name, Image.open(orig_path).convert("RGB"), device)
        input_ids_b, image01_b = build_inputs(processor, target_name, Image.open(bat_path).convert("RGB"), device)
        h_o = get_hidden_at_layer(model, processor, decision_point, input_ids_o, image01_o, layer_idx, device)
        h_b = get_hidden_at_layer(model, processor, decision_point, input_ids_b, image01_b, layer_idx, device)
        diffs.append((h_o - h_b))
    v_L = torch.stack(diffs).mean(0)
    print(f"[exp6] mediator direction v_{layer_idx} computed from {len(diffs)} G10 images, ||v||={v_L.norm().item():.3f}")

    hook = DirectionAblationHook(model, layer_idx, v_L.to(device))

    def measure(image_id: int, question_category: str) -> dict:
        image = load_image(image_id)
        input_ids, image01 = build_inputs(processor, question_category, image, device)
        hook.armed = False
        yes0, no0 = yes_no_logits(model, processor, decision_point, input_ids, image01)
        resp0 = generate_greedy_answer(model, processor, input_ids, image01)
        hook.armed = True
        yes1, no1 = yes_no_logits(model, processor, decision_point, input_ids, image01)
        resp1 = generate_greedy_answer(model, processor, input_ids, image01)
        hook.armed = False
        return {
            "s_before": yes0 - no0, "s_after": yes1 - no1,
            "is_yes_before": is_yes_response(resp0), "is_yes_after": is_yes_response(resp1),
        }

    # --- Main test: G10, target question ("unsupported ball evidence") ---
    main_rows = []
    for image_id in g10_ids:
        m = measure(image_id, target_name)
        main_rows.append({"image_id": image_id, "group": "G10", "test": "main_unsupported_ball", **m})
    print(f"[exp6] main test (G10, unsupported ball) done: {len(main_rows)} images")

    # --- Control 1: genuine ball evidence (G01/G11) should be preserved ---
    g01_ids, g11_ids = [], []
    for image_id, cats in val_index.image_categories.items():
        bat, ball = context_id in cats, target_id in cats
        if not bat and ball:
            g01_ids.append(image_id)
        elif bat and ball:
            g11_ids.append(image_id)
    import random

    rng = random.Random(int(config["seed"]))
    rng.shuffle(g01_ids)
    control1_ids = [(iid, "G01") for iid in g01_ids[:40]] + [(iid, "G11") for iid in g11_ids]
    control1_rows = []
    for image_id, group in control1_ids:
        m = measure(image_id, target_name)
        control1_rows.append({"image_id": image_id, "group": group, "test": "control1_genuine_ball", **m})
    print(f"[exp6] control 1 (genuine ball, G01/G11) done: {len(control1_rows)} images")

    # --- Control 2: bat recognition should be preserved ---
    control2_rows = []
    for image_id in g10_ids:
        m = measure(image_id, context_name)
        control2_rows.append({"image_id": image_id, "group": "G10", "test": "control2_bat_recognition", **m})
    print(f"[exp6] control 2 (bat recognition, G10) done: {len(control2_rows)} images")

    all_rows = main_rows + control1_rows + control2_rows
    out_path = output_dir / "exp6_internal_intervention.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[exp6] wrote {len(all_rows)} rows to {out_path}")

    # --- Control 3: general output stability (qualitative caption check) ---
    control3 = []
    for image_id in g10_ids[:10]:
        image = load_image(image_id)
        from cooc_diagnostic.llava_runtime import build_inputs_from_text

        input_ids, image01 = build_inputs_from_text(processor, OPEN_ENDED_CAPTION_PROMPT, image, device)
        hook.armed = False
        cap0 = generate_greedy_answer(model, processor, input_ids, image01, max_new_tokens=40)
        hook.armed = True
        cap1 = generate_greedy_answer(model, processor, input_ids, image01, max_new_tokens=40)
        hook.armed = False
        control3.append({"image_id": image_id, "caption_before": cap0, "caption_after": cap1, "n_words_before": len(cap0.split()), "n_words_after": len(cap1.split())})
    hook.remove()

    # --- Statistics ---
    from scipy import stats as sstats

    def summarize(rows: list[dict], label: str) -> dict:
        s_before = np.array([r["s_before"] for r in rows])
        s_after = np.array([r["s_after"] for r in rows])
        yes_before = np.mean([r["is_yes_before"] for r in rows])
        yes_after = np.mean([r["is_yes_after"] for r in rows])
        wilcoxon = sstats.wilcoxon(s_before, s_after) if len(rows) > 0 else None
        return {
            "label": label, "n": len(rows),
            "mean_s_before": float(s_before.mean()), "mean_s_after": float(s_after.mean()),
            "mean_reduction": float((s_before - s_after).mean()),
            "yes_rate_before": float(yes_before), "yes_rate_after": float(yes_after),
            "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon else None,
        }

    main_summary = summarize(main_rows, "main_unsupported_ball_G10")
    g01_summary = summarize([r for r in control1_rows if r["group"] == "G01"], "control1_genuine_ball_G01")
    g11_summary = summarize([r for r in control1_rows if r["group"] == "G11"], "control1_genuine_ball_G11")
    bat_summary = summarize(control2_rows, "control2_bat_recognition_G10")

    statistics = {
        "layer_intervened": layer_idx,
        "mediator_direction_norm": float(v_L.norm().item()),
        "main_test": main_summary,
        "control1_genuine_ball": {"G01": g01_summary, "G11": g11_summary},
        "control2_bat_recognition": bat_summary,
        "control3_general_stability_sample": control3,
        "checkpoint2_signals": {
            "main_reduces_unsupported_ball": main_summary["mean_reduction"] > 0 and main_summary["wilcoxon_p"] < 0.05,
            "control1_genuine_ball_largely_preserved": abs(g01_summary["mean_reduction"]) < 0.5 * main_summary["mean_reduction"] and abs(g11_summary["mean_reduction"]) < 0.5 * main_summary["mean_reduction"],
            "control2_bat_recognition_largely_preserved": abs(bat_summary["mean_reduction"]) < 0.5 * main_summary["mean_reduction"],
        },
    }
    stats_path = output_dir / "exp6_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp6] wrote statistics to {stats_path}")
    print(f"[exp6] main: reduction={main_summary['mean_reduction']:.3f} p={main_summary['wilcoxon_p']:.4g}")
    print(f"[exp6] control1 G01 reduction={g01_summary['mean_reduction']:.3f}  G11 reduction={g11_summary['mean_reduction']:.3f}")
    print(f"[exp6] control2 bat-recognition reduction={bat_summary['mean_reduction']:.3f}")
    print(f"[exp6] checkpoint2 signals: {statistics['checkpoint2_signals']}")


if __name__ == "__main__":
    main()
