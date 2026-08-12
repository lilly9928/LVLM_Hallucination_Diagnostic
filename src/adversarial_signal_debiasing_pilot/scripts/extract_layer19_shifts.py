"""Part III: extract the adversarially induced Layer-19 representation shift
for every TRAIN G10 image.

Reuses the exact hidden-state extraction convention already validated in
Stage 11 Exp5/Exp6 (see audit/repository_audit.md item 5):
`model(..., output_hidden_states=True).hidden_states[layer_idx][0, -1, :]`,
last token of the teacher-forced "Is there a sports ball in the image?"
decision sequence. h_clean/h_adv are read from the SAME clean/adversarial
image pair already produced by the prior pilot's PGD run (see
scripts/prepare_data.py) -- no new attack, no new model behavior, only a new
readout.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/extract_layer19_shifts.py \
        --data-config configs/data.yaml --decomp-config configs/decomposition.yaml [--device cuda:0]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../CooccurrenceHallucinationDiagnostic/src

from cooc_diagnostic.llava_runtime import (  # noqa: E402
    build_inputs,
    detect_yes_no_decision_point,
    load_model,
    normalize,
)


def get_hidden_at_layer(model, processor, decision_point, input_ids, image01, layer_idx: int, device: str) -> torch.Tensor:
    with torch.no_grad():
        if decision_point.prefix_ids:
            prefix = torch.tensor([decision_point.prefix_ids], device=device, dtype=input_ids.dtype)
            full_ids = torch.cat([input_ids, prefix], dim=1)
        else:
            full_ids = input_ids
        pixel_values = normalize(processor, image01).to(model.dtype)
        out = model(input_ids=full_ids, pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        return out.hidden_states[layer_idx][0, -1, :].float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--decomp-config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    with args.data_config.open("r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)
    with args.decomp_config.open("r", encoding="utf-8") as f:
        decomp_config = yaml.safe_load(f)

    device = args.device
    out_dir = Path(data_config["output_dir"])
    layer_idx = int(decomp_config["layer_idx"])
    target_category = data_config["target_category"]

    torch.manual_seed(int(data_config["seed"]))

    print(f"[extract-l19] loading model {data_config['model_id']} on {device}")
    model, processor = load_model(data_config["model_id"], device)

    adv_rows = list(csv.DictReader((out_dir / "data" / "adversarial_forget_set.csv").open("r", encoding="utf-8")))
    print(f"[extract-l19] {len(adv_rows)} TRAIN G10 images (clean/adv pairs)")

    probe_image = Image.open(adv_rows[0]["clean_image_path"]).convert("RGB")
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)
    print(f"[extract-l19] decision point: yes_id={decision_point.yes_token_id} no_id={decision_point.no_token_id}")

    delta_h_list = []
    metadata_rows = []
    hidden_dim = None
    for i, row in enumerate(adv_rows):
        image_id = int(row["image_id"])
        clean_image = Image.open(row["clean_image_path"]).convert("RGB")
        adv_image = Image.open(row["adv_image_path"]).convert("RGB")

        input_ids_c, image01_c = build_inputs(processor, target_category, clean_image, device)
        input_ids_a, image01_a = build_inputs(processor, target_category, adv_image, device)

        h_clean = get_hidden_at_layer(model, processor, decision_point, input_ids_c, image01_c, layer_idx, device)
        h_adv = get_hidden_at_layer(model, processor, decision_point, input_ids_a, image01_a, layer_idx, device)
        delta_h = h_adv - h_clean
        hidden_dim = delta_h.shape[0]

        delta_h_list.append(delta_h)
        metadata_rows.append(
            {
                "image_id": image_id,
                "clean_s_ball": float(row["clean_s_ball"]),
                "adv_s_ball": float(row["adv_s_ball"]),
                "delta_s_ball": float(row["delta_s_ball"]),
                "hidden_dim": hidden_dim,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"[extract-l19] {i + 1}/{len(adv_rows)} done")
            torch.cuda.empty_cache()

    delta_h_tensor = torch.stack(delta_h_list, dim=0)  # (N, hidden_dim)

    cache_dir = out_dir / "cached_activations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta_h": delta_h_tensor,
            "image_ids": [m["image_id"] for m in metadata_rows],
            "layer_idx": layer_idx,
        },
        cache_dir / "layer19_delta_h.pt",
    )
    with (cache_dir / "layer19_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"[extract-l19] delta_h shape: {tuple(delta_h_tensor.shape)}")
    print(f"[extract-l19] mean ||delta_h||: {delta_h_tensor.norm(dim=1).mean().item():.4f}")
    print(f"[extract-l19] wrote {cache_dir / 'layer19_delta_h.pt'}")
    print(f"[extract-l19] wrote {cache_dir / 'layer19_metadata.csv'}")


if __name__ == "__main__":
    main()
