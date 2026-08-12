"""Part V-VII: component functional selectivity test.

For each candidate direction u_k (mean delta_h direction, 20 norm-matched
random directions, and the K fitted PCA components -- plus PLS components if
`decomposition/pls_components.pt` exists, Part VIII) and each fixed
lambda in {0.5, 1.0}, intervene on Layer 19 as

    h'_k = h - lambda * proj_{u_k}(h) ,  proj_u(h) = (h . u) u

applied to every position of the layer's output (same hook granularity as
the prior Exp6 DirectionAblationHook, but projection-based rather than an
unconditional subtraction -- see audit/repository_audit.md item 6 for why
this is a deliberate upgrade, not a modification of existing code).

Per Part VII's discipline, this entire script evaluates ONLY on the internal
VAL split carved out of TRAIN images (configs/data.yaml's dev_val_split) --
never on the final CLEAN TEST split, so the component-selection decision
below cannot leak into Part XI's final model evaluation.

    A. Spurious group  : TRAIN G10_forget VAL images,        Q="sports ball"
    B. Genuine target  : TRAIN GT_target_retain VAL images,  Q="sports ball"
    C. Genuine context : TRAIN GC_context_retain VAL images, Q="baseball bat"
       (same images as A, second question -- by construction, see prepare_data.py)

Baseline (hook disarmed) is computed once per image and reused across all
candidates/lambdas, since it never changes.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/evaluate_components.py \
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

from cooc_diagnostic.coco_index import load_coco_instances  # noqa: E402
from cooc_diagnostic.llava_runtime import (  # noqa: E402
    build_inputs,
    detect_yes_no_decision_point,
    load_model,
    yes_no_logits,
)


class ProjectionInterventionHook:
    """h' = h - lambda * proj_u(h), proj_u(h) = (h . u) u, u unit-norm.

    Applied to the FULL output tensor of language_model.layers[layer_idx - 1]
    (all positions), matching Exp6's DirectionAblationHook granularity -- the
    difference from that hook is that this one is a projection (removes only
    the component of h along u, scaled by lambda), not an unconditional
    subtraction of a fixed vector.
    """

    def __init__(self, model, layer_idx: int):
        self.direction: torch.Tensor | None = None
        self.lambda_: float = 0.0
        self.armed = False
        module = model.language_model.layers[layer_idx - 1]
        self.handle = module.register_forward_hook(self._hook)

    def set(self, direction: torch.Tensor, lambda_: float) -> None:
        self.direction = direction
        self.lambda_ = lambda_

    def _hook(self, module, inputs, output):
        if not self.armed or self.direction is None:
            return output
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        u = self.direction.to(h.dtype).to(h.device)
        coeff = torch.matmul(h, u)  # (..., ) = h . u per position
        proj = coeff.unsqueeze(-1) * u  # (..., hidden_dim)
        h_new = h - self.lambda_ * proj
        return (h_new,) + output[1:] if is_tuple else h_new

    def remove(self):
        self.handle.remove()


def load_val_image_ids(out_dir: Path, role: str) -> list[int]:
    rows = list(csv.DictReader((out_dir / "data" / "train_dev_val_split.csv").open("r", encoding="utf-8")))
    return sorted({int(r["image_id"]) for r in rows if r["role"] == role and r["dev_val"] == "val"})


def build_candidates(out_dir: Path, decomp_config: dict, hidden_dim: int, seed: int, image_ids_g10_train, delta_h_all, dev_ids) -> list[dict]:
    candidates = []

    dev_mask = [iid in dev_ids for iid in image_ids_g10_train]
    d_dev = delta_h_all[dev_mask]
    mean_dir = d_dev.mean(dim=0)
    mean_dir = mean_dir / mean_dir.norm()
    candidates.append({"type": "mean_direction", "id": "mean", "direction": mean_dir})

    n_random = int(decomp_config["intervention"]["n_random_directions"])
    rng = torch.Generator().manual_seed(seed)
    for i in range(n_random):
        v = torch.randn(hidden_dim, generator=rng)
        v = v / v.norm()
        candidates.append({"type": "random_direction", "id": f"random_{i+1}", "direction": v})

    pca_path = out_dir / "decomposition" / "pca_components.pt"
    if pca_path.exists():
        pca = torch.load(pca_path)
        for i in range(pca["k"]):
            candidates.append({"type": "pca_component", "id": f"PC{i+1}", "direction": pca["components"][i]})

    pls_path = out_dir / "decomposition" / "pls_components.pt"
    if pls_path.exists():
        pls = torch.load(pls_path)
        for i in range(pls["k"]):
            candidates.append({"type": "pls_component", "id": f"PLS{i+1}", "direction": pls["components"][i]})

    return candidates


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
    context_category = data_config["context_category"]
    lambda_values = [float(v) for v in decomp_config["intervention"]["lambda_values"]]

    torch.manual_seed(int(data_config["seed"]))

    print(f"[eval-components] loading model {data_config['model_id']} on {device}")
    model, processor = load_model(data_config["model_id"], device)

    val_index = load_coco_instances(data_config["val_annotation_path"])
    image_dir = Path(data_config["val_image_dir"])

    def load_image(image_id: int) -> Image.Image:
        return Image.open(image_dir / val_index.image_filenames[image_id]).convert("RGB")

    g10_val_ids = load_val_image_ids(out_dir, "G10_forget")
    gt_val_ids = load_val_image_ids(out_dir, "GT_target_retain")
    gc_val_ids = load_val_image_ids(out_dir, "GC_context_retain")
    print(f"[eval-components] VAL sizes: G10={len(g10_val_ids)} GT={len(gt_val_ids)} GC={len(gc_val_ids)}")

    probe_image = load_image(g10_val_ids[0])
    decision_point = detect_yes_no_decision_point(model, processor, probe_image, device)

    cached = torch.load(out_dir / "cached_activations" / "layer19_delta_h.pt")
    delta_h_all = cached["delta_h"]
    image_ids_g10_train = cached["image_ids"]
    hidden_dim = delta_h_all.shape[1]

    dev_val_rows = list(csv.DictReader((out_dir / "data" / "train_dev_val_split.csv").open("r", encoding="utf-8")))
    dev_ids = {int(r["image_id"]) for r in dev_val_rows if r["role"] == "G10_forget" and r["dev_val"] == "dev"}

    candidates = build_candidates(out_dir, decomp_config, hidden_dim, int(data_config["seed"]), image_ids_g10_train, delta_h_all, dev_ids)
    print(f"[eval-components] {len(candidates)} candidate directions x {len(lambda_values)} lambdas")

    hook = ProjectionInterventionHook(model, layer_idx)

    def readout(image_id: int, category: str) -> float:
        image = load_image(image_id)
        input_ids, image01 = build_inputs(processor, category, image, device)
        yes_logit, no_logit = yes_no_logits(model, processor, decision_point, input_ids, image01)
        return yes_logit - no_logit

    # --- Baseline (hook disarmed), computed once per (image, question) ---
    hook.armed = False
    baseline_g10 = {iid: readout(iid, target_category) for iid in g10_val_ids}
    baseline_gt = {iid: readout(iid, target_category) for iid in gt_val_ids}
    baseline_gc = {iid: readout(iid, context_category) for iid in gc_val_ids}
    print("[eval-components] baseline (hook disarmed) readouts done")

    # --- Intervention effects ---
    results = []
    for ci, cand in enumerate(candidates):
        direction = cand["direction"].to(device)
        for lam in lambda_values:
            hook.set(direction, lam)
            hook.armed = True

            s_ball_g10 = [readout(iid, target_category) - baseline_g10[iid] for iid in g10_val_ids]
            s_ball_gt = [readout(iid, target_category) - baseline_gt[iid] for iid in gt_val_ids]
            s_bat_gc = [readout(iid, context_category) - baseline_gc[iid] for iid in gc_val_ids]
            hook.armed = False

            delta_spurious = sum(s_ball_g10) / len(s_ball_g10)
            delta_target = sum(s_ball_gt) / len(s_ball_gt)
            delta_context = sum(s_bat_gc) / len(s_bat_gc)
            selectivity = abs(delta_spurious) - abs(delta_target) - abs(delta_context)

            results.append(
                {
                    "candidate_type": cand["type"],
                    "candidate_id": cand["id"],
                    "lambda": lam,
                    "delta_spurious": delta_spurious,
                    "delta_target": delta_target,
                    "delta_context": delta_context,
                    "selectivity": selectivity,
                    "n_g10": len(g10_val_ids),
                    "n_gt": len(gt_val_ids),
                    "n_gc": len(gc_val_ids),
                }
            )
        if (ci + 1) % 5 == 0:
            print(f"[eval-components] {ci + 1}/{len(candidates)} candidates done")
            torch.cuda.empty_cache()
    hook.remove()

    comp_dir = out_dir / "component_intervention"
    comp_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())

    def write(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[eval-components] wrote {len(rows)} rows to {path}")

    write(comp_dir / "intervention_results.csv", [r for r in results if r["candidate_type"] in ("pca_component", "pls_component")])
    write(comp_dir / "mean_direction_results.csv", [r for r in results if r["candidate_type"] == "mean_direction"])
    write(comp_dir / "random_direction_results.csv", [r for r in results if r["candidate_type"] == "random_direction"])

    # --- Component selectivity table (decomposition/component_selectivity.csv) ---
    # Ranking rule: a candidate must be selective at BOTH lambda=0.5 and
    # lambda=1.0 (conservative min, guards against cherry-picking a single
    # lambda) -- fixed rule, decided before inspecting these numbers.
    by_candidate: dict[str, list[dict]] = {}
    for r in results:
        by_candidate.setdefault((r["candidate_type"], r["candidate_id"]), []).append(r)

    selectivity_rows = []
    for (ctype, cid), rows in by_candidate.items():
        min_selectivity = min(r["selectivity"] for r in rows)
        avg = {
            "delta_spurious": sum(r["delta_spurious"] for r in rows) / len(rows),
            "delta_target": sum(r["delta_target"] for r in rows) / len(rows),
            "delta_context": sum(r["delta_context"] for r in rows) / len(rows),
        }
        selectivity_rows.append(
            {
                "candidate_type": ctype,
                "candidate_id": cid,
                "selectivity_min_over_lambda": min_selectivity,
                "mean_delta_spurious": avg["delta_spurious"],
                "mean_delta_target": avg["delta_target"],
                "mean_delta_context": avg["delta_context"],
            }
        )
    selectivity_rows.sort(key=lambda r: r["selectivity_min_over_lambda"], reverse=True)

    decomp_dir = out_dir / "decomposition"
    with (decomp_dir / "component_selectivity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selectivity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selectivity_rows)
    print(f"[eval-components] wrote {decomp_dir / 'component_selectivity.csv'}")

    print("\n[eval-components] ranking (by min-over-lambda selectivity, internal VAL split):")
    for r in selectivity_rows[:8]:
        print(f"  {r['candidate_type']:15s} {r['candidate_id']:10s} selectivity={r['selectivity_min_over_lambda']:+.4f} "
              f"spur={r['mean_delta_spurious']:+.4f} target={r['mean_delta_target']:+.4f} context={r['mean_delta_context']:+.4f}")

    mean_row = next(r for r in selectivity_rows if r["candidate_type"] == "mean_direction")
    best_random = max((r for r in selectivity_rows if r["candidate_type"] == "random_direction"), key=lambda r: r["selectivity_min_over_lambda"])
    best_pca = None
    pca_rows = [r for r in selectivity_rows if r["candidate_type"] == "pca_component"]
    if pca_rows:
        best_pca = max(pca_rows, key=lambda r: r["selectivity_min_over_lambda"])

    print(f"\n[eval-components] mean_direction selectivity={mean_row['selectivity_min_over_lambda']:+.4f}")
    print(f"[eval-components] best random direction ({best_random['candidate_id']}) selectivity={best_random['selectivity_min_over_lambda']:+.4f}")
    if best_pca:
        beats_mean = best_pca["selectivity_min_over_lambda"] > mean_row["selectivity_min_over_lambda"]
        beats_random = best_pca["selectivity_min_over_lambda"] > best_random["selectivity_min_over_lambda"]
        print(f"[eval-components] best PCA component ({best_pca['candidate_id']}) selectivity={best_pca['selectivity_min_over_lambda']:+.4f} "
              f"beats_mean={beats_mean} beats_best_random={beats_random}")
        if not (beats_mean and beats_random):
            print("[eval-components] NOTE: best PCA component does NOT clearly beat both baselines -- "
                  "consider running PLS (decompose_signal.py --with-pls) per Part VIII before finalizing selection.")


if __name__ == "__main__":
    main()
