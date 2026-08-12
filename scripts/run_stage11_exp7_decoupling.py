"""Stage 11 Exp7: functional decoupling feasibility summary.

Not a mitigation-method search -- only checks whether the four requirements
(P1-P4) appear feasible for this one intervention (Exp6, layer-19 direction
ablation) on this one pair. P4 (association-knowledge preservation) is not
measured at all, per association_measure_definition.md (written before this
script was run) -- no substitute metric is invented to fill that gap.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp7_decoupling.py \
        --config configs/stage11_case_bat_ball.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp7: decoupling feasibility summary")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])

    exp6 = json.loads((output_dir / "exp6_statistics.json").read_text())

    main_ = exp6["main_test"]
    g01 = exp6["control1_genuine_ball"]["G01"]
    g11 = exp6["control1_genuine_ball"]["G11"]
    bat = exp6["control2_bat_recognition"]

    p1 = {
        "requirement": "P1: spurious Bat->Ball effect decreases",
        "evidence": f"G10 s_ball mean reduction = {main_['mean_reduction']:.3f} (p={main_['wilcoxon_p']:.4g})",
        "met": bool(main_["mean_reduction"] > 0 and main_["wilcoxon_p"] < 0.05),
    }
    p2 = {
        "requirement": "P2: genuine Ball evidence remains",
        "evidence": (
            f"G01 s_ball reduction = {g01['mean_reduction']:.3f} (yes-rate {g01['yes_rate_before']:.2f}->{g01['yes_rate_after']:.2f}); "
            f"G11 s_ball reduction = {g11['mean_reduction']:.3f} (yes-rate {g11['yes_rate_before']:.2f}->{g11['yes_rate_after']:.2f}); "
            f"main G10 reduction = {main_['mean_reduction']:.3f} for reference"
        ),
        "met": bool(abs(g01["mean_reduction"]) < 0.5 * main_["mean_reduction"] and abs(g11["mean_reduction"]) < 0.5 * main_["mean_reduction"]),
    }
    p3 = {
        "requirement": "P3: useful Bat/context information remains",
        "evidence": f"bat-recognition ('Is there a baseball bat?') reduction on G10 = {bat['mean_reduction']:.3f} (yes-rate {bat['yes_rate_before']:.2f}->{bat['yes_rate_after']:.2f})",
        "met": bool(abs(bat["mean_reduction"]) < 0.5 * main_["mean_reduction"]),
    }
    p4 = {
        "requirement": "P4: Bat-Ball association knowledge remains",
        "evidence": "Not measured -- see association_measure_definition.md (written before Exp6/Exp7; no valid measurement was judged feasible within this case study's scope).",
        "met": None,
    }

    rows = [p1, p2, p3, p4]
    out_path = output_dir / "exp7_decoupling_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["requirement", "evidence", "met"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp7] wrote {out_path}")

    n_met = sum(1 for r in [p1, p2, p3] if r["met"])
    if n_met == 3:
        overall = "Selective suppression of unsupported target evidence while preserving genuine target and context information (P1-P3 all met; P4 not measured)."
    elif n_met >= 2:
        overall = "Partial functional decoupling: some but not all of P1-P3 met; see per-requirement evidence."
    else:
        overall = "Functional decoupling not established at this layer with this intervention."

    statistics = {
        "layer_intervened": exp6["layer_intervened"],
        "requirements": rows,
        "n_of_3_measurable_requirements_met": n_met,
        "association_knowledge_p4": "not measured (see association_measure_definition.md)",
        "overall_claim": overall,
    }
    stats_path = output_dir / "exp7_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp7] wrote {stats_path}")
    print(f"[exp7] {n_met}/3 measurable requirements met")
    print(f"[exp7] overall claim: {overall}")


if __name__ == "__main__":
    main()
