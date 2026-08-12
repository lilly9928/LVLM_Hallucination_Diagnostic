"""Stage 11 Exp3: boundary consistency between clean s_ball (Exp2) and
epsilon* (Exp1), joined on image_id. NOT independent causal evidence -- epsilon*
is defined through an attack objective on the same Yes/No margin, so a strong
negative relationship is partly mechanical. Reported as a consistency link only.

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_stage11_exp3_boundary_consistency.py \
        --config configs/stage11_case_bat_ball.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 11 Exp3: boundary consistency")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output_dir = Path(config["output_dir"])

    with (output_dir / "exp1_epsilon_star.csv").open("r", encoding="utf-8") as f:
        eps_rows = {int(r["image_id"]): r for r in csv.DictReader(f)}
    with (output_dir / "exp2_clean_evidence.csv").open("r", encoding="utf-8") as f:
        evidence_rows = {int(r["image_id"]): r for r in csv.DictReader(f) if r["group"] in ("G00", "G10")}

    joined = []
    for image_id, ev in evidence_rows.items():
        eps = eps_rows.get(image_id)
        if eps is None:
            continue
        eps_star = eps["epsilon_star"]
        joined.append(
            {
                "image_id": image_id,
                "group": ev["group"],
                "s_ball": float(ev["s_ball"]),
                "clean_is_yes": ev["prediction"] == "Yes",
                "epsilon_star": float(eps_star) if eps_star not in ("", "None") else None,
                "attack_status": eps["attack_status"],
            }
        )
    print(f"[exp3] joined {len(joined)} rows (of {len(evidence_rows)} clean-evidence rows in G00/G10)")

    out_path = output_dir / "exp3_boundary_consistency.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
        writer.writeheader()
        writer.writerows(joined)
    print(f"[exp3] wrote {len(joined)} rows to {out_path}")

    def correlate(rows: list[dict], label: str) -> dict:
        obs = [r for r in rows if r["epsilon_star"] is not None]
        if len(obs) < 3:
            return {"label": label, "n": len(obs), "note": "too few observed (non-censored) points"}
        x = np.array([r["s_ball"] for r in obs])
        y = np.array([r["epsilon_star"] for r in obs])
        pear = sstats.pearsonr(x, y)
        spear = sstats.spearmanr(x, y)
        return {
            "label": label,
            "n": len(obs),
            "n_censored_excluded": len(rows) - len(obs),
            "pearson_r": float(pear.statistic),
            "pearson_p": float(pear.pvalue),
            "spearman_rho": float(spear.statistic),
            "spearman_p": float(spear.pvalue),
        }

    all_samples = correlate(joined, "all_samples")
    excl_already_yes = correlate([r for r in joined if not r["clean_is_yes"]], "excluding_already_yes")

    statistics = {
        "all_samples": all_samples,
        "excluding_already_yes": excl_already_yes,
        "interpretation": (
            "Consistency link only, not independent causal evidence: epsilon* is derived from the "
            "same yes/no margin that defines s_ball, so part of any negative correlation is mechanical."
        ),
    }
    stats_path = output_dir / "exp3_statistics.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(statistics, f, indent=2)
    print(f"[exp3] wrote statistics to {stats_path}")
    print(f"[exp3] all_samples: pearson_r={all_samples.get('pearson_r')} spearman_rho={all_samples.get('spearman_rho')}")


if __name__ == "__main__":
    main()
