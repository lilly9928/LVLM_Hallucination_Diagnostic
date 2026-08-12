"""Orchestrates the full pilot pipeline in the exact order proposed in
audit/repository_audit.md. Each step is a separate script/process (matches
how the pipeline was actually developed and run -- easier to inspect
intermediate outputs and resume from a failure than one monolithic script).

Steps 1-2 (dataset, adversarial exposure) are reuse-only and cheap; steps
3-4 are GPU-bound; PLS (step 4b) only runs if step 4's PCA-only pass shows
poor selectivity (Part VIII decision rule; if you already know PLS is
needed, pass --with-pls from the start to avoid running evaluate_components
twice).

Usage:
    /opt/anaconda3/envs/py3_11/bin/python scripts/run_full_pilot.py --device cuda:0 [--with-pls]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--with-pls", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Stop after component selection (skip Model C training/eval)")
    args = parser.parse_args()
    py = sys.executable

    run([py, "scripts/prepare_data.py", "--config", "configs/data.yaml"])
    run([py, "scripts/extract_layer19_shifts.py", "--data-config", "configs/data.yaml", "--decomp-config", "configs/decomposition.yaml", "--device", args.device])

    decompose_cmd = [py, "scripts/decompose_signal.py", "--data-config", "configs/data.yaml", "--decomp-config", "configs/decomposition.yaml"]
    if args.with_pls:
        decompose_cmd.append("--with-pls")
    run(decompose_cmd)

    run([py, "scripts/evaluate_components.py", "--data-config", "configs/data.yaml", "--decomp-config", "configs/decomposition.yaml", "--device", args.device])

    if args.skip_training:
        print("\n[run-full-pilot] --skip-training set; stopping after component selection.")
        return

    run([py, "scripts/train_adv_decomp_debias.py", "--data-config", "configs/data.yaml", "--decomp-config", "configs/decomposition.yaml",
         "--train-config", "configs/training_adv_decomp.yaml", "--device", args.device])
    run([py, "scripts/evaluate_all_models.py", "--data-config", "configs/data.yaml", "--device", args.device])
    run([py, "analysis/statistics.py", "--data-config", "configs/data.yaml"])
    run([py, "analysis/visualization.py", "--data-config", "configs/data.yaml"])

    print("\n[run-full-pilot] done.")


if __name__ == "__main__":
    main()
