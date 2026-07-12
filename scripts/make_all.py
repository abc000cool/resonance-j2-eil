"""Windows-friendly pipeline driver (mirrors the Snakefile DAG).

    uv run python scripts/make_all.py --phase validation
    uv run python scripts/make_all.py --phase all

Phases: validation, drift, perfect, screening, ablation, figures, all.
Completed stages are skipped when their outputs exist (delete the parquet /
data directory to force a re-run).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n=== {' '.join(cmd)}")
    subprocess.run([sys.executable, *cmd], check=True, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["validation", "drift", "perfect", "screening",
                             "ablation", "figures", "all"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    p = args.phase

    if p in ("validation", "all"):
        if not (REPO / "data/validation/model_vs_truth.parquet").exists():
            run(["scripts/run_validation.py"])
    if p in ("drift", "all"):
        run(["scripts/run_drift.py", "--n-jobs", str(args.n_jobs)])
    if p in ("perfect", "all"):
        if not (REPO / "data/perfect_state/pareto_stm.parquet").exists():
            run(["scripts/run_perfect_state.py", "--truth", "stm",
                 "--n-jobs", str(args.n_jobs)])
    if p in ("screening", "all"):
        run(["-m", "eilj2.campaign", "config/screening.yaml",
             "--n-jobs", str(args.n_jobs)])
    if p in ("ablation", "all"):
        run(["-m", "eilj2.campaign", "config/ablation.yaml",
             "--n-jobs", str(args.n_jobs)])
    if p in ("figures", "all"):
        for mod in ["fig1_geometry", "fig2_validation", "fig3_drift_rates",
                    "fig4_drift_contour", "fig5_perfect_pareto",
                    "fig6_dv_vs_sigma", "fig7_pareto_surface", "fig8_ablation"]:
            try:
                run(["-m", f"eilj2.figures.{mod}"])
            except subprocess.CalledProcessError as exc:
                print(f"[make_all] {mod} skipped: {exc}")


if __name__ == "__main__":
    main()
