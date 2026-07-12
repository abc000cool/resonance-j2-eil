"""Phase D: perfect-state controller Pareto baselines (paper Fig. 5).

Sweeps each controller's tightness knob with perfect navigation and records
the (delta-V/yr, RMS formation-keeping error) Pareto curve.

Run:  uv run python scripts/run_perfect_state.py [--truth stm|numerical]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eilj2.simulate import SimConfig, run_sim

SWEEPS = {
    "lqr": [("lqr_r_weight", w) for w in np.logspace(9.0, 13.5, 20)],
    "mpc": [("mpc_state_weight", w) for w in np.logspace(-5.0, -1.0, 20)],
    "impulsive": [("imp_deadband", d) for d in np.linspace(0.0, 60.0, 20)],
}


def _run(controller: str, param: str, value: float, truth: str, days: float) -> dict:
    kw = {param: float(value)}
    cfg = SimConfig(controller=controller, filter_kind="perfect",
                    truth=truth, truth_model="kgd", duration_days=days,
                    family="ei_safe", size=1000.0, i_deg=98.0, seed=7, **kw)
    try:
        s = run_sim(cfg).summary
        return dict(controller=controller, param=param, value=float(value),
                    dv_per_year=s["dv_per_year"], rms_pos_err=s["rms_pos_err"],
                    p99_pos_err=s["p99_pos_err"], diverged=s["diverged"])
    except Exception as exc:
        return dict(controller=controller, param=param, value=float(value),
                    dv_per_year=np.nan, rms_pos_err=np.nan, p99_pos_err=np.nan,
                    diverged=True, error=str(exc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", choices=["stm", "numerical"], default="stm",
                    help="stm for the quick pass; numerical for paper-final")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default="data/perfect_state")
    ap.add_argument("--controllers", nargs="+",
                    default=["lqr", "mpc", "impulsive"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = [(c, p, v) for c in args.controllers for p, v in SWEEPS[c]]
    print(f"[perfect] {len(jobs)} runs (truth={args.truth}) ...")
    rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_run)(c, p, v, args.truth, args.days) for c, p, v in jobs)
    df = pd.DataFrame(rows)
    f = out / f"pareto_{args.truth}.parquet"
    df.to_parquet(f, index=False)
    print(f"[perfect] wrote {f}")
    print(df.groupby("controller")[["dv_per_year", "rms_pos_err"]].describe())


if __name__ == "__main__":
    main()
