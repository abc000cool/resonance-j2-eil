"""Phase C: uncontrolled drift characterization (paper Figs. 3 and 4).

Part 1: drift rate vs chief inclination x formation size x geometry family.
Part 2: drift-rate contour in (delta-a, delta-ix) offset space at fixed
geometry.

Run:  uv run python scripts/run_drift.py [--days 30] [--n-jobs -1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eilj2.brouwer import mean_to_osc
from eilj2.elements import coe_deputy_from_roe, propagate_mean
from eilj2.frames import eci_to_lvlh
from eilj2.geometry import GEOMETRY_FAMILIES, geometry_roe
from eilj2.metrics import drift_rate
from eilj2.roe_map import lvlh_from_roe
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

INCLINATIONS = [30.0, 45.0, 60.0, 63.4, 70.0, 87.0, 98.0]
SIZES = [100.0, 1000.0, 10000.0]


def _drift_run(i_deg: float, family: str, size: float,
               droe_offset: np.ndarray | None, days: float, dt: float) -> dict:
    coe_c = np.array([7078137.0, 0.001, np.deg2rad(i_deg), 0.3, 0.2, 0.1])
    a = coe_c[0]
    droe_des = geometry_roe(family, size, a)
    droe0 = droe_des + (droe_offset if droe_offset is not None else 0.0)
    coe_d = coe_deputy_from_roe(coe_c, droe0)
    tw = TwoSatTruth(TruthConfig(n_zonal=4, method="RK4"))

    y = state_from_coe(mean_to_osc(coe_c), mean_to_osc(coe_d))
    coe_mean = coe_c.copy()
    s_des = droe_des * a
    n_steps = int(days * 86400.0 / dt)
    t_arr, err_arr = [], []
    for k in range(1, n_steps + 1):
        y = tw.step(y, dt)
        coe_mean = propagate_mean(coe_mean, dt)
        x_true = eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=4)
        x_ref = lvlh_from_roe(s_des, coe_mean)
        t_arr.append(k * dt)
        err_arr.append(np.linalg.norm(x_true[:3] - x_ref[:3]))
    rate = drift_rate(np.array(t_arr), np.array(err_arr))
    out = dict(i_deg=i_deg, family=family, size=size,
               drift_m_per_day=rate * 86400.0,
               final_err=float(err_arr[-1]))
    if droe_offset is not None:
        out["da_m"] = float(droe_offset[0] * a)
        out["dix_m"] = float(droe_offset[4] * a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default="data/drift")
    ap.add_argument("--contour-n", type=int, default=9,
                    help="grid points per axis for the (da, dix) contour")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- part 1: inclination x size x family map (Fig. 3) -----------------
    f1 = out / "drift_rates.parquet"
    if f1.exists():
        print(f"[drift] skip (exists): {f1}")
    else:
        jobs = [(i, fam, L, None) for i in INCLINATIONS for fam in GEOMETRY_FAMILIES
                for L in SIZES]
        print(f"[drift] part 1: {len(jobs)} uncontrolled 30-day runs ...")
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_drift_run)(i, fam, L, off, args.days, args.dt)
            for i, fam, L, off in jobs)
        pd.DataFrame(rows).to_parquet(f1, index=False)
        print(f"[drift] wrote {f1}")

    # ---- part 2: (da, dix) contour at fixed geometry (Fig. 4) -------------
    f2 = out / "contour.parquet"
    if f2.exists():
        print(f"[drift] skip (exists): {f2}")
    else:
        a = 7078137.0
        grid = np.linspace(-100.0, 100.0, args.contour_n)  # meters
        jobs = []
        for da_m in grid:
            for dix_m in grid:
                off = np.zeros(6)
                off[0] = da_m / a
                off[4] = dix_m / a
                jobs.append(off)
        print(f"[drift] part 2: {len(jobs)} contour runs ...")
        rows = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(_drift_run)(98.0, "ei_safe", 1000.0, off, args.days, args.dt)
            for off in jobs)
        pd.DataFrame(rows).to_parquet(f2, index=False)
        print(f"[drift] wrote {f2}")


if __name__ == "__main__":
    main()
