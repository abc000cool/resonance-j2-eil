"""Closed-loop process-noise re-tuning by NIS matching.

`scripts/calibrate_q.py` fits a white-acceleration PSD to *one-step, open-loop*
STM prediction residuals over 3 days.  That underestimates the covariance a
filter needs in the *closed-loop, 30-day* regime, because the STM's error is
secular rather than white: it accumulates faster than a random walk.  The
consequence is an overconfident filter -- median NIS of 1.3e4 for KGD at
cdgps sigma = 5 mm, against an expected value of dim(z) = 3 -- and it is worst
for the *most* accurate STM, which is handed the smallest q_accel.

This script inflates q_accel until the time-averaged NIS matches its expected
value in closed loop (standard covariance matching), anchored at the smallest
swept sigma where measurement noise cannot mask the model error.

Run:  uv run python scripts/tune_q_closedloop.py --filter-model kgd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eilj2.simulate import CALIBRATED_Q_ACCEL, SimConfig, run_sim  # noqa: E402

MEAS_DIM = {"cdgps": 3, "angles": 2, "rf": 2}

BASE = dict(a=7078137.0, e=0.001, i_deg=98.0, dt=60.0, truth="numerical",
            truth_n_zonal=4, filter_kind="ekf", u_max=2.0e-4,
            ctrl_warmup_orbits=2.0, family="along_track", size=1000.0)


def nis_at(filter_model, controller, meas_kind, meas_sigma, q_accel, seeds,
           days, n_jobs=-1):
    """Median NIS/dim over seeds; non-diverged trials only."""
    def one(seed):
        cfg = SimConfig(filter_model=filter_model, controller=controller,
                        meas_kind=meas_kind, meas_sigma=meas_sigma,
                        q_accel=q_accel, duration_days=days, seed=seed, **BASE)
        s = run_sim(cfg).summary
        return s["mean_nis"], bool(s["diverged"]), s["dv_per_year"], s["rms_pos_err"]
    out = Parallel(n_jobs=n_jobs)(delayed(one)(s) for s in seeds)
    good = [(n, dv, rms) for n, div, dv, rms in out if not div and np.isfinite(n)]
    if not good:
        return np.nan, np.nan, np.nan, 0
    nis = np.median([g[0] for g in good]) / MEAS_DIM[meas_kind]
    return nis, np.median([g[1] for g in good]), np.median([g[2] for g in good]), len(good)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-model", default="kgd")
    ap.add_argument("--controller", default="lqr")
    ap.add_argument("--meas-kind", default="cdgps")
    ap.add_argument("--anchor-sigma", type=float, default=0.005,
                    help="smallest swept sigma; model error is unmasked here")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.35, help="tolerance in log10(NIS/dim)")
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    q0 = CALIBRATED_Q_ACCEL[args.filter_model]
    print(f"[tune] {args.filter_model} / {args.controller} / {args.meas_kind} "
          f"@ sigma={args.anchor_sigma:g}, {args.days:g} d, {args.seeds} seeds")
    print(f"[tune] open-loop calibrated q_accel = {q0:.3e}; target NIS/dim = 1.0\n")

    # bracket: NIS/dim falls monotonically with q, so climb until we cross 1
    lo_log = np.log10(q0)
    r_lo, _, _, n = nis_at(args.filter_model, args.controller, args.meas_kind,
                           args.anchor_sigma, 10 ** lo_log, seeds, args.days)
    print(f"  q={10**lo_log:.3e}  NIS/dim={r_lo:9.4g}  (n={n})")
    hi_log = lo_log
    r_hi = r_lo
    while r_hi > 1.0 and hi_log < lo_log + 9:
        hi_log += 1.5
        r_hi, _, _, n = nis_at(args.filter_model, args.controller, args.meas_kind,
                               args.anchor_sigma, 10 ** hi_log, seeds, args.days)
        print(f"  q={10**hi_log:.3e}  NIS/dim={r_hi:9.4g}  (n={n})")
    if r_hi > 1.0:
        print("[tune] FAILED to bracket: NIS stays above target even at huge q.")
        print("       That points at a systematic bias, not process-noise starvation.")
        return

    # bisect in log10(q)
    best = (hi_log, r_hi)
    for i in range(args.iters):
        mid = 0.5 * (lo_log + hi_log)
        r, dv, rms, n = nis_at(args.filter_model, args.controller, args.meas_kind,
                               args.anchor_sigma, 10 ** mid, seeds, args.days)
        print(f"  [{i+1}] q={10**mid:.3e}  NIS/dim={r:9.4g}  dv={dv:8.2f} m/s/yr  "
              f"rms={rms:7.3f} m  (n={n})")
        if not np.isfinite(r):
            break
        if abs(np.log10(r)) < abs(np.log10(best[1])):
            best = (mid, r)
        if abs(np.log10(r)) < args.tol:
            best = (mid, r)
            break
        if r > 1.0:
            lo_log = mid
        else:
            hi_log = mid

    q_star = 10 ** best[0]
    print(f"\n[tune] {args.filter_model}: q_accel {q0:.3e} -> {q_star:.3e} "
          f"(x{q_star/q0:.3g});  NIS/dim = {best[1]:.3g}")
    print(f"[tune] put this in the campaign YAML base as:  q_accel: {q_star:.3e}")


if __name__ == "__main__":
    main()
