"""Locate the navigation-accuracy "knee" per (controller, meas, filter, family).

Robust version.  The original metric — first sigma whose median dv/yr exceeds
1.1x the smallest-sigma baseline — proved fragile on the pruned 4-5 point
sigma grid (screening and full tier agreed on the verdict in only 55% of
cells), and it silently ignored two failure modes:

  * infeasible points: every seed diverged, so the cell-sigma has no median
    at all and vanished from the analysis rather than being reported;
  * non-rising trends: with an inconsistent filter the median dv *falls*
    with sigma (see Fig. 9), and "no knee found" is then a symptom of the
    filter, not a property of the sensor.

This script therefore reports, per cell:
  status          rising / flat / falling / infeasible
                  (spearman(sigma, dv) over feasible points, +/-0.5 cutoffs)
  n_sigma, n_infeasible, baseline_sigma, baseline_dv
  knee_p05/p10/p25/p50    first feasible sigma with dv > (1+thr) x baseline,
                          for thr = 5/10/25/50% (inf = never within sweep)
  knee_stable     True when the 5% and 50% knees agree to within one grid
                  step — i.e. the verdict does not hinge on the threshold
  knee_fit        10%-rise point of a fitted dv0*(1+(sigma/sk)^p) curve,
                  only attempted for status == rising with >= 4 points

Run:  uv run python scripts/find_knees.py data/full [--min-survivors 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eilj2.campaign import load_results  # noqa: E402

GROUP = ["controller", "meas_kind", "filter_model", "family"]
THRESHOLDS = [0.05, 0.10, 0.25, 0.50]


def _threshold_knee(sig: np.ndarray, dv: np.ndarray, thr: float) -> float:
    above = sig[dv > (1.0 + thr) * dv[0]]
    return float(above[0]) if len(above) else np.inf


def _fitted_knee(sig: np.ndarray, dv: np.ndarray) -> float:
    """10%-rise point of dv0*(1+(sigma/sk)^p), fit to the medians in log-x."""
    from scipy.optimize import curve_fit

    def f(log_s, dv0, log_sk, p):
        return dv0 * (1.0 + np.exp(p * (log_s - log_sk)))

    try:
        popt, _ = curve_fit(
            f, np.log(sig), dv,
            p0=[dv[0], np.log(sig[-1]), 2.0],
            bounds=([0.0, np.log(sig[0]) - 10.0, 0.2],
                    [np.inf, np.log(sig[-1]) + 10.0, 10.0]),
            maxfev=20000)
    except Exception:
        return np.nan
    dv0, log_sk, p = popt
    return float(np.exp(log_sk + np.log(0.1) / p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--min-survivors", type=int, default=10,
                    help="feasibility floor: sigma points with fewer "
                         "non-diverged trials are marked infeasible")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_results(args.data_dir)
    stats = (df.assign(ok=~df["diverged"].astype(bool))
               .groupby(GROUP + ["meas_sigma"])
               .agg(n_surv=("ok", "sum"), n_tot=("ok", "size")))
    med = (df[~df["diverged"].astype(bool)]
           .groupby(GROUP + ["meas_sigma"])["dv_per_year"].median())

    rows = []
    for key, g in stats.groupby(GROUP):
        g = g.droplevel(GROUP).sort_index()
        feasible = g.index[g["n_surv"] >= args.min_survivors]
        row = dict(zip(GROUP, key),
                   n_sigma=len(g),
                   n_infeasible=int((g["n_surv"] < args.min_survivors).sum()))
        if len(feasible) == 0:
            row.update(status="infeasible", baseline_sigma=np.nan,
                       baseline_dv=np.nan, knee_stable=False, knee_fit=np.nan,
                       knee_fit_extrapolated=False,
                       **{f"knee_p{int(t*100):02d}": np.nan for t in THRESHOLDS})
            rows.append(row)
            continue

        dv = np.array([med.loc[key + (s,)] for s in feasible])
        sig = feasible.to_numpy(float)
        rho = pd.Series(sig).corr(pd.Series(dv), method="spearman") \
            if len(sig) >= 3 else np.nan
        status = ("rising" if rho >= 0.5 else
                  "falling" if rho <= -0.5 else "flat") \
            if np.isfinite(rho) else "flat"

        knees = {f"knee_p{int(t*100):02d}": _threshold_knee(sig, dv, t)
                 for t in THRESHOLDS}
        k_lo, k_hi = knees["knee_p05"], knees["knee_p50"]
        if np.isinf(k_lo) and np.isinf(k_hi):
            stable = True                      # agree: no knee in range
        elif np.isinf(k_lo) or np.isinf(k_hi):
            stable = False
        else:
            step = np.median(np.diff(np.log10(sig))) if len(sig) > 1 else 1.0
            stable = abs(np.log10(k_hi) - np.log10(k_lo)) <= step + 1e-9

        kfit = (_fitted_knee(sig, dv)
                if status == "rising" and len(sig) >= 4 else np.nan)
        row.update(status=status,
                   baseline_sigma=float(sig[0]), baseline_dv=float(dv[0]),
                   knee_stable=bool(stable), knee_fit=kfit,
                   knee_fit_extrapolated=bool(np.isfinite(kfit)
                                              and not sig[0] <= kfit <= sig[-1]),
                   **knees)
        rows.append(row)

    knees = pd.DataFrame(rows)
    out = Path(args.out or (Path(args.data_dir) / "knees.csv"))
    knees.to_csv(out, index=False)

    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(knees.to_string(index=False))
    print(f"\n[knees] status counts: {knees['status'].value_counts().to_dict()}")
    print(f"[knees] threshold-stable verdicts: {int(knees['knee_stable'].sum())}"
          f"/{len(knees)}")
    print(f"[knees] cells with infeasible sigma points: "
          f"{int((knees['n_infeasible'] > 0).sum())}")
    print(f"[knees] wrote {out}")


if __name__ == "__main__":
    main()
