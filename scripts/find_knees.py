"""Locate the Pareto knees from campaign results (paper Table 3).

For each (controller, measurement architecture, filter model, geometry), the
knee is the smallest sigma_nav at which the median delta-V/yr exceeds
(1 + threshold) times the tightest-sigma baseline.

Run:  uv run python scripts/find_knees.py data/screening [--threshold 0.10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eilj2.campaign import load_results

GROUP = ["controller", "meas_kind", "filter_model", "family"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_results(args.data_dir)
    df = df[~df["diverged"].astype(bool)]
    med = (df.groupby(GROUP + ["meas_sigma"])["dv_per_year"]
             .median().reset_index())

    rows = []
    for key, g in med.groupby(GROUP):
        g = g.sort_values("meas_sigma")
        base = g["dv_per_year"].iloc[0]
        above = g[g["dv_per_year"] > (1.0 + args.threshold) * base]
        rows.append(dict(zip(GROUP, key),
                         baseline_dv=base,
                         knee_sigma=(above["meas_sigma"].iloc[0]
                                     if len(above) else np.inf)))
    knees = pd.DataFrame(rows)
    out = Path(args.out or (Path(args.data_dir) / "knees.csv"))
    knees.to_csv(out, index=False)
    print(knees.to_string(index=False))
    print(f"\n[knees] wrote {out}")


if __name__ == "__main__":
    main()
