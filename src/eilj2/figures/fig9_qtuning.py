"""Fig. 9 -- filter consistency governs the apparent navigation/delta-V trade.

Left column: median NIS vs sigma.  A consistent filter sits at dim(z) = 3
(dashed).  The open-loop-calibrated q_accel leaves KGD overconfident by three
to four orders of magnitude; closed-loop NIS matching restores consistency
across the whole sweep.

Right column: the consequence.  With the overconfident filter, both delta-V
*and* position error *fall* as navigation degrades -- physically backwards,
and the reason a rising-cost "knee" is almost never detected.  With the
matched filter the LQR trade is restored (cost rises with sigma) and MPC is
revealed to trade accuracy rather than fuel.

Run:  uv run python -m eilj2.figures.fig9_qtuning
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..campaign import load_results
from . import common

Q_ORIG = {"kgd": 1.0e-14, "gim_alfriend": 8.0e-13}
Q_TUNED = {"kgd": 3.162e-10, "gim_alfriend": 4.499e-12}


def _agg(df, ctrl):
    d = df[(df.controller == ctrl) & (~df["diverged"].astype(bool))]
    return d.groupby("meas_sigma").agg(
        dv=("dv_per_year", "median"),
        dv_lo=("dv_per_year", lambda s: s.quantile(0.25)),
        dv_hi=("dv_per_year", lambda s: s.quantile(0.75)),
        rms=("rms_pos_err", "median"),
        nis=("mean_nis", "median"),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-data", default=str(common.REPO / "data" / "full"))
    ap.add_argument("--tuned-data", default=str(common.REPO / "data" / "corrected_q"))
    ap.add_argument("--filter-model", default="kgd")
    ap.add_argument("--family", default="along_track")
    ap.add_argument("--meas-kind", default="cdgps")
    args = ap.parse_args(argv)

    common.apply_style()

    def sel(path):
        d = load_results(path)
        return d[(d.filter_model == args.filter_model) & (d.family == args.family)
                 & (d.meas_kind == args.meas_kind)]

    orig, tuned = sel(args.orig_data), sel(args.tuned_data)
    ctrls = ["lqr", "mpc"]
    dim = {"cdgps": 3, "angles": 2, "rf": 2}[args.meas_kind]

    fig, axes = plt.subplots(len(ctrls), 2, figsize=(7.2, 5.2), sharex=True)
    for row, ctrl in enumerate(ctrls):
        a, b = _agg(orig, ctrl), _agg(tuned, ctrl)

        ax = axes[row, 0]
        ax.loglog(a.index, a.nis, "o-", color=common.PALETTE[1],
                  label=rf"open-loop $q$ = {Q_ORIG[args.filter_model]:.1e}")
        ax.loglog(b.index, b.nis, "s-", color=common.PALETTE[0],
                  label=rf"NIS-matched $q$ = {Q_TUNED[args.filter_model]:.1e}")
        ax.axhline(dim, ls="--", color="k", lw=1.0)
        ax.text(0.98, 0.06, rf"consistent: $\dim(z)={dim}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="k")
        ax.set_ylabel(f"{common.CTRL_LABELS[ctrl]}\n\nmedian NIS")
        if row == 0:
            ax.set_title("filter consistency")
            ax.legend(fontsize=7, loc="upper right")

        ax = axes[row, 1]
        ax.semilogx(a.index, a.dv, "o-", color=common.PALETTE[1])
        ax.fill_between(a.index, a.dv_lo, a.dv_hi, alpha=0.15,
                        color=common.PALETTE[1])
        ax.semilogx(b.index, b.dv, "s-", color=common.PALETTE[0])
        ax.fill_between(b.index, b.dv_lo, b.dv_hi, alpha=0.15,
                        color=common.PALETTE[0])
        ax.set_ylabel(r"$\Delta V$ [m/s/yr]")
        if row == 0:
            ax.set_title("cost vs navigation accuracy")
        # annotate the trend sign over the originally swept range
        for s, col, dy in ((a, common.PALETTE[1], -0.16), (b, common.PALETTE[0], 0.0)):
            m = s[s.index <= 0.1]
            if len(m) < 3:
                continue
            rho = pd.Series(m.index).corr(pd.Series(m.dv.to_numpy()),
                                          method="spearman")
            ax.annotate(rf"$\rho={rho:+.2f}$", xy=(0.62, 0.90 + dy),
                        xycoords="axes fraction", fontsize=7.5, color=col)

    for ax in axes[-1]:
        unit = "rad" if args.meas_kind == "angles" else "m"
        ax.set_xlabel(rf"$\sigma_\mathrm{{nav}}$ [{unit}]")
    fig.suptitle(
        f"Process-noise calibration governs the apparent trade "
        f"({common.MODEL_LABELS.get(args.filter_model)} filter, "
        f"{common.MEAS_LABELS[args.meas_kind]}, {args.family}, $N=100$)",
        y=0.99, fontsize=9)
    common.save(fig, "fig9_qtuning")


if __name__ == "__main__":
    main()
