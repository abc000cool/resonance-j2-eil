"""Figure 7: the (delta-V, control precision, sigma_nav) Pareto surface —
the direct extension of Koenig & D'Amico 2018 to a surface."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..campaign import load_results
from . import common


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.REPO / "data" / "screening"))
    ap.add_argument("--family", default="ei_safe")
    ap.add_argument("--meas-kind", default="cdgps")
    ap.add_argument("--filter-model", default="kgd")
    args = ap.parse_args(argv)

    common.apply_style()
    df = load_results(args.data)
    df = df[(~df.diverged.astype(bool)) & (df.family == args.family)
            & (df.meas_kind == args.meas_kind)
            & (df.filter_model == args.filter_model)]

    ctrls = sorted(df.controller.unique())
    fig, axes = plt.subplots(1, len(ctrls), figsize=(7.4, 2.8), sharey=True)
    if len(ctrls) == 1:
        axes = [axes]
    med = (df.groupby(["controller", "meas_sigma"])
             [["dv_per_year", "rms_pos_err"]].median().reset_index())
    for ax, ctrl in zip(axes, ctrls):
        g = df[df.controller == ctrl]
        sc = ax.scatter(g.meas_sigma, g.rms_pos_err,
                        c=g.dv_per_year, s=8, cmap="magma",
                        norm=plt.matplotlib.colors.LogNorm())
        m = med[med.controller == ctrl].sort_values("meas_sigma")
        ax.plot(m.meas_sigma, m.rms_pos_err, "k-", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\sigma_\mathrm{nav}$ [m]")
        ax.set_title(common.CTRL_LABELS.get(ctrl, ctrl))
    axes[0].set_ylabel("RMS control precision [m]")
    fig.colorbar(sc, ax=axes, label=r"$\Delta V$ [m/s/yr]", shrink=0.85)
    fig.suptitle("Pareto surface: delta-V over (nav accuracy, control "
                 "precision)", y=1.03, fontsize=9)
    common.save(fig, "fig7_pareto_surface")


if __name__ == "__main__":
    main()
