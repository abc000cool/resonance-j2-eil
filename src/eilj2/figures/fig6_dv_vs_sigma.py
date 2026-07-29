"""Figure 6 (headline): closed-loop delta-V/yr vs navigation accuracy per
controller — the estimation-in-the-loop trade curve."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ..campaign import load_results
from . import common


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.REPO / "data" / "screening"))
    ap.add_argument("--family", default="ei_safe")
    ap.add_argument("--filter-model", default="kgd")
    args = ap.parse_args(argv)

    common.apply_style()
    raw = load_results(args.data)
    raw = raw[(raw.family == args.family)
              & (raw.filter_model == args.filter_model)]
    df = raw[~raw.diverged.astype(bool)]

    kinds = [k for k in ["cdgps", "rf", "angles"] if (raw.meas_kind == k).any()]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.4, 2.8))
    if len(kinds) == 1:
        axes = [axes]
    any_infeasible = False
    for ax, kind in zip(axes, kinds):
        sub = df[df.meas_kind == kind]
        rsub = raw[raw.meas_kind == kind]
        for i, (ctrl, rg) in enumerate(rsub.groupby("controller")):
            color = common.PALETTE[i % len(common.PALETTE)]
            g = sub[sub.controller == ctrl]
            med = g.groupby("meas_sigma")["dv_per_year"].median()
            q1 = g.groupby("meas_sigma")["dv_per_year"].quantile(0.25)
            q3 = g.groupby("meas_sigma")["dv_per_year"].quantile(0.75)
            x = med.index.to_numpy()
            ax.loglog(x, med.to_numpy(), marker="o", ms=3, color=color,
                      label=common.CTRL_LABELS.get(ctrl, ctrl))
            ax.fill_between(x, q1.to_numpy(), q3.to_numpy(), alpha=0.2,
                            color=color)
            # sigma points where every seed diverged: mark, don't omit
            surv = (~rg.diverged.astype(bool)).groupby(rg.meas_sigma).sum()
            dead = surv.index[surv == 0].to_numpy()
            if len(dead):
                any_infeasible = True
                ax.plot(dead, np.full(len(dead), 0.94 - 0.05 * i), "x",
                        ms=5, mew=1.6, color=color, clip_on=False,
                        transform=ax.get_xaxis_transform())
        unit = "rad" if kind == "angles" else "m"
        ax.set_xlabel(rf"$\sigma_\mathrm{{nav}}$ [{unit}]")
        ax.set_title(common.MEAS_LABELS.get(kind, kind))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_ylabel(r"$\Delta V$ [m/s/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    if any_infeasible:
        handles.append(plt.Line2D([], [], marker="x", ls="none", ms=5,
                                  mew=1.6, color="0.35"))
        labels.append("infeasible (100% divergence)")
    fig.legend(handles, labels, fontsize=7, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.suptitle(f"Closed-loop delta-V vs navigation accuracy "
                 f"({args.family}, filter = "
                 f"{common.MODEL_LABELS.get(args.filter_model)})",
                 y=1.03, fontsize=9)
    common.save(fig, "fig6_dv_vs_sigma")


if __name__ == "__main__":
    main()
