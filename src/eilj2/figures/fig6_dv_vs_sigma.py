"""Figure 6 (headline): closed-loop delta-V/yr vs navigation accuracy per
controller — the estimation-in-the-loop trade curve."""

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
    ap.add_argument("--filter-model", default="kgd")
    args = ap.parse_args(argv)

    common.apply_style()
    df = load_results(args.data)
    df = df[(~df.diverged.astype(bool)) & (df.family == args.family)
            & (df.filter_model == args.filter_model)]

    kinds = [k for k in ["cdgps", "rf", "angles"] if (df.meas_kind == k).any()]
    fig, axes = plt.subplots(1, len(kinds), figsize=(7.4, 2.8))
    if len(kinds) == 1:
        axes = [axes]
    for ax, kind in zip(axes, kinds):
        sub = df[df.meas_kind == kind]
        for ctrl, g in sub.groupby("controller"):
            med = g.groupby("meas_sigma")["dv_per_year"].median()
            q1 = g.groupby("meas_sigma")["dv_per_year"].quantile(0.25)
            q3 = g.groupby("meas_sigma")["dv_per_year"].quantile(0.75)
            x = med.index.to_numpy()
            ax.loglog(x, med.to_numpy(), marker="o", ms=3,
                      label=common.CTRL_LABELS.get(ctrl, ctrl))
            ax.fill_between(x, q1.to_numpy(), q3.to_numpy(), alpha=0.2)
        unit = "rad" if kind == "angles" else "m"
        ax.set_xlabel(rf"$\sigma_\mathrm{{nav}}$ [{unit}]")
        ax.set_title(common.MEAS_LABELS.get(kind, kind))
    axes[0].set_ylabel(r"$\Delta V$ [m/s/yr]")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"Closed-loop delta-V vs navigation accuracy "
                 f"({args.family}, filter = "
                 f"{common.MODEL_LABELS.get(args.filter_model)})",
                 y=1.03, fontsize=9)
    common.save(fig, "fig6_dv_vs_sigma")


if __name__ == "__main__":
    main()
