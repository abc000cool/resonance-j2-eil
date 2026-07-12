"""Figure 8: STM-fidelity ablation and certainty-equivalence breakdown."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..campaign import load_results
from . import common


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.REPO / "data" / "ablation"))
    args = ap.parse_args(argv)

    common.apply_style()
    df = load_results(args.data)
    df = df[~df.diverged.astype(bool)]
    sigmas = sorted(df.meas_sigma.unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # ---- panel (a): STM-fidelity ablation ---------------------------------
    sub = df[df.controller.isin(["lqr", "mpc", "impulsive"])]
    med = (sub.groupby(["filter_model", "meas_sigma"])["dv_per_year"]
              .median().reset_index())
    width = 0.35
    models = ["ss", "gim_alfriend", "kgd"]
    xpos = np.arange(len(models))
    for j, sig in enumerate(sigmas):
        vals = [med[(med.filter_model == m) & (med.meas_sigma == sig)]
                ["dv_per_year"].mean() for m in models]
        ax1.bar(xpos + (j - 0.5) * width, vals, width,
                label=rf"$\sigma_\mathrm{{nav}}$ = {sig:g} m")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([common.MODEL_LABELS[m] for m in models])
    ax1.set_ylabel(r"median $\Delta V$ [m/s/yr]")
    ax1.set_title("(a) STM-fidelity ablation", fontsize=9)
    ax1.legend(fontsize=7)

    # ---- panel (b): certainty equivalence vs covariance-aware -------------
    sub = df[df.controller.isin(["lqr", "lqr_cov"]) & (df.filter_model == "kgd")]
    for ctrl, g in sub.groupby("controller"):
        med = g.groupby("meas_sigma")["dv_per_year"].median()
        ax2.semilogx(med.index, med.to_numpy(), marker="o",
                     label=common.CTRL_LABELS.get(ctrl, ctrl))
    ax2.set_xlabel(r"$\sigma_\mathrm{nav}$ [m]")
    ax2.set_ylabel(r"median $\Delta V$ [m/s/yr]")
    ax2.set_title("(b) certainty-equivalence stress test", fontsize=9)
    ax2.legend(fontsize=7)

    common.save(fig, "fig8_ablation")


if __name__ == "__main__":
    main()
