"""Figure 2: analytic-model vs numerical-truth RMS error over 30 days."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import common

DATA = common.REPO / "data" / "validation" / "model_vs_truth.parquet"


def main() -> None:
    common.apply_style()
    common.need(DATA, "uv run python scripts/run_validation.py")
    df = pd.read_parquet(DATA)

    incs = sorted(df.i_deg.unique())
    fig, axes = plt.subplots(1, len(incs), figsize=(7.2, 2.6), sharey=True)
    if len(incs) == 1:
        axes = [axes]
    for ax, i_deg in zip(axes, incs):
        sub = df[df.i_deg == i_deg]
        for model in ["cw", "ss", "gim_alfriend", "kgd"]:
            g = sub[sub.model == model].sort_values("days")
            if len(g):
                ax.semilogy(g.days, g.rms_err,
                            label=common.MODEL_LABELS.get(model, model))
        ax.set_title(f"i = {i_deg:g}°")
        ax.set_xlabel("time [days]")
    axes[0].set_ylabel("per-orbit RMS position error [m]")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Analytic relative-motion models vs J2–J4 numerical truth "
                 "(1 km E/I-safe formation)", y=1.03, fontsize=9)
    common.save(fig, "fig2_validation")


if __name__ == "__main__":
    main()
