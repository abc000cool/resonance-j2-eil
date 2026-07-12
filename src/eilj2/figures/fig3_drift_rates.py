"""Figure 3: uncontrolled drift rate vs inclination at three formation sizes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import common

DATA = common.REPO / "data" / "drift" / "drift_rates.parquet"


def main() -> None:
    common.apply_style()
    common.need(DATA, "uv run python scripts/run_drift.py")
    df = pd.read_parquet(DATA)

    families = sorted(df.family.unique())
    fig, axes = plt.subplots(1, len(families), figsize=(7.4, 2.6), sharey=True)
    for ax, fam in zip(axes, families):
        sub = df[df.family == fam]
        for size in sorted(sub["size"].unique()):
            g = sub[sub["size"] == size].sort_values("i_deg")
            ax.semilogy(g.i_deg, g.drift_m_per_day.abs(),
                        marker="o", ms=3, label=f"L = {size:g} m")
        ax.axvline(63.43, color="gray", ls=":", lw=0.8)
        ax.set_title(fam.replace("_", "-"))
        ax.set_xlabel("chief inclination [deg]")
    axes[0].set_ylabel("|drift rate| [m/day]")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Uncontrolled J2 drift rate (30-day truth, dotted line = "
                 "critical inclination)", y=1.03, fontsize=9)
    common.save(fig, "fig3_drift_rates")


if __name__ == "__main__":
    main()
