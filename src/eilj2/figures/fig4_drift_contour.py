"""Figure 4: drift-rate contour in (delta-a, delta-ix) offset space."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common

DATA = common.REPO / "data" / "drift" / "contour.parquet"


def main() -> None:
    common.apply_style()
    common.need(DATA, "uv run python scripts/run_drift.py")
    df = pd.read_parquet(DATA)

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    tc = ax.tricontourf(df.da_m, df.dix_m, np.abs(df.drift_m_per_day),
                        levels=14, cmap="viridis")
    fig.colorbar(tc, ax=ax, label="|drift rate| [m/day]")
    ax.set_xlabel(r"$a\,\delta a$ offset [m]")
    ax.set_ylabel(r"$a\,\delta i_x$ offset [m]")
    ax.set_title("Drift rate vs element offsets\n(1 km E/I-safe, i = 98°)",
                 fontsize=9)
    common.save(fig, "fig4_drift_contour")


if __name__ == "__main__":
    main()
