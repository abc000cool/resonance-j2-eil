"""Figure 5: perfect-state delta-V/yr vs RMS formation-keeping error Pareto."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from . import common

DIR = common.REPO / "data" / "perfect_state"


def main() -> None:
    common.apply_style()
    f = DIR / "pareto_numerical.parquet"
    if not f.exists():
        f = DIR / "pareto_stm.parquet"
    common.need(f, "uv run python scripts/run_perfect_state.py")
    df = pd.read_parquet(f)
    df = df[~df.diverged.astype(bool)]

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for ctrl, g in df.groupby("controller"):
        g = g.sort_values("rms_pos_err")
        ax.loglog(g.rms_pos_err, g.dv_per_year, marker="o", ms=3,
                  label=common.CTRL_LABELS.get(ctrl, ctrl))
    ax.set_xlabel("RMS formation-keeping error [m]")
    ax.set_ylabel(r"$\Delta V$ [m/s/yr]")
    ax.set_title("Perfect-navigation Pareto (1 km E/I-safe, i = 98°)",
                 fontsize=9)
    ax.legend(fontsize=7)
    common.save(fig, "fig5_perfect_pareto")


if __name__ == "__main__":
    main()
