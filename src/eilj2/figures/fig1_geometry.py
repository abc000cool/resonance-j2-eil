"""Figure 1: LVLH frame and formation-geometry families (schematic) plus the
Phase A hello-world 500-m PCO propagated 5 orbits under CW."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..elements import mean_motion, period
from ..geometry import geometry_roe
from ..roe_map import lvlh_from_roe
from ..stm.cw import cw_stm
from . import common


def main() -> None:
    common.apply_style()
    a = 7078137.0
    n = mean_motion(a)
    T = period(a)
    coe = np.array([a, 0.001, np.deg2rad(98.0), 0.0, 0.0, 0.0])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    families = ["pco", "gero", "ei_safe"]
    for ax, fam in zip(axes, families):
        s = geometry_roe(fam, 500.0, a) * a
        # trace one relative orbit through the ROE->LVLH map
        us = np.linspace(0.0, 2 * np.pi, 400)
        xs, ys, zs = [], [], []
        for u in us:
            c = coe.copy()
            c[5] = u
            x = lvlh_from_roe(s, c)
            xs.append(x[0]); ys.append(x[1]); zs.append(x[2])
        ax.plot(ys, zs, label="y–z (projected)")
        ax.plot(ys, xs, "--", label="y–x (in-plane)")
        ax.set_title(fam.replace("_", "-"))
        ax.set_xlabel("along-track y [m]")
        ax.set_aspect("equal")
    axes[0].set_ylabel("cross-track z / radial x [m]")
    axes[0].legend(fontsize=7, loc="upper right")

    # CW sanity inset: 500-m PCO for 5 orbits stays closed
    s0 = geometry_roe("pco", 500.0, a) * a
    x = lvlh_from_roe(s0, coe)
    pts = []
    dt = 60.0
    Phi = cw_stm(n, dt)
    for _ in range(int(5 * T / dt)):
        x = Phi @ x
        pts.append(x[:3].copy())
    pts = np.array(pts)
    drift = np.linalg.norm(pts[-1] - pts[0])
    fig.suptitle(f"Formation geometries in LVLH (500 m; CW 5-orbit closure "
                 f"drift {drift:.1f} m)", y=1.04, fontsize=9)
    common.save(fig, "fig1_geometry")


if __name__ == "__main__":
    main()
