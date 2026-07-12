"""Calibrate the filter process-noise PSD against STM-vs-truth residuals
(proposal Sec. 1.8): propagate an uncontrolled formation in the numerical
truth, collect one-step prediction residuals for each analytic model, and fit
the white-acceleration PSD. Paste the printed values into the campaign YAMLs.

Run:  uv run python scripts/calibrate_q.py [--days 3]
"""

from __future__ import annotations

import argparse

import numpy as np

from eilj2.brouwer import mean_to_osc, osc_to_mean
from eilj2.elements import coe_deputy_from_roe, roe_from_coe, rv2coe
from eilj2.filters.noise import calibrate_q_accel
from eilj2.frames import eci_to_lvlh
from eilj2.geometry import geometry_roe
from eilj2.stm import get_model
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

MODELS = ["cw", "ss", "gim_alfriend", "kgd"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--i-deg", type=float, default=98.0)
    args = ap.parse_args()

    coe_c = np.array([7078137.0, 0.001, np.deg2rad(args.i_deg), 0.3, 0.2, 0.1])
    a = coe_c[0]
    droe = geometry_roe("ei_safe", 1000.0, a)
    coe_d = coe_deputy_from_roe(coe_c, droe)
    tw = TwoSatTruth(TruthConfig(n_zonal=4, method="RK4"))
    y = state_from_coe(mean_to_osc(coe_c), mean_to_osc(coe_d))

    models = {}
    for name in MODELS:
        try:
            models[name] = get_model(name)
        except Exception as exc:
            print(f"[calibrate] {name} unavailable ({exc}); skipped")

    n_steps = int(args.days * 86400.0 / args.dt)
    resid: dict[str, list] = {name: [] for name in models}
    coe_mean = coe_c.copy()
    x_prev, s_prev = None, None
    for k in range(n_steps + 1):
        x_lvlh = eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=4)
        cc = osc_to_mean(rv2coe(y[0:3], y[3:6]))
        cd = osc_to_mean(rv2coe(y[6:9], y[9:12]))
        s = roe_from_coe(cc, cd) * cc[0]
        if k > 0:
            for name, m in models.items():
                Phi = m.stm(coe_prev, args.dt)
                if m.frame == "roe":
                    pred = Phi @ s_prev
                    J = m.to_lvlh_jacobian(cc)
                    r = J @ (s - pred)
                else:
                    pred = Phi @ x_prev
                    r = x_lvlh - pred
                resid[name].append(r)
        x_prev, s_prev, coe_prev = x_lvlh, s, cc
        y = tw.step(y, args.dt)

    print(f"\n[calibrate] white-accel PSD fits over {args.days} days, dt={args.dt}s")
    print(f"{'model':>14s}   q_accel [(m/s^2)^2 s]   1-step pos resid RMS [m]")
    for name in models:
        R = np.array(resid[name])
        q = calibrate_q_accel(R, args.dt)
        pos_rms = float(np.sqrt(np.mean(np.linalg.norm(R[:, :3], axis=1) ** 2)))
        print(f"{name:>14s}   {q:20.3e}   {pos_rms:12.4e}")


if __name__ == "__main__":
    main()
