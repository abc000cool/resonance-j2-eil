"""Phase B: analytic relative-motion models vs numerical truth (paper Fig. 2).

For each chief inclination, propagate an uncontrolled formation 30 days in
the J2-J4 numerical truth, then predict the same trajectory with each
analytic model (CW, S-S, GA, KGD) stepping along the analytically-propagated
mean chief. Records per-orbit RMS LVLH position error to Parquet.

Run:  uv run python scripts/run_validation.py [--days 30] [--dt 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eilj2.brouwer import mean_to_osc
from eilj2.elements import coe_deputy_from_roe, period, propagate_mean, roe_from_coe
from eilj2.frames import eci_to_lvlh
from eilj2.geometry import geometry_roe
from eilj2.roe_map import lvlh_from_roe
from eilj2.stm import get_model
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

MODELS = ["cw", "ss", "gim_alfriend", "kgd"]


def run_case(i_deg: float, family: str, size: float, days: float, dt: float,
             n_zonal: int) -> pd.DataFrame:
    coe_c = np.array([7078137.0, 0.001, np.deg2rad(i_deg), 0.3, 0.2, 0.1])
    a = coe_c[0]
    droe = geometry_roe(family, size, a)
    coe_d = coe_deputy_from_roe(coe_c, droe)
    tw = TwoSatTruth(TruthConfig(n_zonal=n_zonal, method="RK4"))

    T = period(a)
    n_steps = int(days * 86400.0 / dt)
    y = state_from_coe(mean_to_osc(coe_c), mean_to_osc(coe_d))
    x0_lvlh = eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=n_zonal)
    s0 = roe_from_coe(coe_c, coe_d) * a

    models = {}
    states = {}
    for name in MODELS:
        try:
            m = get_model(name)
        except (ImportError, ValueError, NotImplementedError) as exc:
            print(f"  [warn] model {name} unavailable ({exc}); skipped")
            continue
        models[name] = m
        states[name] = s0.copy() if m.frame == "roe" else x0_lvlh.copy()

    coe_mean = coe_c.copy()
    rows = []
    err_acc: dict[str, list] = {name: [] for name in models}
    for k in range(1, n_steps + 1):
        for name, m in models.items():
            states[name] = m.stm(coe_mean, dt) @ states[name]
        coe_mean = propagate_mean(coe_mean, dt)
        y = tw.step(y, dt)
        x_true = eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=n_zonal)
        for name, m in models.items():
            x_pred = (m.to_lvlh(states[name], coe_mean)
                      if m.frame == "roe" else states[name])
            err_acc[name].append(np.linalg.norm(x_pred[:3] - x_true[:3]))
        t = k * dt
        if t % T < dt:  # orbit boundary: flush per-orbit RMS
            orbit = int(t // T)
            for name in models:
                e = np.array(err_acc[name])
                rows.append(dict(i_deg=i_deg, family=family, size=size,
                                 model=name, orbit=orbit, days=t / 86400.0,
                                 rms_err=float(np.sqrt(np.mean(e**2))),
                                 max_err=float(e.max())))
                err_acc[name] = []
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--inclinations", type=float, nargs="+", default=[45.0, 70.0, 98.0])
    ap.add_argument("--family", default="ei_safe",
                    help="ei_safe exercises every ROE channel (a pure "
                         "leader-follower has almost no differential J2)")
    ap.add_argument("--size", type=float, default=1000.0)
    ap.add_argument("--zonal", type=int, default=4)
    ap.add_argument("--out", default="data/validation")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i_deg in args.inclinations:
        print(f"[validation] i = {i_deg} deg ...")
        frames.append(run_case(i_deg, args.family, args.size, args.days,
                               args.dt, args.zonal))
    df = pd.concat(frames, ignore_index=True)
    f = out / "model_vs_truth.parquet"
    df.to_parquet(f, index=False)
    print(f"[validation] wrote {f} ({len(df)} rows)")
    # console summary: final-orbit RMS per model
    last = df[df.orbit == df.orbit.max()]
    print(last.pivot_table(index="model", columns="i_deg", values="rms_err"))


if __name__ == "__main__":
    main()
