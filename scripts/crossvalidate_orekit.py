"""Independent-oracle cross-validation against Orekit (paper Table 2).

Propagates the same chief/deputy initial conditions in (a) this package's
numerical truth (DOP853, J2-J4 zonals) and (b) an Orekit numerical propagator
with a degree-4/order-0 gravity field, then reports absolute and relative
position agreement over the campaign duration.

Requires the [validation] extra:  uv sync --all-extras
First run downloads orekit-data (~50 MB) into the repo root.

Run:  uv run python scripts/crossvalidate_orekit.py [--days 30]

Note: Orekit's EGM coefficients differ from this package's EGM96 constants at
the ~1e-9 relative level, which bounds achievable agreement; the paper's
acceptance criterion is <10 m absolute over 30 days.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--step", type=float, default=600.0)
    args = ap.parse_args()

    try:
        import orekit_jpype
    except ImportError:
        raise SystemExit("orekit-jpype not installed; run: uv sync --all-extras")

    orekit_jpype.initVM()
    from orekit_jpype.pyhelpers import setup_orekit_data

    data_zip = Path("orekit-data.zip")
    if not data_zip.exists():
        print("[orekit] downloading orekit-data ...")
        from orekit_jpype.pyhelpers import download_orekit_data_curdir

        download_orekit_data_curdir()
    setup_orekit_data(filenames=[str(data_zip)], from_pip_library=False)

    from java.io import File  # noqa: F401  (JVM warm-up)
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.propagation import SpacecraftState
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.forces.gravity.potential import GravityFieldFactory
    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.utils import Constants, PVCoordinates, IERSConventions  # noqa: F401
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

    from eilj2.brouwer import mean_to_osc
    from eilj2.constants import MU_EARTH
    from eilj2.elements import coe_deputy_from_roe
    from eilj2.geometry import geometry_roe
    from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

    # --- initial conditions -------------------------------------------------
    coe_c = np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.3, 0.2, 0.1])
    droe = geometry_roe("ei_safe", 1000.0, coe_c[0])
    coe_d = coe_deputy_from_roe(coe_c, droe)
    y0 = state_from_coe(mean_to_osc(coe_c), mean_to_osc(coe_d))

    # --- our truth -----------------------------------------------------------
    times = np.arange(0.0, args.days * 86400.0 + args.step, args.step)
    tw = TwoSatTruth(TruthConfig(n_zonal=4, method="DOP853"))
    print("[orekit] propagating in eilj2 truth ...")
    Y = tw.propagate(y0, times)

    # --- Orekit --------------------------------------------------------------
    frame = FramesFactory.getGCRF()  # zonal-only field: any inertial frame
    utc = TimeScalesFactory.getUTC()
    t0 = AbsoluteDate(2026, 1, 1, 0, 0, 0.0, utc)
    gravity = GravityFieldFactory.getNormalizedProvider(4, 0)

    def orekit_propagate(r0: np.ndarray, v0: np.ndarray) -> np.ndarray:
        pv = PVCoordinates(Vector3D(*[float(v) for v in r0]),
                           Vector3D(*[float(v) for v in v0]))
        orbit = CartesianOrbit(pv, frame, t0, float(MU_EARTH))
        integ = DormandPrince853Integrator(1e-6, 300.0, 1e-8, 1e-11)
        prop = NumericalPropagator(integ)
        prop.setOrbitType(OrbitType.CARTESIAN)
        prop.addForceModel(HolmesFeatherstoneAttractionModel(
            frame, gravity))
        prop.setInitialState(SpacecraftState(orbit))
        out = np.empty((len(times), 3))
        for k, t in enumerate(times):
            st = prop.propagate(t0.shiftedBy(float(t)))
            p = st.getPVCoordinates(frame).getPosition()
            out[k] = [p.getX(), p.getY(), p.getZ()]
        return out

    print("[orekit] propagating chief in Orekit ...")
    rc_ok = orekit_propagate(y0[0:3], y0[3:6])
    print("[orekit] propagating deputy in Orekit ...")
    rd_ok = orekit_propagate(y0[6:9], y0[9:12])

    dc = np.linalg.norm(Y[:, 0:3] - rc_ok, axis=1)
    dd = np.linalg.norm(Y[:, 6:9] - rd_ok, axis=1)
    rel_us = Y[:, 6:9] - Y[:, 0:3]
    rel_ok = rd_ok - rc_ok
    dr = np.linalg.norm(rel_us - rel_ok, axis=1)

    print("\n[orekit] agreement over", args.days, "days (Table 2):")
    print(f"  chief   abs pos: max {dc.max():8.3f} m   rms {np.sqrt((dc**2).mean()):8.3f} m")
    print(f"  deputy  abs pos: max {dd.max():8.3f} m   rms {np.sqrt((dd**2).mean()):8.3f} m")
    print(f"  relative   pos:  max {dr.max():8.4f} m   rms {np.sqrt((dr**2).mean()):8.4f} m")


if __name__ == "__main__":
    main()
