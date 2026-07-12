"""Two-satellite numerical truth propagator on the J2-J4 zonal field.

The truth state is the 12-vector y = [r_c, v_c, r_d, v_d] in ECI. Control
enters as (i) a constant LVLH acceleration on the deputy over an integration
segment (zero-order hold), and (ii) instantaneous LVLH delta-v impulses on
the deputy. Integration uses scipy DOP853 with tight tolerances; the primary
sweep isolates J2 physics by setting n_zonal per config (J2-only truth for
model validation, J2-J4 for the fidelity campaign — drag/SRP/third-body are
deliberately excluded, per the study design).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .elements import coe2rv
from .frames import eci_to_lvlh, lvlh_dcm
from .gravity import accel_zonal


@dataclass
class TruthConfig:
    n_zonal: int = 4
    method: str = "DOP853"   # "DOP853" (adaptive) or "RK4" (fast fixed-step)
    rtol: float = 1e-11
    atol_pos: float = 1e-7   # m
    atol_vel: float = 1e-10  # m/s
    rk4_substep: float = 10.0  # [s] fixed substep for the RK4 path


class TwoSatTruth:
    """Chief + deputy point-mass propagation with zonal harmonics."""

    def __init__(self, config: TruthConfig | None = None):
        self.cfg = config or TruthConfig()
        atol_one = np.array([self.cfg.atol_pos] * 3 + [self.cfg.atol_vel] * 3)
        self._atol = np.concatenate((atol_one, atol_one))

    # -- dynamics -----------------------------------------------------------

    def rhs(self, t: float, y: np.ndarray, u_lvlh: np.ndarray | None = None) -> np.ndarray:
        r_c, v_c = y[0:3], y[3:6]
        r_d, v_d = y[6:9], y[9:12]
        a_c = accel_zonal(r_c, n_max=self.cfg.n_zonal)
        a_d = accel_zonal(r_d, n_max=self.cfg.n_zonal)
        if u_lvlh is not None:
            R = lvlh_dcm(r_c, v_c)
            a_d = a_d + R.T @ u_lvlh
        return np.concatenate((v_c, a_c, v_d, a_d))

    # -- propagation --------------------------------------------------------

    def step(self, y0: np.ndarray, dt: float, u_lvlh: np.ndarray | None = None) -> np.ndarray:
        """Propagate one control segment of length dt (ZOH LVLH accel on deputy).

        The RK4 path exists because closed-loop campaigns call step() tens of
        thousands of times per trial and solve_ivp's per-call overhead
        dominates; a 10 s fixed substep keeps local truncation orders of
        magnitude below the meter-level effects under study (verified against
        DOP853 in tests/test_truth_rk4.py).
        """
        if self.cfg.method.upper() == "RK4":
            return self._rk4(y0, dt, u_lvlh)
        sol = solve_ivp(
            self.rhs, (0.0, dt), y0,
            method=self.cfg.method, rtol=self.cfg.rtol, atol=self._atol,
            args=(u_lvlh,),
        )
        if not sol.success:
            raise RuntimeError(f"truth propagation failed: {sol.message}")
        return sol.y[:, -1]

    def _rk4(self, y: np.ndarray, dt: float, u_lvlh: np.ndarray | None) -> np.ndarray:
        n_sub = max(1, int(np.ceil(dt / self.cfg.rk4_substep)))
        h = dt / n_sub
        for _ in range(n_sub):
            k1 = self.rhs(0.0, y, u_lvlh)
            k2 = self.rhs(0.0, y + 0.5 * h * k1, u_lvlh)
            k3 = self.rhs(0.0, y + 0.5 * h * k2, u_lvlh)
            k4 = self.rhs(0.0, y + h * k3, u_lvlh)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y

    def propagate(self, y0: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Ballistic (uncontrolled) propagation sampled at `times` (shape (N, 12))."""
        sol = solve_ivp(
            self.rhs, (float(times[0]), float(times[-1])), y0,
            t_eval=np.asarray(times, dtype=float),
            method=self.cfg.method, rtol=self.cfg.rtol, atol=self._atol,
        )
        if not sol.success:
            raise RuntimeError(f"truth propagation failed: {sol.message}")
        return sol.y.T

    # -- control and output -------------------------------------------------

    def apply_impulse_lvlh(self, y: np.ndarray, dv_lvlh: np.ndarray) -> np.ndarray:
        """Instantaneous deputy delta-v given in the chief LVLH frame."""
        y = y.copy()
        R = lvlh_dcm(y[0:3], y[3:6])
        y[9:12] = y[9:12] + R.T @ np.asarray(dv_lvlh, dtype=float)
        return y

    def relative_lvlh(self, y: np.ndarray) -> np.ndarray:
        """LVLH relative state [rho; rho_dot] of deputy w.r.t. chief."""
        return eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=self.cfg.n_zonal)


def state_from_coe(coe_c: np.ndarray, coe_d: np.ndarray) -> np.ndarray:
    """Truth state vector from OSCULATING classical elements of chief and deputy."""
    r_c, v_c = coe2rv(coe_c)
    r_d, v_d = coe2rv(coe_d)
    return np.concatenate((r_c, v_c, r_d, v_d))
