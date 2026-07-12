"""Gim-Alfriend geometric STM via numerically-linearized composition.

Gim & Alfriend (JGCD 2003) propagate the osculating LVLH relative state as

    Phi_GA(t1, t0) = A(t1) * Sigma(t1, t0) * B(t0)

where B maps osculating LVLH Cartesian state to mean-element differences
(through the Brouwer osculating->mean transformation), Sigma propagates the
mean differences under first-order secular J2, and A maps back to osculating
LVLH (mean->osculating). Gim & Alfriend derived A, Sigma, B analytically;
here each factor is the *numerical Jacobian of the exact nonlinear map* built
from this package's verified Brouwer transformation, element conversions, and
exact LVLH kinematics. This is mathematically the same first-order-J2
geometric construction (differences are O(J2^2) and finite-difference error),
valid for arbitrary chief eccentricity, and includes J2 short- and long-
period effects that the purely secular models omit.

State: osculating LVLH Cartesian [m, m/s] (frame = "lvlh"), directly
comparable to CW and Schweighart-Sedwick.
"""

from __future__ import annotations

import numpy as np

from ..brouwer import mean_to_osc, osc_to_mean
from ..elements import (
    coe2rv,
    coe_deputy_from_roe,
    propagate_mean,
    roe_from_coe,
    rv2coe,
)
from ..frames import eci_to_lvlh, lvlh_to_eci
from .base import RelativeMotionModel

# J2-only frame kinematics: the GA construction is a first-order J2 theory.
_N_ZONAL = 2


class GimAlfriendModel(RelativeMotionModel):
    frame = "lvlh"
    name = "gim_alfriend"

    def __init__(self, fd_pos: float = 1.0, fd_vel: float = 1e-3, fd_roe: float = 1e-6):
        # central finite-difference steps: LVLH position [m], velocity [m/s],
        # dimensionless ROE. Deliberately LARGE: the maps are near-linear over
        # tens of meters (truncation ~ (a*fd_roe)^2/a ~ 1e-5 m relative 1e-6),
        # while small steps drown in the ~1e-6 m roundoff floor of the
        # Kepler/Brouwer evaluations.
        self.fd_pos = fd_pos
        self.fd_vel = fd_vel
        self.fd_roe = fd_roe
        # caches: A is reused across consecutive steps (the sim engine passes
        # the same chief-mean array as segment end then segment start); Sigma
        # depends only on (a, e, i, dt), which are secular invariants.
        self._A_cache: dict[bytes, np.ndarray] = {}
        self._Sigma_cache: dict[tuple, np.ndarray] = {}

    # -- exact nonlinear maps ------------------------------------------------

    def _lvlh_to_droe(self, x_lvlh: np.ndarray, coe_c_mean: np.ndarray) -> np.ndarray:
        """Osculating LVLH state -> dimensionless mean qns ROE."""
        coe_c_osc = mean_to_osc(coe_c_mean)
        r_c, v_c = coe2rv(coe_c_osc)
        r_d, v_d = lvlh_to_eci(r_c, v_c, x_lvlh, n_zonal=_N_ZONAL)
        coe_d_mean = osc_to_mean(rv2coe(r_d, v_d))
        return roe_from_coe(coe_c_mean, coe_d_mean)

    def _droe_to_lvlh(self, droe: np.ndarray, coe_c_mean: np.ndarray) -> np.ndarray:
        """Dimensionless mean qns ROE -> osculating LVLH state."""
        coe_d_mean = coe_deputy_from_roe(coe_c_mean, droe)
        coe_c_osc = mean_to_osc(coe_c_mean)
        coe_d_osc = mean_to_osc(coe_d_mean)
        r_c, v_c = coe2rv(coe_c_osc)
        r_d, v_d = coe2rv(coe_d_osc)
        return eci_to_lvlh(r_c, v_c, r_d, v_d, n_zonal=_N_ZONAL)

    @staticmethod
    def _propagate_droe(droe: np.ndarray, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        """Exact secular propagation of the mean ROE over dt."""
        coe_d_mean = coe_deputy_from_roe(coe_c_mean, droe)
        coe_c_1 = propagate_mean(coe_c_mean, dt)
        coe_d_1 = propagate_mean(coe_d_mean, dt)
        return roe_from_coe(coe_c_1, coe_d_1)

    # -- numerically-linearized factors ---------------------------------------

    def _B(self, coe_c_mean: np.ndarray) -> np.ndarray:
        """d(droe)/d(x_lvlh) at the chief (x_lvlh = 0), 6x6."""
        J = np.empty((6, 6))
        steps = np.array([self.fd_pos] * 3 + [self.fd_vel] * 3)
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = steps[k]
            fp = self._lvlh_to_droe(dx, coe_c_mean)
            fm = self._lvlh_to_droe(-dx, coe_c_mean)
            J[:, k] = (fp - fm) / (2.0 * steps[k])
        return J

    def _A(self, coe_c_mean: np.ndarray) -> np.ndarray:
        """d(x_lvlh)/d(droe) at droe = 0, 6x6."""
        J = np.empty((6, 6))
        for k in range(6):
            dd = np.zeros(6)
            dd[k] = self.fd_roe
            fp = self._droe_to_lvlh(dd, coe_c_mean)
            fm = self._droe_to_lvlh(-dd, coe_c_mean)
            J[:, k] = (fp - fm) / (2.0 * self.fd_roe)
        return J

    def _Sigma(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        """d(droe(t1))/d(droe(t0)) at droe = 0, 6x6."""
        J = np.empty((6, 6))
        for k in range(6):
            dd = np.zeros(6)
            dd[k] = self.fd_roe
            fp = self._propagate_droe(dd, coe_c_mean, dt)
            fm = self._propagate_droe(-dd, coe_c_mean, dt)
            J[:, k] = (fp - fm) / (2.0 * self.fd_roe)
        return J

    # -- cached accessors ------------------------------------------------------

    def _A_cached(self, coe_c_mean: np.ndarray) -> np.ndarray:
        key = np.asarray(coe_c_mean, dtype=float).tobytes()
        A = self._A_cache.get(key)
        if A is None:
            A = self._A(coe_c_mean)
            if len(self._A_cache) > 16:
                self._A_cache.clear()
            self._A_cache[key] = A
        return A

    def _Sigma_cached(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        key = (round(float(coe_c_mean[0]), 6), round(float(coe_c_mean[1]), 12),
               round(float(coe_c_mean[2]), 12), round(float(dt), 9))
        S = self._Sigma_cache.get(key)
        if S is None:
            S = self._Sigma(coe_c_mean, dt)
            if len(self._Sigma_cache) > 64:
                self._Sigma_cache.clear()
            self._Sigma_cache[key] = S
        return S

    # -- interface -------------------------------------------------------------

    def stm(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        # B(t0) = A(t0)^-1 exactly: A and B are Jacobians of mutually inverse
        # maps evaluated at corresponding points (droe = 0 <-> x_lvlh = 0), so
        # only A is ever computed; _B below is retained for validation tests.
        coe_c_1 = propagate_mean(coe_c_mean, dt)
        A0 = self._A_cached(coe_c_mean)
        A1 = self._A_cached(coe_c_1)
        Sig = self._Sigma_cached(coe_c_mean, dt)
        return A1 @ Sig @ np.linalg.inv(A0)
