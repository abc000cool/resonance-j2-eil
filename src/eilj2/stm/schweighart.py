"""Schweighart-Sedwick J2-modified linear (LTI) relative dynamics.

Schweighart & Sedwick, "High-Fidelity Linearized J2 Model for Satellite
Formation Flight", JGCD 25(6), 2002 — equations transcription-verified
against Schweighart's MIT S.M. thesis (2001, Eqs. 3.22-3.56) and the
independent reproductions in Bevilacqua, Hall & Romano (CeMDA 2010, Eqs. 1-2)
and Riano-Rios et al. (Appl. Sci. 2021, Eqs. 8-10).

With the reference (chief) circular orbit radius a and inclination i:

    s = (3 J2 Re^2 / (8 a^2)) (1 + 3 cos 2i)
    c = sqrt(1 + s)                      (frame rate omega = n c)
    k = n c + (3 n J2 Re^2 / (2 a^2)) cos^2 i

The two-satellite relative equations are homogeneous (the J2 forcing on the
absolute motion cancels between close satellites):

    x'' - 2 n c y' - (5 c^2 - 2) n^2 x = u_x
    y'' + 2 n c x'                     = u_y
    z'' + q^2 z                        = 2 l q cos(q t + phi) + u_z

For the LTI reduction used in LQR design and STM propagation the cross-track
frequency is q -> k (the equal-inclination limit of the corrected Solution-3
frequency; exact for delta-i = 0 and accurate to O(delta-i) for the
few-arcsecond relative inclinations flown here), and the l-forcing term —
proportional to the differential nodal-precession rate — is omitted from the
plant. That omitted cross-track drift is precisely the fidelity gap between
S-S and the GA/KGD models that the paper's STM ablation quantifies.

State: LVLH Cartesian [x, y, z, xdot, ydot, zdot] (frame = "lvlh").
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..constants import J2, R_EARTH
from ..elements import mean_motion
from .base import RelativeMotionModel


def ss_coefficients(a: float, i: float) -> dict[str, float]:
    n = mean_motion(a)
    s = (3.0 * J2 * R_EARTH**2 / (8.0 * a * a)) * (1.0 + 3.0 * np.cos(2.0 * i))
    c = np.sqrt(1.0 + s)
    k = n * c + (3.0 * n * J2 * R_EARTH**2 / (2.0 * a * a)) * np.cos(i) ** 2
    return {"n": n, "s": s, "c": c, "k": k}


def ss_system_matrix(coe_c_mean: np.ndarray) -> np.ndarray:
    """Continuous-time LTI A (Bevilacqua et al. 2010, Eqs. 1-2, with q -> k)."""
    a, i = coe_c_mean[0], coe_c_mean[2]
    co = ss_coefficients(a, i)
    n, c, k = co["n"], co["c"], co["k"]
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0] = (5.0 * c * c - 2.0) * n * n
    A[3, 4] = 2.0 * n * c
    A[4, 3] = -2.0 * n * c
    A[5, 2] = -k * k
    return A


def ss_system_matrices(coe_c_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)
    return ss_system_matrix(coe_c_mean), B


class SchweighartSedwickModel(RelativeMotionModel):
    frame = "lvlh"
    name = "ss"

    def __init__(self):
        self._stm_cache: dict[tuple, np.ndarray] = {}

    def stm(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        # LTI: Phi depends only on (a, i, dt) — cache the matrix exponential
        key = (round(float(coe_c_mean[0]), 6), round(float(coe_c_mean[2]), 12),
               round(float(dt), 9))
        Phi = self._stm_cache.get(key)
        if Phi is None:
            Phi = expm(ss_system_matrix(coe_c_mean) * dt)
            if len(self._stm_cache) > 64:
                self._stm_cache.clear()
            self._stm_cache[key] = Phi
        return Phi
