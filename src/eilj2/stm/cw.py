"""Clohessy-Wiltshire (Hill) model: closed-form STM and LTI system matrices.

State: x = [x, y, z, xdot, ydot, zdot], LVLH (x radial, y along-track,
z cross-track), circular chief, point-mass gravity. Clohessy & Wiltshire
(JAS 1960); matrix form as in the proposal Sec. 1.1.
"""

from __future__ import annotations

import numpy as np

from ..elements import mean_motion
from .base import RelativeMotionModel


def cw_system_matrix(n: float) -> np.ndarray:
    """Continuous-time A for the CW equations (LVLH state, control = accel)."""
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3, 0] = 3.0 * n * n
    A[3, 4] = 2.0 * n
    A[4, 3] = -2.0 * n
    A[5, 2] = -n * n
    return A


def cw_input_matrix() -> np.ndarray:
    """B mapping LVLH acceleration to state derivative."""
    B = np.zeros((6, 3))
    B[3:, :] = np.eye(3)
    return B


def cw_stm(n: float, t: float) -> np.ndarray:
    """Closed-form 6x6 CW state transition matrix Phi(t)."""
    s, c = np.sin(n * t), np.cos(n * t)
    nt = n * t
    return np.array([
        [4.0 - 3.0 * c,       0.0, 0.0,  s / n,               2.0 * (1.0 - c) / n,       0.0],
        [6.0 * (s - nt),      1.0, 0.0, -2.0 * (1.0 - c) / n, (4.0 * s - 3.0 * nt) / n,  0.0],
        [0.0,                 0.0, c,    0.0,                 0.0,                       s / n],
        [3.0 * n * s,         0.0, 0.0,  c,                   2.0 * s,                   0.0],
        [6.0 * n * (c - 1.0), 0.0, 0.0, -2.0 * s,             4.0 * c - 3.0,             0.0],
        [0.0,                 0.0, -n * s, 0.0,               0.0,                       c],
    ])


def no_drift_ydot(x0: float, n: float) -> float:
    """CW energy-matching condition: ydot0 = -2 n x0 removes secular drift."""
    return -2.0 * n * x0


class CWModel(RelativeMotionModel):
    """CW as an STM provider (LVLH-native)."""

    frame = "lvlh"
    name = "cw"

    def stm(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        n = mean_motion(coe_c_mean[0])
        return cw_stm(n, dt)
