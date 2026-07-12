"""Koenig-Guffanti-D'Amico quasi-nonsingular ROE J2 secular STM.

Full arbitrary-eccentricity first-order J2 secular STM of Koenig, Guffanti &
D'Amico, "New State Transition Matrices for Spacecraft Relative Motion in
Perturbed Orbits", JGCD Vol. 40 No. 7, 2017, Appendix A3 Eq. (A6) (DOI
10.2514/1.G002409) — transcription verified visually against the author PDF
and the identical AIAA 2016-5635 precursor, and cross-validated numerically
against a numerically-linearized exact secular propagation
(tests/test_kgd.py).

Substitutions (paper Eqs. 13-16, A1-A2), all evaluated at the CHIEF's mean
elements at segment start:

    eta = sqrt(1 - e^2)
    kappa = (3/4) J2 Re^2 sqrt(mu) / (a^{7/2} eta^4)
    E = 1 + eta,  F = 4 + 3 eta,  G = 1/eta^2
    P = 3 cos^2 i - 1,  Q = 5 cos^2 i - 1,  S = sin 2i,  T = sin^2 i
    omega_dot = kappa Q  (relative-perigee rotation rate)
    e_xi = e cos(omega_i),  e_yi = e sin(omega_i)     (initial perigee)
    e_xf = e cos(omega_f),  e_yf = e sin(omega_f),  omega_f = omega_i + omega_dot tau

State: scaled qns ROE a_c * [da, dlambda, dex, dey, dix, diy] in meters
(uniform scaling by a_c leaves Phi unchanged).
"""

from __future__ import annotations

import numpy as np

from ..constants import J2, MU_EARTH, R_EARTH
from ..elements import mean_motion
from ..roe_map import gve_control_matrix, roe_to_lvlh_matrix
from .base import RelativeMotionModel


def kgd_coefficients(a: float, e: float, i: float) -> dict[str, float]:
    n = mean_motion(a)
    eta = np.sqrt(1.0 - e * e)
    kappa = 0.75 * J2 * R_EARTH**2 * np.sqrt(MU_EARTH) / (a**3.5 * eta**4)
    ci = np.cos(i)
    return {
        "n": n,
        "eta": eta,
        "kappa": kappa,
        "E": 1.0 + eta,
        "F": 4.0 + 3.0 * eta,
        "G": 1.0 / eta**2,
        "P": 3.0 * ci * ci - 1.0,
        "Q": 5.0 * ci * ci - 1.0,
        "S": np.sin(2.0 * i),
        "T": np.sin(i) ** 2,
    }


def kgd_stm(coe_c_mean: np.ndarray, tau: float) -> np.ndarray:
    """JGCD 2017 Eq. (A6): J2 secular qns-ROE STM, arbitrary eccentricity."""
    a, e, i = coe_c_mean[0], coe_c_mean[1], coe_c_mean[2]
    w_i = coe_c_mean[4]
    k = kgd_coefficients(a, e, i)
    kp, G, P, Q, S, T = k["kappa"], k["G"], k["P"], k["Q"], k["S"], k["T"]
    wdot = kp * Q
    w_f = w_i + wdot * tau
    exi, eyi = e * np.cos(w_i), e * np.sin(w_i)
    exf, eyf = e * np.cos(w_f), e * np.sin(w_f)
    cw, sw = np.cos(wdot * tau), np.sin(wdot * tau)

    Phi = np.eye(6)
    # row dlambda
    Phi[1, 0] = -(1.5 * k["n"] + 3.5 * kp * k["E"] * P) * tau
    Phi[1, 2] = kp * exi * k["F"] * G * P * tau
    Phi[1, 3] = kp * eyi * k["F"] * G * P * tau
    Phi[1, 4] = -kp * k["F"] * S * tau
    # row dex
    Phi[2, 0] = 3.5 * kp * eyf * Q * tau
    Phi[2, 2] = cw - 4.0 * kp * exi * eyf * G * Q * tau
    Phi[2, 3] = -sw - 4.0 * kp * eyi * eyf * G * Q * tau
    Phi[2, 4] = 5.0 * kp * eyf * S * tau
    # row dey
    Phi[3, 0] = -3.5 * kp * exf * Q * tau
    Phi[3, 2] = sw + 4.0 * kp * exi * exf * G * Q * tau
    Phi[3, 3] = cw + 4.0 * kp * eyi * exf * G * Q * tau
    Phi[3, 4] = -5.0 * kp * exf * S * tau
    # row diy
    Phi[5, 0] = 3.5 * kp * S * tau
    Phi[5, 2] = -4.0 * kp * exi * G * S * tau
    Phi[5, 3] = -4.0 * kp * eyi * G * S * tau
    Phi[5, 4] = 2.0 * kp * T * tau
    return Phi


def arg_lat_rate(coe_c_mean: np.ndarray) -> float:
    """J2-perturbed mean argument-of-latitude rate n + kappa (eta P + Q)."""
    k = kgd_coefficients(coe_c_mean[0], coe_c_mean[1], coe_c_mean[2])
    return k["n"] + k["kappa"] * (k["eta"] * k["P"] + k["Q"])


class KGDModel(RelativeMotionModel):
    frame = "roe"
    name = "kgd"

    def stm(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        return kgd_stm(coe_c_mean, dt)

    def to_lvlh_jacobian(self, coe_c_mean: np.ndarray) -> np.ndarray:
        return roe_to_lvlh_matrix(coe_c_mean)

    def control_input(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        # impulse at segment start (GVE jump), then secular propagation
        return self.stm(coe_c_mean, dt) @ gve_control_matrix(coe_c_mean)
