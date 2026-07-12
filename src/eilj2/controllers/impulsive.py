"""Closed-form impulsive formation-keeping (Chernick & D'Amico).

Implements the near-circular, J2-perturbed closed-form maneuver schemes of
Chernick & D'Amico (JGCD 2018, DOI 10.2514/1.G002848; equations verified
against the identically-titled AIAA 2016-5659 precursor and Chernick's 2021
Stanford thesis):

- pseudo-state targeting: the desired ROE correction is
  Delta_dsbar = s_des - Phi(t_f, t_0) s_hat, i.e. the J2 drift over the
  control window is pre-compensated through the KGD STM;
- out-of-plane: one cross-track impulse at the argument of latitude aligned
  with the desired relative-inclination-vector change (2016 Eq. 23), with the
  J2-refined location from the transcendental Eq. 26 solved by Newton
  iteration from the Keplerian guess;
- in-plane: three tangential impulses at the J2-shifted optimal phases
  u_Tk = (Ubar + k pi - c u_f) / (1 - c), c = omega_dot / u_dot (2016
  Eq. 31), with magnitudes obtained by solving the exact linear system
  built from the same STM/GVE matrices the closed forms are derived from
  (equivalent to 2016 Eq. 27 / Tables 3-4, and numerically robust in the
  degenerate dominant-da / dominant-dlambda regimes).

All states are SCALED qns ROE [m]; delta-v components are RTN (= LVLH).
"""

from __future__ import annotations

import numpy as np

from ..elements import mean_arg_latitude
from ..roe_map import gve_control_matrix
from ..stm.kgd import arg_lat_rate, kgd_coefficients, kgd_stm

_TINY = 1e-12


class ChernickDAmicoController:
    kind = "impulsive"

    def __init__(self, deadband: float = 0.0, window_orbits: float = 1.0,
                 j2_refine: bool = True):
        self.deadband = float(deadband)
        self.window_orbits = float(window_orbits)
        self.j2_refine = j2_refine

    # ------------------------------------------------------------------

    def plan(self, t: float, s_hat: np.ndarray, s_des: np.ndarray,
             coe_c_mean: np.ndarray) -> list[tuple[float, np.ndarray]]:
        a, e, i = coe_c_mean[0], coe_c_mean[1], coe_c_mean[2]
        k = kgd_coefficients(a, e, i)
        n, kappa, T_sin2 = k["n"], k["kappa"], k["T"]
        udot = arg_lat_rate(coe_c_mean)
        wdot = kappa * k["Q"]
        c = wdot / udot

        u0 = mean_arg_latitude(coe_c_mean)
        # window sized so the in-plane triple (spanning ~2 pi in u) fits
        u_f = u0 + 2.0 * np.pi * (self.window_orbits + 0.75)
        tau_f = (u_f - u0) / udot

        Phi_f0 = kgd_stm(coe_c_mean, tau_f)
        ds = s_des - Phi_f0 @ s_hat  # pseudo-state correction [m]
        if np.linalg.norm(ds) < self.deadband:
            return []

        burns: list[tuple[float, np.ndarray]] = []

        # ---- out-of-plane: single cross-track impulse ---------------------
        dix, diy = ds[4], ds[5]
        din = np.hypot(dix, diy)
        if din > 1e-6:  # ignore sub-micrometer corrections
            uN = np.arctan2(diy, dix)
            m = int(np.ceil((u0 + 0.02 - uN) / np.pi))
            uN_abs = uN + m * np.pi
            if self.j2_refine:
                beta = 2.0 * kappa * T_sin2 / udot
                for _ in range(3):  # Newton on f(u) = dix (sin u + beta (u_f-u) cos u) - diy cos u
                    su, cu = np.sin(uN_abs), np.cos(uN_abs)
                    f = dix * (su + beta * (u_f - uN_abs) * cu) - diy * cu
                    fp = (dix * (cu - beta * cu - beta * (u_f - uN_abs) * su)
                          + diy * su)
                    if abs(fp) < _TINY:
                        break
                    uN_abs -= f / fp
            # magnitude from the exact 2x1 system (handles cos u -> 0)
            tau_b = (uN_abs - u0) / udot
            su, cu = np.sin(uN_abs), np.cos(uN_abs)
            E = np.array([cu / n,
                          (su + 2.0 * kappa * T_sin2 * (tau_f - tau_b) * cu) / n])
            dvN = float(np.linalg.lstsq(E[:, None], np.array([dix, diy]), rcond=None)[0][0])
            burns.append((t + tau_b, np.array([0.0, 0.0, dvN])))

        # ---- in-plane: three tangential impulses --------------------------
        da, dl, dex, dey = ds[0], ds[1], ds[2], ds[3]
        if max(abs(da), abs(dl), np.hypot(dex, dey)) > 1e-6:
            Ubar = np.arctan2(dey, dex) if np.hypot(dex, dey) > _TINY else u0
            # smallest k with u_T >= u0 + margin  (2016 Eq. 31)
            k_start = int(np.ceil(((u0 + 0.02) * (1.0 - c) + c * u_f - Ubar) / np.pi))
            u_T = np.array([(Ubar + (k_start + j) * np.pi - c * u_f) / (1.0 - c)
                            for j in range(3)])
            tau_b = (u_T - u0) / udot
            # exact effect of each tangential burn on [da, dl, dex, dey](t_f)
            A = np.zeros((4, 3))
            for j in range(3):
                coe_b = coe_c_mean.copy()
                coe_b[5] = coe_b[5] + (u_T[j] - u0)  # advance mean arg of latitude
                col = kgd_stm(coe_b, tau_f - tau_b[j]) @ gve_control_matrix(coe_b)[:, 1]
                A[:, j] = col[:4]
            dv_T, *_ = np.linalg.lstsq(A, np.array([da, dl, dex, dey]), rcond=None)
            for j in range(3):
                if abs(dv_T[j]) > 1e-9:
                    burns.append((t + tau_b[j], np.array([0.0, float(dv_T[j]), 0.0])))

        burns.sort(key=lambda b: b[0])
        return burns
