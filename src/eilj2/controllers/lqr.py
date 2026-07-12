"""Steady-state continuous LQR on an LTI relative-motion plant (CW or S-S).

u = -K (x_hat - x_ref) with K from the continuous algebraic Riccati equation
(proposal Sec. 1.7). The tightness-vs-delta-V Pareto is traced by sweeping
the scalar control weight r_weight; optional saturation models a
low-thrust-limited actuator and its events are counted for the
certainty-equivalence-breakdown analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


class LQRController:
    kind = "continuous"

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        q_pos: float = 1.0,
        q_vel: float | None = None,
        r_weight: float = 1e10,
        u_max: float | None = None,
        n_ref: float | None = None,
    ):
        """q_vel defaults to q_pos / n^2 (natural CW scaling) when n_ref given."""
        if q_vel is None:
            if n_ref is None:
                raise ValueError("provide q_vel or n_ref")
            q_vel = q_pos / n_ref**2
        Q = np.diag([q_pos] * 3 + [q_vel] * 3)
        R = r_weight * np.eye(3)
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)
        self.u_max = u_max
        self.saturation_count = 0

    def accel(self, t: float, x_hat_lvlh: np.ndarray, x_ref_lvlh: np.ndarray) -> np.ndarray:
        u = -self.K @ (x_hat_lvlh - x_ref_lvlh)
        if self.u_max is not None:
            norm = np.linalg.norm(u)
            if norm > self.u_max:
                self.saturation_count += 1
                u = u * (self.u_max / norm)
        return u


class CovarianceAwareLQR(LQRController):
    """Nav-aware LQR variant for the certainty-equivalence stress test
    (proposal Phase F): the control weight R is inflated by the current
    filter position covariance, R_eff = r_weight (1 + beta tr(P_pos)/p_ref^2),
    so the controller backs off when the navigation solution is poor instead
    of amplifying estimator noise into thrust. Gains are precomputed on a
    log-spaced inflation bank (CARE is too expensive per step).
    """

    def __init__(self, A, B, q_pos: float = 1.0, q_vel: float | None = None,
                 r_weight: float = 1e10, u_max: float | None = None,
                 n_ref: float | None = None, beta: float = 1.0,
                 p_ref: float = 1.0, n_bank: int = 24):
        super().__init__(A, B, q_pos=q_pos, q_vel=q_vel, r_weight=r_weight,
                         u_max=u_max, n_ref=n_ref)
        if q_vel is None:
            q_vel = q_pos / n_ref**2
        Q = np.diag([q_pos] * 3 + [q_vel] * 3)
        self.beta = beta
        self.p_ref = p_ref
        self._mults = np.logspace(0.0, 3.0, n_bank)
        self._bank = []
        for m in self._mults:
            R = r_weight * m * np.eye(3)
            P = solve_continuous_are(A, B, Q, R)
            self._bank.append(np.linalg.solve(R, B.T @ P))
        self.K = self._bank[0]

    def set_covariance(self, P_lvlh: np.ndarray) -> None:
        infl = 1.0 + self.beta * float(np.trace(P_lvlh[:3, :3])) / self.p_ref**2
        idx = int(np.argmin(np.abs(self._mults - min(infl, self._mults[-1]))))
        self.K = self._bank[idx]
