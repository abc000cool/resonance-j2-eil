"""Unscented Kalman filter (Merwe scaled sigma points).

Drop-in ablation for the EKF (proposal Sec. 1.8): prediction uses the same
linear STM (exact for a linear model, so the UKF and EKF predictions agree);
the difference is the measurement update, where sigma points propagate the
measurement nonlinearity (range, bearings) without Jacobian truncation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class UnscentedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        n = len(self.x)
        self.n = n
        lam = alpha**2 * (n + kappa) - n
        self.lam = lam
        self.wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.wc = self.wm.copy()
        self.wm[0] = lam / (n + lam)
        self.wc[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)

    def _sigma_points(self) -> np.ndarray:
        n = self.n
        try:
            L = np.linalg.cholesky((n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            # regularize: symmetric part + jitter
            P = 0.5 * (self.P + self.P.T) + 1e-12 * np.eye(n)
            L = np.linalg.cholesky((n + self.lam) * P)
        pts = np.empty((2 * n + 1, n))
        pts[0] = self.x
        for k in range(n):
            pts[1 + k] = self.x + L[:, k]
            pts[1 + n + k] = self.x - L[:, k]
        return pts

    def predict(self, Phi: np.ndarray, Q: np.ndarray) -> None:
        # Linear dynamics: sigma-point propagation reduces to the linear map.
        self.x = Phi @ self.x
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray, h_fn: Callable[[np.ndarray], np.ndarray],
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """h_fn maps a NATIVE-frame state vector to a predicted measurement."""
        pts = self._sigma_points()
        Z = np.array([h_fn(p) for p in pts])
        z_pred = self.wm @ Z
        dZ = Z - z_pred
        dX = pts - self.x
        S = dZ.T @ (self.wc[:, None] * dZ) + R
        Pxz = dX.T @ (self.wc[:, None] * dZ)
        K = np.linalg.solve(S.T, Pxz.T).T
        innov = z - z_pred
        self.x = self.x + K @ innov
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return innov, S
