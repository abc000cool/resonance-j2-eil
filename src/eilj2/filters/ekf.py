"""Error-state extended Kalman filter, Joseph-form update.

The filter state is the relative state in the chosen STM's native frame
(LVLH Cartesian or scaled ROE); the STM provides the linear prediction, so
this is the error-state formulation of the proposal Sec. 1.8: the large
deterministic chief motion lives in the reference (chief mean elements), and
the filter sees only the small relative state.
"""

from __future__ import annotations

import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray):
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()

    def predict(self, Phi: np.ndarray, Q: np.ndarray) -> None:
        self.x = Phi @ self.x
        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z: np.ndarray, h_of_x: np.ndarray, H: np.ndarray,
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Measurement update. Returns (innovation, innovation covariance).

        h_of_x is the predicted measurement evaluated at the current state
        (the caller owns the measurement model and any frame composition).
        """
        innov = z - h_of_x
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T  # P H^T S^-1
        self.x = self.x + K @ innov
        IKH = np.eye(len(self.x)) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return innov, S
