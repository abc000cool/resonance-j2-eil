"""Process-noise construction and calibration (proposal Sec. 1.8).

Q must absorb what the prediction STM does not model (higher zonals, the
truncation of the analytic theory, and — in the full campaign — the J3/J4
terms deliberately present in the truth but absent from the J2-only STMs).
The white-acceleration model gives the standard discrete Q; the calibration
helper fits the acceleration PSD to actual one-step STM-vs-truth residuals.
"""

from __future__ import annotations

import numpy as np


def white_accel_Q(q_accel: float, dt: float) -> np.ndarray:
    """Discrete Q for LVLH [pos; vel] driven by white acceleration.

    q_accel: acceleration PSD [(m/s^2)^2 * s] (i.e. sigma_a^2 * tau).
    """
    I3 = np.eye(3)
    return q_accel * np.block([
        [dt**3 / 3.0 * I3, dt**2 / 2.0 * I3],
        [dt**2 / 2.0 * I3, dt * I3],
    ])


def native_Q(model, coe_c_mean: np.ndarray, Q_lvlh: np.ndarray) -> np.ndarray:
    """Map an LVLH-frame Q into the model's native frame (J^-1 Q J^-T)."""
    if model.frame == "lvlh":
        return Q_lvlh
    J = model.to_lvlh_jacobian(coe_c_mean)
    return np.linalg.solve(J, np.linalg.solve(J, Q_lvlh.T).T)


def calibrate_q_accel(residuals_lvlh: np.ndarray, dt: float) -> float:
    """Fit the white-acceleration PSD to one-step prediction residuals.

    residuals_lvlh: (N, 6) differences (truth - STM prediction) over steps of
    length dt. Uses the velocity block (variance q*dt) as the estimator, which
    is the best-conditioned entry of the white-accel model.
    """
    v_res = np.asarray(residuals_lvlh, dtype=float)[:, 3:]
    return float(np.mean(v_res**2) / dt)
