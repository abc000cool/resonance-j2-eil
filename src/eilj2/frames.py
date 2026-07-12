"""LVLH (Hill) frame construction and exact ECI <-> LVLH relative-state maps.

Axes convention (proposal Sec. 1.1): x radial (nadir-to-zenith), z along the
orbit angular momentum, y = z cross x (close to along-track for near-circular
orbits). The rotation-matrix time derivative is computed exactly from the
chief acceleration — including the J2-induced orbit-plane precession — so
LVLH relative velocities are consistent with the truth propagator to the
accuracy of the gravity model, not just to the Keplerian approximation.
"""

from __future__ import annotations

import numpy as np

from .gravity import accel_zonal


def lvlh_dcm(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Direction cosine matrix R mapping ECI vectors into the LVLH frame.

    Rows of R are the LVLH basis vectors [x_hat; y_hat; z_hat] in ECI.
    """
    rn = np.linalg.norm(r)
    x_hat = r / rn
    h = np.cross(r, v)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack((x_hat, y_hat, z_hat))


def lvlh_dcm_rate(r: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Time derivative of the LVLH DCM, exact given the chief acceleration a."""
    rn = np.linalg.norm(r)
    x_hat = r / rn
    x_hat_dot = (v - np.dot(x_hat, v) * x_hat) / rn

    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    z_hat = h / hn
    h_dot = np.cross(r, a)  # r x a  (r x v-dot term; v x v = 0)
    z_hat_dot = (h_dot - np.dot(z_hat, h_dot) * z_hat) / hn

    y_hat_dot = np.cross(z_hat_dot, x_hat) + np.cross(z_hat, x_hat_dot)
    return np.vstack((x_hat_dot, y_hat_dot, z_hat_dot))


def eci_to_lvlh(
    r_c: np.ndarray, v_c: np.ndarray, r_d: np.ndarray, v_d: np.ndarray,
    n_zonal: int = 4,
) -> np.ndarray:
    """Relative state [rho; rho_dot] of deputy w.r.t. chief in the chief LVLH frame.

    rho_dot is the LVLH-frame-relative velocity (as seen by an observer
    rotating with the frame), computed with the exact frame rate.
    """
    a_c = accel_zonal(r_c, n_max=n_zonal)
    R = lvlh_dcm(r_c, v_c)
    R_dot = lvlh_dcm_rate(r_c, v_c, a_c)
    dr = r_d - r_c
    dv = v_d - v_c
    rho = R @ dr
    rho_dot = R @ dv + R_dot @ dr
    return np.concatenate((rho, rho_dot))


def lvlh_to_eci(
    r_c: np.ndarray, v_c: np.ndarray, x_lvlh: np.ndarray,
    n_zonal: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Deputy ECI state from chief ECI state and LVLH relative state."""
    a_c = accel_zonal(r_c, n_max=n_zonal)
    R = lvlh_dcm(r_c, v_c)
    R_dot = lvlh_dcm_rate(r_c, v_c, a_c)
    rho = x_lvlh[:3]
    rho_dot = x_lvlh[3:]
    dr = R.T @ rho
    dv = R.T @ (rho_dot - R_dot @ dr)
    return r_c + dr, v_c + dv
