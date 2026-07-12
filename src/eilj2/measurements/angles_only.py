"""Angles-only optical relative navigation: bearing of the deputy line of
sight in the chief LVLH frame.

Parameterization: with rho = (x, y, z) the LVLH relative position,

    az = atan2(x, y)      in-plane bearing from the along-track axis
    el = asin(z / |rho|)  out-of-plane elevation

which is regular for the along-track-separated geometries flown in this
study (singular only for a line of sight along cross-track). Typical camera
accuracy 10-100 arcsec (proposal Sec. 1.8). Range is unobservable at the
CW-linear level (Woffinden dilemma); J2 and the closed-loop maneuvers break
the ambiguity — that recovery is part of what the study measures.
"""

from __future__ import annotations

import numpy as np

from .base import MeasurementModel

ARCSEC = np.pi / (180.0 * 3600.0)


class AnglesOnly(MeasurementModel):
    name = "angles_only"
    dim = 2

    def __init__(self, sigma_angle: float):
        """sigma_angle in radians (use eilj2.measurements.angles_only.ARCSEC)."""
        self.sigma_angle = float(sigma_angle)

    def h(self, x_lvlh: np.ndarray) -> np.ndarray:
        x, y, z = x_lvlh[:3]
        r = np.linalg.norm(x_lvlh[:3])
        return np.array([np.arctan2(x, y), np.arcsin(z / r)])

    def jacobian(self, x_lvlh: np.ndarray) -> np.ndarray:
        x, y, z = x_lvlh[:3]
        r2xy = x * x + y * y
        r = np.linalg.norm(x_lvlh[:3])
        rxy = np.sqrt(r2xy)
        H = np.zeros((2, 6))
        H[0, 0] = y / r2xy
        H[0, 1] = -x / r2xy
        H[1, 0] = -x * z / (r * r * rxy)
        H[1, 1] = -y * z / (r * r * rxy)
        H[1, 2] = rxy / (r * r)
        return H

    @property
    def R(self) -> np.ndarray:
        return np.diag([self.sigma_angle**2] * 2)
