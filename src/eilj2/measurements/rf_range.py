"""Inter-satellite RF ranging: scalar range and (optionally) range-rate.

Typical accuracy 10 cm - 10 m depending on link budget (proposal Sec. 1.8).
Range alone is weakly observable in the along-track direction; range-rate
(Doppler) is included by default.
"""

from __future__ import annotations

import numpy as np

from .base import MeasurementModel


class RFRange(MeasurementModel):
    name = "rf_range"

    def __init__(self, sigma_range: float, include_rate: bool = True,
                 sigma_rate: float | None = None):
        self.sigma_range = float(sigma_range)
        self.include_rate = include_rate
        # Doppler-derived range-rate: ~1e-3 /s of the range accuracy by default
        self.sigma_rate = float(sigma_rate) if sigma_rate is not None else 1e-3 * self.sigma_range
        self.dim = 2 if include_rate else 1

    def h(self, x_lvlh: np.ndarray) -> np.ndarray:
        rho = x_lvlh[:3]
        rhod = x_lvlh[3:]
        r = np.linalg.norm(rho)
        if self.include_rate:
            return np.array([r, float(rho @ rhod) / r])
        return np.array([r])

    def jacobian(self, x_lvlh: np.ndarray) -> np.ndarray:
        rho = x_lvlh[:3]
        rhod = x_lvlh[3:]
        r = np.linalg.norm(rho)
        H = np.zeros((self.dim, 6))
        H[0, :3] = rho / r
        if self.include_rate:
            rr = float(rho @ rhod) / r
            H[1, :3] = rhod / r - rr * rho / r**2
            H[1, 3:] = rho / r
        return H

    @property
    def R(self) -> np.ndarray:
        if self.include_rate:
            return np.diag([self.sigma_range**2, self.sigma_rate**2])
        return np.array([[self.sigma_range**2]])
