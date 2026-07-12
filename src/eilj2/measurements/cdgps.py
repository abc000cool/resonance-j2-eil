"""Carrier-phase differential GPS: direct relative-position (and optionally
relative-velocity) measurements in the LVLH frame.

Flight heritage: PRISMA, TanDEM-X, GRACE-FO. Typical 1-sigma relative
position: 1 cm - 1 m depending on baseline/processing (Kroes 2006;
Montenbruck et al. 2011). Modeled as unbiased white noise per axis.
"""

from __future__ import annotations

import numpy as np

from .base import MeasurementModel


class CDGPS(MeasurementModel):
    name = "cdgps"

    def __init__(self, sigma_pos: float, include_velocity: bool = False,
                 sigma_vel: float | None = None):
        self.sigma_pos = float(sigma_pos)
        self.include_velocity = include_velocity
        # CDGPS velocity accuracy scales roughly 1e-3/s of the position accuracy
        self.sigma_vel = float(sigma_vel) if sigma_vel is not None else 1e-3 * self.sigma_pos
        self.dim = 6 if include_velocity else 3

    def h(self, x_lvlh: np.ndarray) -> np.ndarray:
        return x_lvlh[: self.dim].copy()

    def jacobian(self, x_lvlh: np.ndarray) -> np.ndarray:
        H = np.zeros((self.dim, 6))
        H[:, : self.dim] = np.eye(self.dim)
        return H

    @property
    def R(self) -> np.ndarray:
        if self.include_velocity:
            return np.diag([self.sigma_pos**2] * 3 + [self.sigma_vel**2] * 3)
        return np.diag([self.sigma_pos**2] * 3)
