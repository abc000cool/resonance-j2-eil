"""Measurement-model interface: z = h(x_lvlh) + v, v ~ N(0, R)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MeasurementModel(ABC):
    dim: int
    name: str = "base"

    @abstractmethod
    def h(self, x_lvlh: np.ndarray) -> np.ndarray:
        """Noise-free measurement of the LVLH relative state."""

    @abstractmethod
    def jacobian(self, x_lvlh: np.ndarray) -> np.ndarray:
        """dh/dx, shape (dim, 6), with respect to the LVLH state."""

    @property
    @abstractmethod
    def R(self) -> np.ndarray:
        """Measurement noise covariance, shape (dim, dim)."""

    def sample(self, x_lvlh: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Noisy measurement (R is diagonal for every model in this study)."""
        return self.h(x_lvlh) + np.sqrt(np.diag(self.R)) * rng.standard_normal(self.dim)
