"""Shared interface for analytic relative-motion models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RelativeMotionModel(ABC):
    """See package docstring in eilj2.stm for the interface contract."""

    frame: str = "lvlh"  # or "roe"
    name: str = "base"

    @abstractmethod
    def stm(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        """6x6 STM over dt, evaluated at the chief mean elements at segment start."""

    def to_lvlh(self, state: np.ndarray, coe_c_mean: np.ndarray) -> np.ndarray:
        return self.to_lvlh_jacobian(coe_c_mean) @ state

    def from_lvlh(self, x_lvlh: np.ndarray, coe_c_mean: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.to_lvlh_jacobian(coe_c_mean), x_lvlh)

    def to_lvlh_jacobian(self, coe_c_mean: np.ndarray) -> np.ndarray:
        """d(x_lvlh)/d(state); identity for LVLH-native models."""
        return np.eye(6)

    def control_input(self, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        """6x3: native-state change at segment end per unit LVLH impulse at start.

        Default: impulse changes the LVLH velocity instantaneously, then
        propagates through the STM. ROE-native models override with the
        Gauss-variational-equation input matrix.
        """
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        if self.frame == "lvlh":
            return self.stm(coe_c_mean, dt) @ B
        # ROE-native: LVLH delta-v maps to a native-state jump through the
        # inverse of the (linear) native->LVLH map, then propagates.
        J = self.to_lvlh_jacobian(coe_c_mean)
        return self.stm(coe_c_mean, dt) @ np.linalg.solve(J, B)

    def propagate(self, state: np.ndarray, coe_c_mean: np.ndarray, dt: float) -> np.ndarray:
        return self.stm(coe_c_mean, dt) @ state
