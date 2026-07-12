"""Relative-navigation measurement models (proposal Sec. 1.8).

All models measure functions of the LVLH relative state of the deputy with
respect to the chief; filters whose native state is ROE compose these
Jacobians with the ROE->LVLH map.
"""

from __future__ import annotations

from .angles_only import AnglesOnly
from .base import MeasurementModel
from .cdgps import CDGPS
from .rf_range import RFRange

__all__ = ["MeasurementModel", "CDGPS", "RFRange", "AnglesOnly", "get_measurement"]


def get_measurement(kind: str, sigma: float, **kwargs) -> MeasurementModel:
    """Factory. `sigma` is the architecture's headline 1-sigma accuracy:

    - cdgps:  relative-position sigma per axis [m]
    - rf:     range sigma [m] (range-rate sigma scales as sigma * 1e-3 /s
              unless overridden)
    - angles: bearing sigma [rad]
    """
    key = kind.lower()
    if key in ("cdgps", "gps"):
        return CDGPS(sigma_pos=sigma, **kwargs)
    if key in ("rf", "rf_range", "range"):
        return RFRange(sigma_range=sigma, **kwargs)
    if key in ("angles", "angles_only", "optical"):
        return AnglesOnly(sigma_angle=sigma, **kwargs)
    raise ValueError(f"unknown measurement architecture {kind!r}")
