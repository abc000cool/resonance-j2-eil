"""Error-state Kalman filters (EKF, Joseph form) and UKF ablation."""

from __future__ import annotations

from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter

__all__ = ["ExtendedKalmanFilter", "UnscentedKalmanFilter", "get_filter"]


def get_filter(kind: str, x0, P0):
    key = kind.lower()
    if key == "ekf":
        return ExtendedKalmanFilter(x0, P0)
    if key == "ukf":
        return UnscentedKalmanFilter(x0, P0)
    raise ValueError(f"unknown filter type {kind!r}")
