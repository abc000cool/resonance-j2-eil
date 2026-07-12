"""Scalar performance metrics: delta-V, formation-keeping error, drift rates,
and filter-consistency statistics (NEES/NIS)."""

from __future__ import annotations

import numpy as np

from .constants import SECONDS_PER_YEAR


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def annualized_dv(dv_total: float, duration_s: float) -> float:
    """Total delta-V [m/s] over the campaign scaled to m/s per year."""
    return dv_total * SECONDS_PER_YEAR / duration_s


def position_error_stats(err_xyz: np.ndarray) -> dict[str, float]:
    """RMS and 99th-percentile of the position-error magnitude time series.

    err_xyz: (N, 3) LVLH position error relative to the desired trajectory.
    """
    mag = np.linalg.norm(np.asarray(err_xyz, dtype=float), axis=1)
    return {
        "rms_pos_err": float(np.sqrt(np.mean(mag**2))),
        "p99_pos_err": float(np.percentile(mag, 99.0)),
        "max_pos_err": float(np.max(mag)),
    }


def drift_rate(t: np.ndarray, err_mag: np.ndarray, robust: bool = True) -> float:
    """Linear drift rate [m/s] of the error magnitude via (robust) regression.

    With robust=True uses a decimated Theil-Sen estimator (median of pairwise
    slopes), which ignores the periodic component of the relative motion far
    better than least squares.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(err_mag, dtype=float)
    if not robust:
        return float(np.polyfit(t, e, 1)[0])
    # decimate to <= 200 samples so pairwise slopes stay cheap
    step = max(1, len(t) // 200)
    ts, es = t[::step], e[::step]
    i, j = np.triu_indices(len(ts), k=1)
    slopes = (es[j] - es[i]) / (ts[j] - ts[i])
    return float(np.median(slopes))


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Normalized estimation error squared per epoch.

    err: (N, n) state errors; P: (N, n, n) covariances. For a consistent
    filter, mean NEES ~ n (chi-square with n dof).
    """
    err = np.asarray(err, dtype=float)
    out = np.empty(len(err))
    for k in range(len(err)):
        out[k] = err[k] @ np.linalg.solve(P[k], err[k])
    return out


def nis(innov: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Normalized innovation squared per update (chi-square with m dof)."""
    innov = np.asarray(innov, dtype=float)
    out = np.empty(len(innov))
    for k in range(len(innov)):
        out[k] = innov[k] @ np.linalg.solve(S[k], innov[k])
    return out
