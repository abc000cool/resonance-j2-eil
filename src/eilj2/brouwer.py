"""First-order J2 mean <-> osculating classical-element transformation.

Brouwer (AJ 1959) first-order J2 theory with the Lyddane-style nonsingular
recombination, as stated in Schaub & Junkins, *Analytical Mechanics of Space
Systems*, 4th ed., Appendix F, Eqs. (F.1)-(F.22). This implementation is a
faithful port of the AVS Lab Basilisk reference implementation
(``clMeanOscMap`` in ``src/utilities/orbitalMotion.py`` and its C twin,
ISC license, (c) 2016 Autonomous Vehicle Systems Lab, University of Colorado
Boulder), with the interface adapted to this package's element convention
[a, e, i, Omega, omega, M] (mean anomaly, not true anomaly).

Caveats (per the sources): long-period terms carry 1/(1 - 5 cos^2 i) and are
singular at the critical inclinations (63.43 deg / 116.57 deg); the output
recombination keeps small-e and small-i cases well-behaved (Lyddane 1963).
The osc->mean direction uses the first-order sign flip, optionally refined by
fixed-point iteration to machine-level roundtrip consistency.
"""

from __future__ import annotations

import numpy as np

from .constants import J2, R_EARTH
from .elements import (
    TWO_PI,
    mean_from_true,
    true_from_mean,
    wrap_angle,
)

# |1 - 5 cos^2 i| below this triggers a hard error: Brouwer first-order
# long-period terms blow up at the critical inclination.
_CRITICAL_INCLINATION_GUARD = 1e-6


def _brouwer_map(coe: np.ndarray, sign: int, re: float, j2: float) -> np.ndarray:
    """Core F.1-F.22 map. coe = [a, e, i, Omega, omega, M]; sign=+1 mean->osc."""
    a, e, i, Omega, omega, M = coe
    f = true_from_mean(M, e)

    ci = np.cos(i)
    one_m5c2 = 1.0 - 5.0 * ci * ci
    if abs(one_m5c2) < _CRITICAL_INCLINATION_GUARD:
        raise ValueError(
            "Brouwer first-order theory is singular at the critical inclination "
            f"(|1-5cos^2 i| = {abs(one_m5c2):.2e}); offset the inclination."
        )

    gamma2 = sign * j2 / 2.0 * (re / a) ** 2
    eta = np.sqrt(1.0 - e * e)
    gamma2p = gamma2 / eta**4
    a_r = (1.0 + e * np.cos(f)) / eta**2
    cf = np.cos(f)

    # (F.7)
    ap = a + a * gamma2 * (
        (3.0 * ci * ci - 1.0) * (a_r**3 - 1.0 / eta**3)
        + 3.0 * (1.0 - ci * ci) * a_r**3 * np.cos(2.0 * omega + 2.0 * f)
    )

    # (F.8)
    lp_bracket = 1.0 - 11.0 * ci * ci - 40.0 * ci**4 / one_m5c2
    de1 = gamma2p / 8.0 * e * eta**2 * lp_bracket * np.cos(2.0 * omega)

    # (F.9)
    de = de1 + eta**2 / 2.0 * (
        gamma2 * (
            (3.0 * ci * ci - 1.0) / eta**6
            * (e * eta + e / (1.0 + eta) + 3.0 * cf + 3.0 * e * cf * cf + e * e * cf**3)
            + 3.0 * (1.0 - ci * ci) / eta**6
            * (e + 3.0 * cf + 3.0 * e * cf * cf + e * e * cf**3) * np.cos(2.0 * omega + 2.0 * f)
        )
        - gamma2p * (1.0 - ci * ci) * (3.0 * np.cos(2.0 * omega + f) + np.cos(2.0 * omega + 3.0 * f))
    )

    # (F.10)
    di = (
        -e * de1 / (eta**2 * np.tan(i))
        + gamma2p / 2.0 * ci * np.sqrt(1.0 - ci * ci)
        * (3.0 * np.cos(2.0 * omega + 2.0 * f) + 3.0 * e * np.cos(2.0 * omega + f) + e * np.cos(2.0 * omega + 3.0 * f))
    )

    # (F.11)
    MpopOp = (
        M + omega + Omega
        + gamma2p / 8.0 * eta**3 * lp_bracket * np.sin(2.0 * omega)
        - gamma2p / 16.0 * (
            2.0 + e * e
            - 11.0 * (2.0 + 3.0 * e * e) * ci * ci
            - 40.0 * (2.0 + 5.0 * e * e) * ci**4 / one_m5c2
            - 400.0 * e * e * ci**6 / one_m5c2**2
        ) * np.sin(2.0 * omega)
        + gamma2p / 4.0 * (
            -6.0 * one_m5c2 * (f - M + e * np.sin(f))
            + (3.0 - 5.0 * ci * ci)
            * (3.0 * np.sin(2.0 * omega + 2.0 * f) + 3.0 * e * np.sin(2.0 * omega + f) + e * np.sin(2.0 * omega + 3.0 * f))
        )
        - gamma2p / 8.0 * e * e * ci * (
            11.0 + 80.0 * ci * ci / one_m5c2 + 200.0 * ci**4 / one_m5c2**2
        ) * np.sin(2.0 * omega)
        - gamma2p / 2.0 * ci * (
            6.0 * (f - M + e * np.sin(f))
            - 3.0 * np.sin(2.0 * omega + 2.0 * f)
            - 3.0 * e * np.sin(2.0 * omega + f)
            - e * np.sin(2.0 * omega + 3.0 * f)
        )
    )

    # (F.12)
    are = (a_r * eta) ** 2
    edM = (
        gamma2p / 8.0 * e * eta**3 * lp_bracket * np.sin(2.0 * omega)
        - gamma2p / 4.0 * eta**3 * (
            2.0 * (3.0 * ci * ci - 1.0) * (are + a_r + 1.0) * np.sin(f)
            + 3.0 * (1.0 - ci * ci) * (
                (-are - a_r + 1.0) * np.sin(2.0 * omega + f)
                + (are + a_r + 1.0 / 3.0) * np.sin(2.0 * omega + 3.0 * f)
            )
        )
    )

    # (F.13)
    dOmega = (
        -gamma2p / 8.0 * e * e * ci * (
            11.0 + 80.0 * ci * ci / one_m5c2 + 200.0 * ci**4 / one_m5c2**2
        ) * np.sin(2.0 * omega)
        - gamma2p / 2.0 * ci * (
            6.0 * (f - M + e * np.sin(f))
            - 3.0 * np.sin(2.0 * omega + 2.0 * f)
            - 3.0 * e * np.sin(2.0 * omega + f)
            - e * np.sin(2.0 * omega + 3.0 * f)
        )
    )

    # (F.14)-(F.22): Lyddane-style nonsingular recombination
    d1 = (e + de) * np.sin(M) + edM * np.cos(M)
    d2 = (e + de) * np.cos(M) - edM * np.sin(M)
    Mp = np.arctan2(d1, d2)
    ep = np.sqrt(d1 * d1 + d2 * d2)

    d3 = (np.sin(i / 2.0) + np.cos(i / 2.0) * di / 2.0) * np.sin(Omega) + np.sin(i / 2.0) * dOmega * np.cos(Omega)
    d4 = (np.sin(i / 2.0) + np.cos(i / 2.0) * di / 2.0) * np.cos(Omega) - np.sin(i / 2.0) * dOmega * np.sin(Omega)
    Omegap = np.arctan2(d3, d4)
    ip = 2.0 * np.arcsin(np.clip(np.sqrt(d3 * d3 + d4 * d4), -1.0, 1.0))
    omegap = MpopOp - Mp - Omegap

    return np.array([ap, ep, ip, Omegap % TWO_PI, omegap % TWO_PI, Mp % TWO_PI])


def mean_to_osc(coe_mean: np.ndarray, re: float = R_EARTH, j2: float = J2) -> np.ndarray:
    """Osculating classical elements from mean elements (first-order J2)."""
    return _brouwer_map(np.asarray(coe_mean, dtype=float), +1, re, j2)


def _to_nonsingular(coe: np.ndarray) -> np.ndarray:
    """[a, e cos w, e sin w, i, Omega, u=w+M] — well-conditioned at small e."""
    a, e, i, Om, w, M = coe
    return np.array([a, e * np.cos(w), e * np.sin(w), i, Om, (w + M) % TWO_PI])


def _from_nonsingular(v: np.ndarray) -> np.ndarray:
    a, ex, ey, i, Om, u = v
    e = np.hypot(ex, ey)
    w = np.arctan2(ey, ex) % TWO_PI
    return np.array([a, e, i, Om % TWO_PI, w, (u - w) % TWO_PI])


def osc_to_mean(
    coe_osc: np.ndarray,
    re: float = R_EARTH,
    j2: float = J2,
    iterate: bool = True,
    tol: float = 1e-12,
    max_iter: int = 15,
) -> np.ndarray:
    """Mean classical elements from osculating (sign-flipped map, then fixed point).

    The sign flip alone is accurate to O(J2^2); the fixed-point refinement
    drives mean_to_osc(osc_to_mean(x)) - x to the tolerance, following
    Orekit's computeMeanOrbit approach. The iteration runs in nonsingular
    variables (a, e cos w, e sin w, i, Omega, u): at small e the J2
    short-period eccentricity oscillation is comparable to e itself, and a
    fixed point iterated on (e, w, M) separately is ill-conditioned there
    (it can spuriously collapse e to zero).
    """
    coe_osc = np.asarray(coe_osc, dtype=float)
    mean = _brouwer_map(coe_osc, -1, re, j2)
    if not iterate:
        return mean
    target = _to_nonsingular(coe_osc)
    scale = np.array([max(abs(coe_osc[0]), 1.0), 1.0, 1.0, 1.0, 1.0, 1.0])
    for _ in range(max_iter):
        resid = target - _to_nonsingular(mean_to_osc(mean, re, j2))
        resid[4] = wrap_angle(resid[4])
        resid[5] = wrap_angle(resid[5])
        ns = _to_nonsingular(mean) + resid
        mean = _from_nonsingular(ns)
        if np.max(np.abs(resid) / scale) < tol:
            break
    return mean
