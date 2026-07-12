"""Orbital elements, anomaly conversions, J2 secular rates, and ROE.

Conventions
-----------
A classical element set is a length-6 numpy array

    coe = [a, e, i, Omega, omega, M]

with a in meters, angles in radians, and M the *mean* anomaly. Wherever a
distinction matters, functions state whether they expect osculating or mean
elements; the ROE functions operate on mean elements (D'Amico convention).

Quasi-nonsingular relative orbital elements (ROE), dimensionless:

    droe = [da, dlambda, dex, dey, dix, diy]

    da      = (a_d - a_c) / a_c
    dlambda = (u_d - u_c) + (Omega_d - Omega_c) cos i_c   (u = omega + M)
    dex     = e_d cos omega_d - e_c cos omega_c
    dey     = e_d sin omega_d - e_c sin omega_c
    dix     = i_d - i_c
    diy     = (Omega_d - Omega_c) sin i_c

following D'Amico (2010) and Koenig, Guffanti & D'Amico (JGCD 2017).
"""

from __future__ import annotations

import numpy as np

from .constants import J2, MU_EARTH, R_EARTH

TWO_PI = 2.0 * np.pi


def wrap_angle(x):
    """Wrap angle(s) to (-pi, pi]."""
    return -((-np.asarray(x) + np.pi) % TWO_PI - np.pi)


def rot1(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def rot3(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def mean_motion(a: float, mu: float = MU_EARTH) -> float:
    return np.sqrt(mu / a**3)


def period(a: float, mu: float = MU_EARTH) -> float:
    return TWO_PI / mean_motion(a, mu)


# ---------------------------------------------------------------------------
# Anomaly conversions
# ---------------------------------------------------------------------------

def solve_kepler(M, e: float, tol: float = 1e-13, max_iter: int = 50):
    """Eccentric anomaly E from mean anomaly M (Newton's method)."""
    M = np.asarray(M, dtype=float)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M)).astype(float)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        dE = -f / (1.0 - e * np.cos(E))
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E if E.shape else float(E)


def true_from_ecc(E, e: float):
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(np.asarray(E) / 2.0),
                            np.sqrt(1.0 - e) * np.cos(np.asarray(E) / 2.0))


def ecc_from_true(nu, e: float):
    return 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(np.asarray(nu) / 2.0),
                            np.sqrt(1.0 + e) * np.cos(np.asarray(nu) / 2.0))


def true_from_mean(M, e: float):
    return true_from_ecc(solve_kepler(M, e), e)


def mean_from_true(nu, e: float):
    E = ecc_from_true(nu, e)
    return E - e * np.sin(E)


# ---------------------------------------------------------------------------
# Element <-> Cartesian
# ---------------------------------------------------------------------------

def coe2rv(coe: np.ndarray, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """ECI position/velocity from classical elements [a, e, i, Omega, omega, M]."""
    a, e, i, raan, argp, M = coe
    nu = true_from_mean(M, e)
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(nu))
    r_pf = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    R = rot3(-raan) @ rot1(-i) @ rot3(-argp)  # perifocal -> ECI
    return R @ r_pf, R @ v_pf


def rv2coe(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> np.ndarray:
    """Classical elements [a, e, i, Omega, omega, M] from ECI state.

    Robust for small (but nonzero) eccentricity and inclination via
    atan2-based quadrant resolution. For exactly circular/equatorial orbits
    the ambiguous angles are set to zero by convention.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = np.linalg.norm(r)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    h_hat = h / hn

    e_vec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(e_vec)
    energy = 0.5 * np.dot(v, v) - mu / rn
    a = -mu / (2.0 * energy)
    inc = np.arccos(np.clip(h_hat[2], -1.0, 1.0))

    n_vec = np.array([-h[1], h[0], 0.0])  # z_hat x h
    nn = np.linalg.norm(n_vec)
    if nn < 1e-12 * hn:  # equatorial
        n_hat = np.array([1.0, 0.0, 0.0])
        raan = 0.0
    else:
        n_hat = n_vec / nn
        raan = np.arctan2(n_hat[1], n_hat[0])

    if e < 1e-13:  # circular: measure true anomaly from the node
        e_hat = n_hat
        argp = 0.0
    else:
        e_hat = e_vec / e
        argp = np.arctan2(np.dot(np.cross(n_hat, e_hat), h_hat), np.dot(n_hat, e_hat))

    r_hat = r / rn
    nu = np.arctan2(np.dot(np.cross(e_hat, r_hat), h_hat), np.dot(e_hat, r_hat))
    M = mean_from_true(nu, e)
    return np.array([a, e, inc, raan % TWO_PI, argp % TWO_PI, M % TWO_PI])


# ---------------------------------------------------------------------------
# J2 secular rates and mean-element propagation
# ---------------------------------------------------------------------------

def j2_secular_rates(
    a: float, e: float, i: float,
    mu: float = MU_EARTH, re: float = R_EARTH, j2: float = J2,
) -> tuple[float, float, float]:
    """First-order J2 secular rates (Omega_dot, omega_dot, M_dot_J2) [rad/s].

    M_dot_J2 is the perturbation on top of the Keplerian mean motion n:
        Omega_dot = -(3/2) n J2 (Re/p)^2 cos i
        omega_dot = +(3/4) n J2 (Re/p)^2 (5 cos^2 i - 1)
        M_dot_J2  = +(3/4) n J2 (Re/p)^2 sqrt(1-e^2) (3 cos^2 i - 1)
    """
    n = mean_motion(a, mu)
    p = a * (1.0 - e * e)
    eta = np.sqrt(1.0 - e * e)
    k = 1.5 * j2 * (re / p) ** 2 * n
    ci = np.cos(i)
    raan_dot = -k * ci
    argp_dot = 0.5 * k * (5.0 * ci * ci - 1.0)
    m_dot_j2 = 0.5 * k * eta * (3.0 * ci * ci - 1.0)
    return raan_dot, argp_dot, m_dot_j2


def propagate_mean(coe: np.ndarray, dt: float,
                   mu: float = MU_EARTH, re: float = R_EARTH, j2: float = J2) -> np.ndarray:
    """Propagate MEAN classical elements under first-order J2 secular theory."""
    a, e, i, raan, argp, M = coe
    n = mean_motion(a, mu)
    raan_dot, argp_dot, m_dot_j2 = j2_secular_rates(a, e, i, mu, re, j2)
    return np.array([
        a, e, i,
        (raan + raan_dot * dt) % TWO_PI,
        (argp + argp_dot * dt) % TWO_PI,
        (M + (n + m_dot_j2) * dt) % TWO_PI,
    ])


def mean_arg_latitude(coe: np.ndarray) -> float:
    """u = omega + M (mean argument of latitude)."""
    return (coe[4] + coe[5]) % TWO_PI


# ---------------------------------------------------------------------------
# Quasi-nonsingular relative orbital elements
# ---------------------------------------------------------------------------

def roe_from_coe(coe_c: np.ndarray, coe_d: np.ndarray) -> np.ndarray:
    """Dimensionless quasi-nonsingular ROE of deputy w.r.t. chief (mean elements)."""
    a_c, e_c, i_c, O_c, w_c, M_c = coe_c
    a_d, e_d, i_d, O_d, w_d, M_d = coe_d
    dO = wrap_angle(O_d - O_c)
    da = (a_d - a_c) / a_c
    du = wrap_angle((w_d + M_d) - (w_c + M_c))
    dlam = du + dO * np.cos(i_c)
    dex = e_d * np.cos(w_d) - e_c * np.cos(w_c)
    dey = e_d * np.sin(w_d) - e_c * np.sin(w_c)
    dix = i_d - i_c
    diy = dO * np.sin(i_c)
    return np.array([da, dlam, dex, dey, dix, diy])


def coe_deputy_from_roe(coe_c: np.ndarray, droe: np.ndarray) -> np.ndarray:
    """Deputy mean classical elements from chief elements and dimensionless ROE."""
    a_c, e_c, i_c, O_c, w_c, M_c = coe_c
    da, dlam, dex, dey, dix, diy = droe
    a_d = a_c * (1.0 + da)
    i_d = i_c + dix
    dO = diy / np.sin(i_c)
    O_d = (O_c + dO) % TWO_PI
    ex_d = e_c * np.cos(w_c) + dex
    ey_d = e_c * np.sin(w_c) + dey
    e_d = np.hypot(ex_d, ey_d)
    w_d = np.arctan2(ey_d, ex_d) % TWO_PI
    u_d = (w_c + M_c) + dlam - dO * np.cos(i_c)
    M_d = (u_d - w_d) % TWO_PI
    return np.array([a_d, e_d, i_d, O_d, w_d, M_d])
