"""Zonal-harmonic Earth gravity: point mass + J2, J3, J4.

Acceleration formulas follow Vallado, *Fundamentals of Astrodynamics and
Applications* (4th ed.), Sec. 8.7 (unnormalized zonal harmonics).
``potential_zonal`` implements the corresponding potential with the
convention ``a = grad(U)`` (so U = mu/r for a point mass), and the unit tests
verify ``accel_zonal`` against a numerical gradient of ``potential_zonal`` —
any transcription error in either function fails that test.

Frame: Earth-centered inertial (ECI) with z along the Earth spin axis. A
zonal-only field is axisymmetric, so the Earth rotation angle never enters.
"""

from __future__ import annotations

import numpy as np

from .constants import J2, J3, J4, MU_EARTH, R_EARTH


def potential_zonal(
    r_eci: np.ndarray,
    n_max: int = 4,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    j2: float = J2,
    j3: float = J3,
    j4: float = J4,
) -> float:
    """Gravitational potential U(r) with a = grad(U), zonals through J{n_max}."""
    x, y, z = r_eci
    r = np.sqrt(x * x + y * y + z * z)
    s = z / r  # sin(geocentric latitude)
    u = 1.0
    if n_max >= 2:
        u -= j2 * (re / r) ** 2 * 0.5 * (3.0 * s * s - 1.0)
    if n_max >= 3:
        u -= j3 * (re / r) ** 3 * 0.5 * (5.0 * s**3 - 3.0 * s)
    if n_max >= 4:
        u -= j4 * (re / r) ** 4 * 0.125 * (35.0 * s**4 - 30.0 * s * s + 3.0)
    return mu / r * u


def accel_zonal(
    r_eci: np.ndarray,
    n_max: int = 4,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    j2: float = J2,
    j3: float = J3,
    j4: float = J4,
) -> np.ndarray:
    """Inertial acceleration [m/s^2] from point mass + zonals through J{n_max}.

    n_max = 0 or 1 gives pure two-body; 2, 3, 4 add J2, J3, J4 cumulatively.
    """
    x, y, z = r_eci
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    r3 = r2 * r

    ax = -mu * x / r3
    ay = -mu * y / r3
    az = -mu * z / r3

    if n_max >= 2:
        # Vallado 4th ed., Eq. (8-30)
        k2 = -1.5 * j2 * mu * re**2 / r2 / r3
        zr2 = z * z / r2
        ax += k2 * x * (1.0 - 5.0 * zr2)
        ay += k2 * y * (1.0 - 5.0 * zr2)
        az += k2 * z * (3.0 - 5.0 * zr2)

    if n_max >= 3:
        # Vallado 4th ed., Eq. (8-31)
        k3 = -2.5 * j3 * mu * re**3 / (r2 * r2 * r3)
        ax += k3 * x * (3.0 * z - 7.0 * z**3 / r2)
        ay += k3 * y * (3.0 * z - 7.0 * z**3 / r2)
        az += k3 * (6.0 * z * z - 7.0 * z**4 / r2 - 0.6 * r2)

    if n_max >= 4:
        # Vallado 4th ed., Eq. (8-32)
        k4 = 1.875 * j4 * mu * re**4 / (r2 * r2 * r3)
        zr2 = z * z / r2
        zr4 = zr2 * zr2
        ax += k4 * x * (1.0 - 14.0 * zr2 + 21.0 * zr4)
        ay += k4 * y * (1.0 - 14.0 * zr2 + 21.0 * zr4)
        az += k4 * z * (5.0 - 70.0 / 3.0 * zr2 + 21.0 * zr4)

    return np.array([ax, ay, az])
