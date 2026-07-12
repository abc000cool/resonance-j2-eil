"""Formation geometry families (proposal Sec. 1.6 and Phase C).

Each family maps a scalar size L [m] (and optional phase) to a dimensionless
quasi-nonsingular ROE vector for the desired formation. All families use
da = 0 (no energy offset -> no Keplerian along-track drift by design; the
J2 differential drift is what the controller must fight).

Families
--------
along_track   pure leader-follower: dlambda = L/a.
pco           projected circular orbit of radius L in the along-track/cross-
              track (y-z) plane: a*|de| = L/2, a*|di| = L, theta = phi - 90 deg.
gero          general elliptical relative orbit: a*|de| = L/2, a*|di| = L/2,
              theta = phi - 90 deg (relaxed amplitude ratio).
ei_safe       E/I-vector-separation safe geometry (D'Amico & Montenbruck,
              JGCD 2006): parallel relative e- and i-vectors (theta = phi),
              a*|de| = a*|di| = L/2, so the radial-normal separation never
              collapses even under along-track uncertainty.
"""

from __future__ import annotations

import numpy as np

GEOMETRY_FAMILIES = ("along_track", "pco", "gero", "ei_safe")


def geometry_roe(family: str, L: float, a_chief: float, phase: float = 0.0) -> np.ndarray:
    """Dimensionless desired ROE [da, dlambda, dex, dey, dix, diy]."""
    fam = family.lower()
    phi = phase
    if fam == "along_track":
        return np.array([0.0, L / a_chief, 0.0, 0.0, 0.0, 0.0])
    if fam == "pco":
        de = L / (2.0 * a_chief)
        di = L / a_chief
        theta = phi - np.pi / 2.0
    elif fam == "gero":
        de = L / (2.0 * a_chief)
        di = L / (2.0 * a_chief)
        theta = phi - np.pi / 2.0
    elif fam == "ei_safe":
        de = L / (2.0 * a_chief)
        di = L / (2.0 * a_chief)
        theta = phi
    else:
        raise ValueError(f"unknown geometry family {family!r}")
    return np.array([
        0.0, 0.0,
        de * np.cos(phi), de * np.sin(phi),
        di * np.cos(theta), di * np.sin(theta),
    ])
