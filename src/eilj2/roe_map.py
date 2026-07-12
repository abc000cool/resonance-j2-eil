"""Linear map between scaled quasi-nonsingular ROE and LVLH Cartesian state.

Near-circular first-order mapping (D'Amico 2010; Sullivan, Grimberg &
D'Amico, JGCD 2017). With u = omega + M the chief mean argument of latitude,
n the chief mean motion, and the SCALED ROE state

    s = a_c * [da, dlambda, dex, dey, dix, diy]   (all six in meters)

the LVLH state [x, y, z, xdot, ydot, zdot] (x radial, y along-track,
z cross-track) is x_lvlh = M(u, n) @ s with

    x  =  da - dex cos u - dey sin u
    y  =  dlambda + 2 dex sin u - 2 dey cos u
    z  =  dix sin u - diy cos u
    x' =  n ( dex sin u - dey cos u )
    y' =  -(3/2) n da + 2 n ( dex cos u + dey sin u )
    z' =  n ( dix cos u + diy sin u )

(each line implicitly scaled by a_c). Valid to first order in ROE and
eccentricity; this is the standard mapping used for formation design and
navigation in the D'Amico-school literature.
"""

from __future__ import annotations

import numpy as np

from .elements import mean_arg_latitude, mean_motion


def roe_to_lvlh_matrix(coe_c_mean: np.ndarray) -> np.ndarray:
    """6x6 map from scaled ROE (meters) to LVLH state [m, m/s]."""
    a = coe_c_mean[0]
    n = mean_motion(a)
    u = mean_arg_latitude(coe_c_mean)
    s, c = np.sin(u), np.cos(u)
    return np.array([
        [1.0,          0.0, -c,          -s,          0.0,     0.0],
        [0.0,          1.0, 2.0 * s,     -2.0 * c,    0.0,     0.0],
        [0.0,          0.0, 0.0,         0.0,         s,       -c],
        [0.0,          0.0, n * s,       -n * c,      0.0,     0.0],
        [-1.5 * n,     0.0, 2.0 * n * c, 2.0 * n * s, 0.0,     0.0],
        [0.0,          0.0, 0.0,         0.0,         n * c,   n * s],
    ])


def lvlh_from_roe(scaled_roe: np.ndarray, coe_c_mean: np.ndarray) -> np.ndarray:
    return roe_to_lvlh_matrix(coe_c_mean) @ scaled_roe


def roe_from_lvlh(x_lvlh: np.ndarray, coe_c_mean: np.ndarray) -> np.ndarray:
    return np.linalg.solve(roe_to_lvlh_matrix(coe_c_mean), x_lvlh)


def gve_control_matrix(coe_c_mean: np.ndarray) -> np.ndarray:
    """Instantaneous scaled-ROE change per RTN/LVLH delta-v [m/s].

    Gauss variational equations for quasi-nonsingular ROE, near-circular
    chief (Chernick & D'Amico, AIAA 2016-5659 Eq. 10; JGCD 2018), scaled by
    a_c so that d(a*droe) = Gamma @ [dv_R, dv_T, dv_N]:

        Gamma = (1/n) [[0, 2, 0], [-2, 0, 0], [sin u, 2 cos u, 0],
                       [-cos u, 2 sin u, 0], [0, 0, cos u], [0, 0, sin u]]
    """
    n = mean_motion(coe_c_mean[0])
    u = mean_arg_latitude(coe_c_mean)
    s, c = np.sin(u), np.cos(u)
    return (1.0 / n) * np.array([
        [0.0, 2.0, 0.0],
        [-2.0, 0.0, 0.0],
        [s, 2.0 * c, 0.0],
        [-c, 2.0 * s, 0.0],
        [0.0, 0.0, c],
        [0.0, 0.0, s],
    ])
