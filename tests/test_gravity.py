"""Gravity-field self-consistency: acceleration must equal grad(potential)."""

import numpy as np
import pytest

from eilj2.gravity import accel_zonal, potential_zonal


def _num_grad(f, r, h=20.0):
    # h balances truncation against roundoff: U ~ 6e7 m^2/s^2, so double-
    # precision cancellation in the central difference is ~6e-9/(2h).
    g = np.zeros(3)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = h
        g[k] = (f(r + dp) - f(r - dp)) / (2.0 * h)
    return g


@pytest.mark.parametrize("n_max", [0, 2, 3, 4])
@pytest.mark.parametrize("r", [
    np.array([7000e3, 0.0, 0.0]),
    np.array([4000e3, 4000e3, 3000e3]),
    np.array([-2000e3, 5000e3, -4500e3]),
    np.array([1000e3, -2000e3, 6800e3]),  # high latitude
])
def test_accel_is_gradient_of_potential(n_max, r):
    a = accel_zonal(r, n_max=n_max)
    g = _num_grad(lambda x: potential_zonal(x, n_max=n_max), r)
    np.testing.assert_allclose(a, g, rtol=1e-5, atol=5e-10)


def test_j2_dominates_higher_zonals():
    r = np.array([7000e3, 0.0, 500e3])
    a0 = accel_zonal(r, n_max=0)
    d2 = np.linalg.norm(accel_zonal(r, n_max=2) - a0)
    d3 = np.linalg.norm(accel_zonal(r, n_max=3) - accel_zonal(r, n_max=2))
    d4 = np.linalg.norm(accel_zonal(r, n_max=4) - accel_zonal(r, n_max=3))
    assert d2 > 100.0 * d3
    assert d2 > 100.0 * d4
