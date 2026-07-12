"""Element conversions, Kepler solver, ROE roundtrips."""

import numpy as np
import pytest

from eilj2.constants import MU_EARTH
from eilj2.elements import (
    coe2rv,
    coe_deputy_from_roe,
    j2_secular_rates,
    mean_from_true,
    roe_from_coe,
    rv2coe,
    solve_kepler,
    true_from_mean,
)

COE_CASES = [
    np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.3, 1.2, 0.7]),
    np.array([6878137.0, 0.01, np.deg2rad(45.0), 2.0, 4.0, 3.0]),
    np.array([26560e3, 0.7, np.deg2rad(63.0), 5.5, 0.1, 6.0]),
]


def test_kepler_solver():
    for e in (0.0, 0.001, 0.3, 0.9):
        M = np.linspace(0.0, 2 * np.pi, 33)
        E = solve_kepler(M, e)
        np.testing.assert_allclose(E - e * np.sin(E), M, atol=1e-12)


def test_anomaly_roundtrip():
    for e in (0.001, 0.2, 0.8):
        M = np.linspace(-np.pi + 0.01, np.pi - 0.01, 21)
        M2 = mean_from_true(true_from_mean(M, e), e)
        np.testing.assert_allclose(np.mod(M2 - M + np.pi, 2 * np.pi) - np.pi,
                                   0.0, atol=1e-11)


@pytest.mark.parametrize("coe", COE_CASES)
def test_coe_rv_roundtrip(coe):
    r, v = coe2rv(coe)
    back = rv2coe(r, v)
    np.testing.assert_allclose(back[0], coe[0], rtol=1e-10)
    np.testing.assert_allclose(back[1], coe[1], atol=1e-10)
    for idx in (2, 3, 4, 5):
        d = np.mod(back[idx] - coe[idx] + np.pi, 2 * np.pi) - np.pi
        assert abs(d) < 1e-8


def test_vis_viva():
    coe = COE_CASES[0]
    r, v = coe2rv(coe)
    energy = 0.5 * v @ v - MU_EARTH / np.linalg.norm(r)
    np.testing.assert_allclose(energy, -MU_EARTH / (2 * coe[0]), rtol=1e-12)


def test_roe_roundtrip():
    chief = np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.3, 1.2, 0.7])
    droe = np.array([1e-5, -2e-5, 3e-5, -1e-5, 2e-5, 1.5e-5])
    deputy = coe_deputy_from_roe(chief, droe)
    np.testing.assert_allclose(roe_from_coe(chief, deputy), droe, atol=1e-12)


def test_j2_secular_rates_signs():
    # prograde LEO: node regresses; SSO designed so raan_dot ~ +2 pi/year
    rd, wd, md = j2_secular_rates(7078137.0, 0.001, np.deg2rad(45.0))
    assert rd < 0.0 and wd > 0.0 and md > 0.0
    rd_sso, _, _ = j2_secular_rates(7078137.0, 0.001, np.deg2rad(98.0))
    assert rd_sso > 0.0  # retrograde-inclination orbits precess eastward
    # critical inclination: argp_dot = 0
    _, wd_crit, _ = j2_secular_rates(7078137.0, 0.001, np.arccos(np.sqrt(0.2)))
    assert abs(wd_crit) < 1e-15
