"""Schweighart-Sedwick model: coefficients, STM properties, truth comparison."""

import numpy as np
import pytest

from eilj2.brouwer import mean_to_osc
from eilj2.elements import coe_deputy_from_roe, mean_motion, period
from eilj2.frames import eci_to_lvlh
from eilj2.stm.cw import cw_stm
from eilj2.stm.schweighart import (
    SchweighartSedwickModel,
    ss_coefficients,
    ss_system_matrix,
)
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

CHIEF = np.array([7078137.0, 0.001, np.deg2rad(45.0), 0.2, 0.4, 0.1])


def test_coefficients_limits():
    a = 7078137.0
    # J2 -> 0 recovers CW: s -> 0, c -> 1, k -> n
    co = ss_coefficients(a, np.deg2rad(45.0))
    n = mean_motion(a)
    assert abs(co["s"]) < 1e-3
    assert abs(co["c"] - 1.0) < 1e-3
    assert abs(co["k"] / n - 1.0) < 2e-3
    # sign of s: positive at low inclination, negative near polar
    assert ss_coefficients(a, np.deg2rad(10.0))["s"] > 0.0
    assert ss_coefficients(a, np.deg2rad(90.0))["s"] < 0.0


def test_stm_group_property_and_cache():
    m = SchweighartSedwickModel()
    Phi1 = m.stm(CHIEF, 500.0)
    Phi2 = m.stm(CHIEF, 1000.0)
    np.testing.assert_allclose(Phi1 @ Phi1, Phi2, rtol=1e-10, atol=1e-12)
    assert m.stm(CHIEF, 500.0) is Phi1  # cached object


def test_stm_matches_ode_matrix():
    from scipy.integrate import solve_ivp

    A = ss_system_matrix(CHIEF)
    x0 = np.array([50.0, -120.0, 80.0, 0.02, -0.11, 0.01])
    tf = 3000.0
    sol = solve_ivp(lambda t, x: A @ x, (0, tf), x0, rtol=1e-12, atol=1e-12,
                    method="DOP853")
    m = SchweighartSedwickModel()
    np.testing.assert_allclose(m.stm(CHIEF, tf) @ x0, sol.y[:, -1],
                               rtol=1e-8, atol=1e-8)


def test_in_plane_eigenfrequency():
    # in-plane characteristic frequency of the S-S plant is n sqrt(2 - c^2)
    A = ss_system_matrix(CHIEF)
    co = ss_coefficients(CHIEF[0], CHIEF[2])
    eig = np.linalg.eigvals(A[np.ix_([0, 1, 3, 4], [0, 1, 3, 4])])
    freq = np.max(np.abs(eig.imag))
    assert abs(eig.real).max() < 1e-12
    np.testing.assert_allclose(freq, co["n"] * np.sqrt(2.0 - co["c"] ** 2),
                               rtol=1e-9)


@pytest.mark.slow
def test_ss_not_worse_than_cw_under_j2():
    """For close satellites the differential-J2 drift dominates and is
    captured by neither LTI model, so S-S ~ CW in-plane (measured ~5% apart);
    this guards against regressions making S-S actively worse. The S-S
    fidelity gap vs GA/KGD is a *result* of the paper (Fig. 2), not a test."""
    droe = np.array([0.0, 1.4e-4, 2e-5, -1e-5, 0.0, 0.0])  # in-plane only
    coe_d = coe_deputy_from_roe(CHIEF, droe)
    tw = TwoSatTruth(TruthConfig(n_zonal=2))
    y0 = state_from_coe(mean_to_osc(CHIEF), mean_to_osc(coe_d))
    x0 = eci_to_lvlh(y0[0:3], y0[3:6], y0[6:9], y0[9:12], n_zonal=2)

    T = period(CHIEF[0])
    n = mean_motion(CHIEF[0])
    m = SchweighartSedwickModel()
    step = 600.0
    times = np.arange(0.0, 5 * T + step, step)
    Y = tw.propagate(y0, times)

    x_ss, x_cw = x0.copy(), x0.copy()
    err_ss, err_cw = [], []
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        x_ss = m.stm(CHIEF, dt) @ x_ss
        x_cw = cw_stm(n, dt) @ x_cw
        y = Y[k]
        x_true = eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=2)
        err_ss.append(np.linalg.norm(x_ss[:2] - x_true[:2]))  # in-plane
        err_cw.append(np.linalg.norm(x_cw[:2] - x_true[:2]))

    assert max(err_ss) < 1.15 * max(err_cw)
