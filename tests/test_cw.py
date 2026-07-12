"""CW model: closed form vs ODE, group property, no-drift condition, truth check."""

import numpy as np
from scipy.integrate import solve_ivp

from eilj2.elements import coe2rv, mean_motion, period
from eilj2.frames import eci_to_lvlh
from eilj2.stm.cw import cw_input_matrix, cw_stm, cw_system_matrix, no_drift_ydot
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

A_REF = 7078137.0
N_REF = mean_motion(A_REF)


def test_identity_and_group():
    assert np.allclose(cw_stm(N_REF, 0.0), np.eye(6))
    t1, t2 = 500.0, 1300.0
    np.testing.assert_allclose(cw_stm(N_REF, t1 + t2),
                               cw_stm(N_REF, t2) @ cw_stm(N_REF, t1), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(cw_stm(N_REF, t1) @ cw_stm(N_REF, -t1),
                               np.eye(6), rtol=1e-9, atol=1e-12)


def test_stm_matches_ode():
    A = cw_system_matrix(N_REF)
    x0 = np.array([120.0, -300.0, 80.0, 0.05, -0.21, 0.03])
    tf = 2.5 * period(A_REF)
    sol = solve_ivp(lambda t, x: A @ x, (0, tf), x0, rtol=1e-12, atol=1e-12,
                    method="DOP853")
    np.testing.assert_allclose(cw_stm(N_REF, tf) @ x0, sol.y[:, -1],
                               rtol=1e-8, atol=1e-8)


def test_no_drift_condition_bounds_motion():
    x0 = np.array([100.0, 0.0, 0.0, 0.0, no_drift_ydot(100.0, N_REF), 0.0])
    for k in range(1, 11):
        x = cw_stm(N_REF, k * period(A_REF)) @ x0
        assert np.linalg.norm(x[:3]) < 500.0  # bounded 2:1 ellipse, no secular growth
    # drifting IC for contrast
    x_bad = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x10 = cw_stm(N_REF, 10 * period(A_REF)) @ x_bad
    assert abs(x10[1]) > 5e3


def test_cw_matches_twobody_truth():
    """Point-mass truth, circular chief: CW error ~ linearization only."""
    coe_c = np.array([A_REF, 1e-8, np.deg2rad(51.0), 0.4, 0.0, 1.0])
    tw = TwoSatTruth(TruthConfig(n_zonal=0))
    r_c, v_c = coe2rv(coe_c)
    x0 = np.array([50.0, 200.0, 30.0, 0.01, -2 * N_REF * 50.0, -0.02])
    from eilj2.frames import lvlh_to_eci

    r_d, v_d = lvlh_to_eci(r_c, v_c, x0, n_zonal=0)
    y = np.concatenate((r_c, v_c, r_d, v_d))
    tf = period(A_REF)
    y1 = tw.step(y, tf)
    x_truth = eci_to_lvlh(y1[0:3], y1[3:6], y1[6:9], y1[9:12], n_zonal=0)
    x_cw = cw_stm(N_REF, tf) @ x0
    # second-order linearization error ~ rho^2/a accumulates to ~0.1 m/orbit
    # for this ~200 m geometry (measured 0.0999 m); assert with margin
    assert np.linalg.norm(x_cw[:3] - x_truth[:3]) < 0.2


def test_input_matrix_shape():
    B = cw_input_matrix()
    assert B.shape == (6, 3) and np.allclose(B[3:], np.eye(3))
