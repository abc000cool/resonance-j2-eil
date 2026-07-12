"""Controller correctness: LQR stability, impulsive/MPC target acquisition."""

import numpy as np
import pytest

from eilj2.controllers.impulsive import ChernickDAmicoController
from eilj2.controllers.lqr import LQRController
from eilj2.elements import mean_arg_latitude, mean_motion, propagate_mean
from eilj2.roe_map import gve_control_matrix
from eilj2.stm.cw import cw_input_matrix, cw_system_matrix
from eilj2.stm.kgd import KGDModel, arg_lat_rate

CHIEF = np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.1, 0.4, 0.9])
N = mean_motion(CHIEF[0])


def _propagate_with_burns(s0, coe0, burns, t_end, step=60.0):
    """Propagate scaled ROE through the KGD STM applying scheduled burns."""
    model = KGDModel()
    s = s0.copy()
    coe = coe0.copy()
    t = 0.0
    events = sorted(burns, key=lambda b: b[0]) + [(t_end, None)]
    for t_ev, dv in events:
        dt = t_ev - t
        if dt > 0:
            s = model.stm(coe, dt) @ s
            coe = propagate_mean(coe, dt)
            t = t_ev
        if dv is not None:
            s = s + gve_control_matrix(coe) @ dv
    return s


def test_lqr_stabilizes_cw_plant():
    A, B = cw_system_matrix(N), cw_input_matrix()
    ctrl = LQRController(A, B, r_weight=1e10, n_ref=N)
    eig = np.linalg.eigvals(A - B @ ctrl.K)
    assert np.max(eig.real) < 0.0


def test_lqr_zero_error_zero_control():
    A, B = cw_system_matrix(N), cw_input_matrix()
    ctrl = LQRController(A, B, n_ref=N)
    x = np.array([10.0, 20.0, -5.0, 0.0, 0.0, 0.0])
    assert np.allclose(ctrl.accel(0.0, x, x), 0.0)


def test_impulsive_acquires_target():
    ctrl = ChernickDAmicoController(window_orbits=1.0)
    s_des = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0]) * 20.0  # some safe geometry
    s_err0 = np.array([8.0, 60.0, 12.0, -9.0, 7.0, -5.0])    # meters of ROE error
    s_hat = s_des + s_err0

    burns = ctrl.plan(0.0, s_hat, s_des, CHIEF)
    assert burns, "controller must plan burns for a large error"
    dv_total = sum(np.linalg.norm(dv) for _, dv in burns)
    assert dv_total < 0.1  # sanity: correcting tens of meters costs << 0.1 m/s

    udot = arg_lat_rate(CHIEF)
    t_end = 2.0 * np.pi * (1.0 + 0.75) / udot  # the controller's planning horizon
    s_end = _propagate_with_burns(s_hat, CHIEF, burns, t_end)

    # drift-free comparison: where we'd be with no control
    s_free = _propagate_with_burns(s_hat, CHIEF, [], t_end)
    err_ctrl = np.linalg.norm(s_end - s_des)
    err_free = np.linalg.norm(s_free - s_des)
    assert err_ctrl < 0.15 * err_free
    assert err_ctrl < 0.1 * np.linalg.norm(s_err0)


def test_impulsive_deadband():
    ctrl = ChernickDAmicoController(deadband=100.0)
    s_des = np.zeros(6)
    s_hat = s_des + np.array([0.1, 1.0, 0.2, -0.1, 0.15, -0.05])  # ~1 m error
    assert ctrl.plan(0.0, s_hat, s_des, CHIEF) == []


def test_mpc_reduces_error():
    casadi = pytest.importorskip("casadi")  # noqa: F841
    from eilj2.controllers.mpc import MPCController

    model = KGDModel()
    ctrl = MPCController(model, horizon_orbits=6, plan_interval_orbits=6,
                         dv_weight=1.0, state_weight=1e-2)
    s_des = np.array([0.0, 0.0, 100.0, 0.0, 100.0, 0.0])
    s_hat = s_des + np.array([5.0, 40.0, 10.0, -8.0, 6.0, -4.0])

    burns = ctrl.plan(0.0, s_hat, s_des, CHIEF)
    assert burns, "MPC must act on a large error"
    udot = arg_lat_rate(CHIEF)
    t_end = 6 * 2.0 * np.pi / udot + 1.0
    s_end = _propagate_with_burns(s_hat, CHIEF, burns, t_end)
    s_free = _propagate_with_burns(s_hat, CHIEF, [], t_end)
    assert np.linalg.norm(s_end - s_des) < 0.5 * np.linalg.norm(s_free - s_des)
