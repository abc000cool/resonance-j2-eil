"""Filter consistency (NEES/NIS) on a synthetic linear CW problem."""

import numpy as np

from eilj2.elements import mean_motion
from eilj2.filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from eilj2.filters.noise import white_accel_Q
from eilj2.measurements import CDGPS
from eilj2.stm.cw import cw_stm

N = mean_motion(7078137.0)
DT = 60.0
STEPS = 400


def _run_linear_scenario(filter_cls, seed=7):
    rng = np.random.default_rng(seed)
    Phi = cw_stm(N, DT)
    Q = white_accel_Q(1e-12, DT)
    Lq = np.linalg.cholesky(Q + 1e-30 * np.eye(6))
    meas = CDGPS(sigma_pos=0.1)

    x_true = np.array([30.0, 500.0, 20.0, 0.01, -2 * N * 30.0, 0.005])
    P0 = np.diag([25.0] * 3 + [1e-4] * 3)
    x0 = x_true + np.linalg.cholesky(P0) @ rng.standard_normal(6)
    filt = filter_cls(x0, P0)

    nees, nis = [], []
    for _ in range(STEPS):
        x_true = Phi @ x_true + Lq @ rng.standard_normal(6)
        filt.predict(Phi, Q)
        z = meas.sample(x_true, rng)
        if isinstance(filt, UnscentedKalmanFilter):
            innov, S = filt.update(z, meas.h, meas.R)
        else:
            innov, S = filt.update(z, meas.h(filt.x), meas.jacobian(filt.x), meas.R)
        e = filt.x - x_true
        nees.append(e @ np.linalg.solve(filt.P, e))
        nis.append(innov @ np.linalg.solve(S, innov))
    return np.mean(nees), np.mean(nis), np.linalg.norm(filt.x[:3] - x_true[:3])


def test_ekf_consistency():
    mean_nees, mean_nis, final_err = _run_linear_scenario(ExtendedKalmanFilter)
    assert 4.0 < mean_nees < 8.5     # chi-square dof 6
    assert 2.0 < mean_nis < 4.5      # chi-square dof 3
    assert final_err < 0.15          # converged well below initial 5 m sigma


def test_ukf_matches_ekf_on_linear_problem():
    nees_e, nis_e, err_e = _run_linear_scenario(ExtendedKalmanFilter, seed=3)
    nees_u, nis_u, err_u = _run_linear_scenario(UnscentedKalmanFilter, seed=3)
    # identical linear problem, same noise draws: statistics must agree closely
    assert abs(nees_e - nees_u) < 0.5
    assert abs(nis_e - nis_u) < 0.3
    assert abs(err_e - err_u) < 0.05
