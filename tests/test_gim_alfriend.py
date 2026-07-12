"""Gim-Alfriend numerically-linearized STM vs J2 truth and CW baseline."""

import numpy as np
import pytest

from eilj2.brouwer import mean_to_osc
from eilj2.elements import coe_deputy_from_roe, mean_motion, period, propagate_mean
from eilj2.frames import eci_to_lvlh
from eilj2.stm.cw import cw_stm
from eilj2.stm.gim_alfriend import GimAlfriendModel
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

CHIEF = np.array([7078137.0, 0.001, np.deg2rad(45.0), 0.2, 0.4, 0.1])
DROE = np.array([0.0, 1.4e-4, 2e-5, -1e-5, 3e-5, 1e-5])  # ~1 km formation


@pytest.mark.slow
def test_ga_beats_cw_under_j2():
    coe_d = coe_deputy_from_roe(CHIEF, DROE)
    tw = TwoSatTruth(TruthConfig(n_zonal=2))
    y0 = state_from_coe(mean_to_osc(CHIEF), mean_to_osc(coe_d))
    x0 = eci_to_lvlh(y0[0:3], y0[3:6], y0[6:9], y0[9:12], n_zonal=2)

    T = period(CHIEF[0])
    n_orbits, step = 5, 600.0
    times = np.arange(0.0, n_orbits * T + step, step)
    Y = tw.propagate(y0, times)

    ga = GimAlfriendModel()
    n = mean_motion(CHIEF[0])
    x_ga, x_cw = x0.copy(), x0.copy()
    coe_c = CHIEF.copy()
    err_ga, err_cw = [], []
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        x_ga = ga.stm(coe_c, dt) @ x_ga
        x_cw = cw_stm(n, dt) @ x_cw
        coe_c = propagate_mean(coe_c, dt)
        y = Y[k]
        x_true = eci_to_lvlh(y[0:3], y[3:6], y[6:9], y[9:12], n_zonal=2)
        err_ga.append(np.linalg.norm(x_ga[:3] - x_true[:3]))
        err_cw.append(np.linalg.norm(x_cw[:3] - x_true[:3]))

    # GA must clearly beat CW under J2. Measured: ~19 m vs ~76 m over 5 orbits
    # at 1 km (the GA residual is O(J2^2) secular truncation of first-order
    # Brouwer theory, not implementation error).
    assert max(err_ga) < 0.35 * max(err_cw)
    assert max(err_ga) < 25.0
