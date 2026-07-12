"""RK4 fast path vs DOP853, and GA cache identities."""

import numpy as np

from eilj2.brouwer import mean_to_osc
from eilj2.elements import coe_deputy_from_roe, period
from eilj2.stm.gim_alfriend import GimAlfriendModel
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

CHIEF = np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.5, 1.0, 0.3])
DROE = np.array([0.0, 1.4e-4, 2e-5, -1e-5, 3e-5, 1e-5])


def test_rk4_matches_dop853_over_an_orbit():
    coe_d = coe_deputy_from_roe(CHIEF, DROE)
    y0 = state_from_coe(mean_to_osc(CHIEF), mean_to_osc(coe_d))
    T = period(CHIEF[0])

    tw_hi = TwoSatTruth(TruthConfig(n_zonal=4, method="DOP853"))
    tw_fast = TwoSatTruth(TruthConfig(n_zonal=4, method="RK4", rk4_substep=10.0))

    y_hi, y_fast = y0.copy(), y0.copy()
    for _ in range(int(T // 60.0)):
        y_hi = tw_hi.step(y_hi, 60.0)
        y_fast = tw_fast.step(y_fast, 60.0)

    # absolute agreement: ~1.3 cm/orbit along-track phase (common mode)
    assert np.linalg.norm(y_hi[0:3] - y_fast[0:3]) < 0.05
    assert np.linalg.norm(y_hi[6:9] - y_fast[6:9]) < 0.05
    # relative (deputy - chief) state — the quantity under study — agrees to
    # micrometers (measured 3e-6 m): the RK4 truncation is common mode
    rel_hi = y_hi[6:9] - y_hi[0:3]
    rel_fast = y_fast[6:9] - y_fast[0:3]
    assert np.linalg.norm(rel_hi - rel_fast) < 1e-4


def test_ga_b_is_inverse_of_a():
    ga = GimAlfriendModel()
    A = ga._A(CHIEF)
    B = ga._B(CHIEF)
    np.testing.assert_allclose(B @ A, np.eye(6), atol=5e-5)


def test_ga_stm_near_identity_at_small_dt():
    # inv(A)-based composition guarantees Phi(0) = I exactly, unlike the
    # independently finite-differenced B (which carries ~1e-4 FD error that
    # amplifies through the velocity rows)
    ga = GimAlfriendModel()
    Phi = ga.stm(CHIEF, 1e-6)
    np.testing.assert_allclose(Phi, np.eye(6), rtol=1e-5, atol=1e-5)
