"""KGD near-circular J2 secular ROE STM vs exact secular propagation and truth."""

import numpy as np
import pytest

from eilj2.brouwer import mean_to_osc, osc_to_mean
from eilj2.elements import period, roe_from_coe, rv2coe, coe_deputy_from_roe
from eilj2.stm.gim_alfriend import GimAlfriendModel
from eilj2.stm.kgd import kgd_stm
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe

CHIEF = np.array([7078137.0, 0.001, np.deg2rad(60.0), 0.5, 1.0, 0.3])


def test_kgd_matches_numerical_secular_jacobian():
    """The analytic STM must match the numerically-linearized exact secular map."""
    dt = 6000.0
    Sigma = GimAlfriendModel()._Sigma(CHIEF, dt)
    Phi = kgd_stm(CHIEF, dt)
    np.testing.assert_allclose(Phi, Sigma, rtol=0.03, atol=2e-3)


def test_kgd_identity_at_zero():
    np.testing.assert_allclose(kgd_stm(CHIEF, 0.0), np.eye(6))


def test_kgd_da_invariant_and_drift_signs():
    dt = 86400.0
    Phi = kgd_stm(CHIEF, dt)
    np.testing.assert_allclose(Phi[0], np.eye(6)[0], atol=1e-15)  # da conserved
    assert Phi[1, 0] < 0.0        # +da -> backward along-track drift
    assert Phi[5, 4] > 0.0        # +dix -> +diy drift (prograde, i=60 deg)


@pytest.mark.slow
def test_kgd_predicts_truth_mean_roe_drift():
    droe0 = np.array([0.0, 1e-4, 2e-5, -1e-5, 4e-5, 1e-5])
    coe_d = coe_deputy_from_roe(CHIEF, droe0)
    tw = TwoSatTruth(TruthConfig(n_zonal=2))  # J2-only truth
    y0 = state_from_coe(mean_to_osc(CHIEF), mean_to_osc(coe_d))
    tf = 20 * period(CHIEF[0])
    y1 = tw.propagate(y0, np.array([0.0, tf]))[-1]

    coe_c1 = osc_to_mean(rv2coe(y1[0:3], y1[3:6]))
    coe_d1 = osc_to_mean(rv2coe(y1[6:9], y1[9:12]))
    droe_true = roe_from_coe(coe_c1, coe_d1)
    droe_pred = kgd_stm(CHIEF, tf) @ droe0

    # compare the CHANGE, which isolates the J2 drift terms
    change_true = droe_true - droe0
    change_pred = droe_pred - droe0
    denom = max(np.max(np.abs(change_true)), 1e-9)
    assert np.max(np.abs(change_pred - change_true)) / denom < 0.05
