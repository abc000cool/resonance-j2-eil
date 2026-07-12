"""Brouwer mean<->osculating map: roundtrip, variance reduction, secular rates."""

import numpy as np
import pytest

from eilj2.brouwer import mean_to_osc, osc_to_mean
from eilj2.elements import j2_secular_rates, period, rv2coe, wrap_angle
from eilj2.truth import TruthConfig, TwoSatTruth, state_from_coe


def test_roundtrip():
    coe_mean = np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.5, 1.0, 2.0])
    coe_osc = mean_to_osc(coe_mean)
    # osculating differs from mean at the J2 short-period scale (~km in a)
    assert 100.0 < abs(coe_osc[0] - coe_mean[0]) < 20e3
    back = osc_to_mean(coe_osc)
    np.testing.assert_allclose(back[0], coe_mean[0], atol=1e-5)
    np.testing.assert_allclose(back[1:3], coe_mean[1:3], atol=1e-11)
    # at small e, omega and M are individually ill-conditioned but the
    # well-determined combinations must round-trip tightly:
    assert abs(wrap_angle(back[3] - coe_mean[3])) < 1e-10           # Omega
    assert abs(wrap_angle((back[4] + back[5]) - (coe_mean[4] + coe_mean[5]))) < 1e-10  # u
    e_m, w_m = coe_mean[1], coe_mean[4]
    np.testing.assert_allclose(back[1] * np.cos(back[4]), e_m * np.cos(w_m), atol=1e-11)
    np.testing.assert_allclose(back[1] * np.sin(back[4]), e_m * np.sin(w_m), atol=1e-11)


@pytest.mark.slow
def test_mean_elements_are_quiet_and_drift_at_analytic_rates():
    coe_mean0 = np.array([7078137.0, 0.001, np.deg2rad(60.0), 0.5, 1.0, 2.0])
    tw = TwoSatTruth(TruthConfig(n_zonal=2))  # J2-only truth, matches the theory
    y0 = state_from_coe(mean_to_osc(coe_mean0), mean_to_osc(coe_mean0))
    T = period(coe_mean0[0])
    times = np.linspace(0.0, 10 * T, 400)
    Y = tw.propagate(y0, times)

    osc = np.array([rv2coe(y[0:3], y[3:6]) for y in Y])
    mean = np.array([osc_to_mean(o) for o in osc])

    # mean a should be far quieter than osculating a
    assert np.std(mean[:, 0]) < 0.05 * np.std(osc[:, 0])

    # node regression rate matches first-order theory to ~1%
    raan_dot_fit = np.polyfit(times, np.unwrap(mean[:, 3]), 1)[0]
    raan_dot_th, _, _ = j2_secular_rates(*coe_mean0[:3])
    np.testing.assert_allclose(raan_dot_fit, raan_dot_th, rtol=0.01)
