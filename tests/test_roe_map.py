"""ROE <-> LVLH linear map and GVE input matrix vs exact nonlinear geometry."""

import numpy as np

from eilj2.elements import (
    coe2rv,
    coe_deputy_from_roe,
    propagate_mean,
    roe_from_coe,
    rv2coe,
)
from eilj2.frames import eci_to_lvlh
from eilj2.roe_map import gve_control_matrix, lvlh_from_roe, roe_from_lvlh
from eilj2.truth import TwoSatTruth, TruthConfig, state_from_coe

CHIEF = np.array([7078137.0, 0.001, np.deg2rad(98.0), 0.3, 0.8, 0.2])
DROE = np.array([2e-6, 6e-5, 3e-5, -2e-5, 2.5e-5, 1.5e-5])  # ~few hundred m


def test_map_matches_nonlinear_geometry_along_orbit():
    """Two-body (J2=0) so mean == osculating; map error is first-order only."""
    for frac in np.linspace(0.0, 1.0, 7):
        dt = frac * 5900.0
        coe_c = propagate_mean(CHIEF, dt, j2=0.0)
        coe_d = propagate_mean(coe_deputy_from_roe(CHIEF, DROE), dt, j2=0.0)
        r_c, v_c = coe2rv(coe_c)
        r_d, v_d = coe2rv(coe_d)
        x_exact = eci_to_lvlh(r_c, v_c, r_d, v_d, n_zonal=0)
        s = roe_from_coe(coe_c, coe_d) * coe_c[0]
        x_map = lvlh_from_roe(s, coe_c)
        # first-order map: error ~ O(e + rho/a) * rho ~ meters for ~500 m rho
        assert np.linalg.norm(x_map[:3] - x_exact[:3]) < 5.0
        assert np.linalg.norm(x_map[3:] - x_exact[3:]) < 5e-3


def test_map_inverse_consistency():
    s = DROE * CHIEF[0]
    x = lvlh_from_roe(s, CHIEF)
    np.testing.assert_allclose(roe_from_lvlh(x, CHIEF), s, rtol=1e-12)


def test_gve_matrix_against_finite_burn():
    """Apply a small RTN delta-v in the two-body truth; ROE jump ~ Gamma dv."""
    tw = TwoSatTruth(TruthConfig(n_zonal=0))
    coe_d = coe_deputy_from_roe(CHIEF, DROE)
    y = state_from_coe(CHIEF, coe_d)

    dv = np.array([0.002, -0.003, 0.004])  # m/s
    y2 = tw.apply_impulse_lvlh(y, dv)

    def scaled_roe(yy):
        cc = rv2coe(yy[0:3], yy[3:6])
        cd = rv2coe(yy[6:9], yy[9:12])
        return roe_from_coe(cc, cd) * cc[0]

    ds_true = scaled_roe(y2) - scaled_roe(y)
    ds_lin = gve_control_matrix(CHIEF) @ dv
    # first order in e and dv: ~1% agreement
    np.testing.assert_allclose(ds_lin, ds_true, rtol=0.02, atol=2e-3)
