"""End-to-end closed-loop smoke tests on the fast screening tier."""

import numpy as np
import pytest

from eilj2.simulate import SimConfig, run_sim


def _base_cfg(**kw) -> SimConfig:
    defaults = dict(
        truth="stm", truth_model="kgd", duration_days=0.3, dt=60.0,
        family="ei_safe", size=1000.0, filter_model="kgd", filter_kind="ekf",
        meas_kind="cdgps", meas_sigma=0.1, seed=42,
        ctrl_model="cw",  # CW LTI plant for LQR in smoke tests
    )
    defaults.update(kw)
    return SimConfig(**defaults)


def test_uncontrolled_runs():
    res = run_sim(_base_cfg(controller="none"))
    assert not res.summary["diverged"]
    assert res.summary["dv_total"] == 0.0
    assert np.isfinite(res.summary["rms_pos_err"])


def test_lqr_closed_loop():
    res = run_sim(_base_cfg(controller="lqr"))
    s = res.summary
    assert not s["diverged"]
    assert 0.0 < s["dv_total"] < 1.0
    assert s["rms_pos_err"] < 200.0


def test_impulsive_closed_loop():
    res = run_sim(_base_cfg(controller="impulsive", duration_days=0.5))
    s = res.summary
    assert not s["diverged"]
    assert 0.0 < s["dv_total"] < 1.0


def test_perfect_nav_baseline():
    res = run_sim(_base_cfg(controller="lqr", filter_kind="perfect"))
    s = res.summary
    assert not s["diverged"]
    assert s["rms_pos_err"] < 100.0


def test_ukf_runs():
    res = run_sim(_base_cfg(controller="lqr", filter_kind="ukf"))
    assert not res.summary["diverged"]


def test_rf_and_angles_architectures():
    for kind, sigma in (("rf", 1.0), ("angles", 2e-4)):
        res = run_sim(_base_cfg(controller="lqr", meas_kind=kind, meas_sigma=sigma))
        assert not res.summary["diverged"], kind


def test_mpc_closed_loop():
    pytest.importorskip("casadi")
    res = run_sim(_base_cfg(controller="mpc", duration_days=0.5))
    s = res.summary
    assert not s["diverged"]
    assert np.isfinite(s["dv_total"])


def test_numerical_truth_short():
    res = run_sim(_base_cfg(truth="numerical", controller="lqr",
                            duration_days=0.1))
    s = res.summary
    assert not s["diverged"]
    assert s["rms_pos_err"] < 300.0


def test_history_recording():
    res = run_sim(_base_cfg(controller="lqr", duration_days=0.1),
                  record_history=True)
    h = res.history
    assert h is not None and len(h["t"]) == len(h["err_pos"])
