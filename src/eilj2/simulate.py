"""Closed-loop estimation-in-the-loop simulation engine (proposal Sec. 1.8, 3.4).

One simulation couples:

- a truth backend — the numerical J2-J4 propagator ("numerical") or an
  analytic-STM surrogate ("stm", used by the fast screening tier);
- a navigation filter (EKF/UKF) whose prediction model is one of the four
  analytic STMs (CW / S-S / GA / KGD), or perfect navigation;
- a measurement architecture (CDGPS / RF range / angles-only) at a given
  1-sigma accuracy;
- a formation-keeping controller (LQR / MPC / impulsive Chernick-D'Amico /
  none), acting on the ESTIMATE: u = u(x_hat) — the certainty-equivalence
  coupling under study.

Axes: LVLH x radial, y along-track, z cross-track (= RTN). The chief's
absolute state is assumed known (standard relative-navigation assumption);
its mean elements are recovered from the truth via the Brouwer osc->mean map
each step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .brouwer import mean_to_osc, osc_to_mean
from .constants import MU_EARTH
from .elements import mean_motion, period, propagate_mean, rv2coe, coe_deputy_from_roe
from .filters import get_filter
from .filters.noise import native_Q, white_accel_Q
from .geometry import geometry_roe
from .metrics import annualized_dv, position_error_stats
from .roe_map import lvlh_from_roe, roe_from_lvlh, roe_to_lvlh_matrix
from .stm import get_model
from .measurements import get_measurement
from .truth import TruthConfig, TwoSatTruth, state_from_coe


@dataclass
class SimConfig:
    # chief orbit (MEAN elements; angles in degrees for config ergonomics)
    a: float = 7078137.0          # 700 km altitude
    e: float = 0.001
    i_deg: float = 98.0
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    M0_deg: float = 0.0
    # formation geometry
    family: str = "ei_safe"
    size: float = 1000.0          # [m]
    phase_deg: float = 0.0
    # duration / cadence
    duration_days: float = 30.0
    dt: float = 60.0              # step = measurement cadence = LQR ZOH [s]
    plan_interval_orbits: float = 1.0
    # architecture
    truth: str = "numerical"      # "numerical" | "stm"
    truth_model: str = "kgd"      # STM used when truth == "stm"
    truth_n_zonal: int = 4
    truth_process_q: float = 0.0  # accel PSD injected into STM truth
    filter_model: str = "kgd"
    filter_kind: str = "ekf"      # "ekf" | "ukf" | "perfect"
    controller: str = "lqr"       # "lqr" | "mpc" | "impulsive" | "none"
    ctrl_model: str = "ss"        # LTI plant for LQR; prediction model for MPC/impulsive
    # measurement
    meas_kind: str = "cdgps"
    meas_sigma: float = 0.1
    meas_every_n: int = 1         # measurements every n steps
    # filter tuning. q_accel = None selects the per-model PSD calibrated
    # against the J2-J4 numerical truth by scripts/calibrate_q.py — each
    # filter runs with its own honest Q (fair STM ablation).
    q_accel: float | None = None  # process-noise accel PSD [(m/s^2)^2 s]
    p0_pos: float = 50.0          # initial 1-sigma position [m]
    p0_vel: float = 0.05          # initial 1-sigma velocity [m/s]
    init_nav_err: bool = True
    # post-acquisition residual dispersion (fraction of size). Large values
    # combined with a saturated actuator can produce an unrecoverable
    # delta-a drift runaway — real physics, but not the regime under study.
    init_roe_disp: float = 0.005
    # LQR
    lqr_q_pos: float = 1.0
    lqr_r_weight: float = 3e11
    u_max: float | None = 2e-4    # [m/s^2]
    # MPC
    mpc_horizon_orbits: int = 10
    mpc_dv_weight: float = 1.0
    mpc_state_weight: float = 1e-3
    mpc_impulses_per_orbit: int = 4
    # impulsive
    imp_deadband: float = 0.0     # [m] of scaled-ROE error; below -> no burns
    # control warm-up: no maneuvers until the filter has converged (standard
    # commissioning practice; avoids burning fuel on transient nav error)
    ctrl_warmup_orbits: float = 2.0
    # bookkeeping
    seed: int = 0
    label: str = ""

    def chief_coe_mean(self) -> np.ndarray:
        d = np.pi / 180.0
        return np.array([
            self.a, self.e, self.i_deg * d,
            self.raan_deg * d, self.argp_deg * d, self.M0_deg * d,
        ])


# ---------------------------------------------------------------------------
# Truth backends
# ---------------------------------------------------------------------------

class _NumericalTruth:
    """Wraps TwoSatTruth; keeps chief mean elements via Brouwer osc->mean."""

    def __init__(self, cfg: SimConfig, droe_true0: np.ndarray):
        self.tw = TwoSatTruth(TruthConfig(n_zonal=cfg.truth_n_zonal))
        coe_c_mean = cfg.chief_coe_mean()
        coe_d_mean = coe_deputy_from_roe(coe_c_mean, droe_true0)
        self.y = state_from_coe(mean_to_osc(coe_c_mean), mean_to_osc(coe_d_mean))
        self._chief_mean = coe_c_mean.copy()

    def chief_mean(self) -> np.ndarray:
        return self._chief_mean

    def rel_lvlh(self) -> np.ndarray:
        return self.tw.relative_lvlh(self.y)

    def scaled_roe(self) -> np.ndarray:
        coe_c = osc_to_mean(rv2coe(self.y[0:3], self.y[3:6]))
        coe_d = osc_to_mean(rv2coe(self.y[6:9], self.y[9:12]))
        from .elements import roe_from_coe

        return roe_from_coe(coe_c, coe_d) * coe_c[0]

    def step(self, dt: float, u_lvlh: np.ndarray | None) -> None:
        self.y = self.tw.step(self.y, dt, u_lvlh)
        self._chief_mean = osc_to_mean(rv2coe(self.y[0:3], self.y[3:6]))

    def impulse(self, dv_lvlh: np.ndarray) -> None:
        self.y = self.tw.apply_impulse_lvlh(self.y, dv_lvlh)


class _STMTruth:
    """Screening-tier truth: relative state propagated by a designated STM.

    The chief mean elements advance analytically; the scaled-ROE truth state
    advances by the truth STM plus (optionally) white-acceleration process
    noise mapped into ROE space, representing unmodeled dynamics.
    """

    def __init__(self, cfg: SimConfig, droe_true0: np.ndarray,
                 rng: np.random.Generator):
        self.model = get_model(cfg.truth_model)
        self.coe = cfg.chief_coe_mean()
        self.s = droe_true0 * self.coe[0]  # scaled ROE [m]
        self.q = cfg.truth_process_q
        self.rng = rng
        self._pending_u: np.ndarray | None = None

    def chief_mean(self) -> np.ndarray:
        return self.coe

    def rel_lvlh(self) -> np.ndarray:
        if self.model.frame == "roe":
            return self.model.to_lvlh(self.s, self.coe)
        return self.s  # LVLH-native truth model

    def scaled_roe(self) -> np.ndarray:
        if self.model.frame == "roe":
            return self.s
        return roe_from_lvlh(self.s, self.coe)

    def step(self, dt: float, u_lvlh: np.ndarray | None) -> None:
        Phi = self.model.stm(self.coe, dt)
        s_new = Phi @ self.s
        if u_lvlh is not None:
            s_new = s_new + self.model.control_input(self.coe, dt) @ (u_lvlh * dt)
        if self.q > 0.0:
            Qn = native_Q(self.model, self.coe, white_accel_Q(self.q, dt))
            s_new = s_new + np.linalg.cholesky(
                Qn + 1e-30 * np.eye(6)) @ self.rng.standard_normal(6)
        self.s = s_new
        self.coe = propagate_mean(self.coe, dt)

    def impulse(self, dv_lvlh: np.ndarray) -> None:
        # instantaneous velocity change mapped into the native frame
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        if self.model.frame == "roe":
            J = self.model.to_lvlh_jacobian(self.coe)
            self.s = self.s + np.linalg.solve(J, B @ dv_lvlh)
        else:
            self.s = self.s + B @ dv_lvlh


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

def _make_controller(cfg: SimConfig, coe_c_mean: np.ndarray):
    n = mean_motion(cfg.a)
    if cfg.controller == "none":
        return None
    if cfg.controller in ("lqr", "lqr_cov"):
        key = cfg.ctrl_model.lower()
        if key in ("ss", "schweighart", "schweighart_sedwick"):
            from .stm.schweighart import ss_system_matrices

            A, B = ss_system_matrices(coe_c_mean)
        else:  # CW fallback plant
            from .stm.cw import cw_input_matrix, cw_system_matrix

            A, B = cw_system_matrix(n), cw_input_matrix()
        if cfg.controller == "lqr_cov":
            from .controllers.lqr import CovarianceAwareLQR

            return CovarianceAwareLQR(A, B, q_pos=cfg.lqr_q_pos,
                                      r_weight=cfg.lqr_r_weight,
                                      u_max=cfg.u_max, n_ref=n,
                                      p_ref=max(cfg.meas_sigma, 1e-3))
        from .controllers.lqr import LQRController

        return LQRController(A, B, q_pos=cfg.lqr_q_pos, r_weight=cfg.lqr_r_weight,
                             u_max=cfg.u_max, n_ref=n)
    if cfg.controller == "impulsive":
        from .controllers.impulsive import ChernickDAmicoController

        return ChernickDAmicoController(deadband=cfg.imp_deadband)
    if cfg.controller == "mpc":
        from .controllers.mpc import MPCController

        return MPCController(
            model=get_model(cfg.ctrl_model if cfg.ctrl_model not in ("ss", "cw") else "kgd"),
            horizon_orbits=cfg.mpc_horizon_orbits,
            plan_interval_orbits=cfg.plan_interval_orbits,
            dv_weight=cfg.mpc_dv_weight,
            state_weight=cfg.mpc_state_weight,
            impulses_per_orbit=cfg.mpc_impulses_per_orbit,
        )
    raise ValueError(f"unknown controller {cfg.controller!r}")


# ---------------------------------------------------------------------------
# The closed loop
# ---------------------------------------------------------------------------

# scripts/calibrate_q.py output (J2-J4 truth, 700 km SSO, 60 s steps)
CALIBRATED_Q_ACCEL = {
    "cw": 6.3e-10,
    "ss": 5.5e-10,
    "schweighart": 5.5e-10,
    "gim_alfriend": 8.0e-13,
    "ga": 8.0e-13,
    "kgd": 1.0e-14,
}


@dataclass
class SimResult:
    summary: dict[str, Any]
    history: dict[str, np.ndarray] | None = None


def run_sim(cfg: SimConfig, record_history: bool = False) -> SimResult:
    rng = np.random.default_rng(np.random.SeedSequence(cfg.seed))
    coe_c0 = cfg.chief_coe_mean()
    a_c = coe_c0[0]
    T_orbit = period(a_c)
    n_steps = int(round(cfg.duration_days * 86400.0 / cfg.dt))
    plan_every = max(1, int(round(cfg.plan_interval_orbits * T_orbit / cfg.dt)))

    # desired geometry and truth initial dispersion
    droe_des = geometry_roe(cfg.family, cfg.size, a_c, np.deg2rad(cfg.phase_deg))
    s_des = droe_des * a_c
    disp = cfg.init_roe_disp * cfg.size / a_c
    droe_true0 = droe_des + disp * rng.standard_normal(6)

    # truth backend
    if cfg.truth == "numerical":
        truth = _NumericalTruth(cfg, droe_true0)
    elif cfg.truth == "stm":
        truth = _STMTruth(cfg, droe_true0, rng)
    else:
        raise ValueError(f"unknown truth backend {cfg.truth!r}")

    # navigation
    fmodel = get_model(cfg.filter_model)
    q_accel = (cfg.q_accel if cfg.q_accel is not None
               else CALIBRATED_Q_ACCEL.get(cfg.filter_model.lower(), 1e-13))
    meas = get_measurement(cfg.meas_kind, cfg.meas_sigma)
    P0_lvlh = np.diag([cfg.p0_pos**2] * 3 + [cfg.p0_vel**2] * 3)
    coe_now = truth.chief_mean()
    P0 = native_Q(fmodel, coe_now, P0_lvlh)

    def truth_native() -> np.ndarray:
        if fmodel.frame == "roe":
            return roe_from_lvlh(truth.rel_lvlh(), truth.chief_mean())
        return truth.rel_lvlh()

    perfect_nav = cfg.filter_kind == "perfect"
    if perfect_nav:
        filt = None
    else:
        x0 = truth_native()
        if cfg.init_nav_err:
            x0 = x0 + np.linalg.cholesky(P0 + 1e-30 * np.eye(6)) @ rng.standard_normal(6)
        filt = get_filter(cfg.filter_kind, x0, P0)

    controller = _make_controller(cfg, coe_c0)

    # bookkeeping
    dv_total = 0.0
    pending_burns: list[tuple[int, np.ndarray]] = []  # (step index, dv)
    nis_list: list[float] = []
    nees_list: list[float] = []
    err_pos_all: list[np.ndarray] = []  # always kept: primary metric
    diverged = False
    hist: dict[str, list] = {k: [] for k in
                             ("t", "err_pos", "err_hat", "dv_cum", "sroe_err")} if record_history else {}

    divergence_limit = max(100.0 * cfg.size, 50e3)

    for k in range(n_steps):
        t = k * cfg.dt
        coe_now = truth.chief_mean()
        x_ref = lvlh_from_roe(s_des, coe_now)
        x_true_lvlh = truth.rel_lvlh()

        # navigation output available to the controller
        if perfect_nav:
            x_hat_native = truth_native()
        else:
            x_hat_native = filt.x
        x_hat_lvlh = fmodel.to_lvlh(x_hat_native, coe_now)

        # -- controller ------------------------------------------------------
        in_warmup = t < cfg.ctrl_warmup_orbits * T_orbit and not perfect_nav
        u = None
        if in_warmup:
            pass
        elif controller is not None and controller.kind == "continuous":
            if hasattr(controller, "set_covariance") and not perfect_nav:
                J = fmodel.to_lvlh_jacobian(coe_now)
                controller.set_covariance(J @ filt.P @ J.T)
            u = controller.accel(t, x_hat_lvlh, x_ref)
            dv_total += float(np.linalg.norm(u)) * cfg.dt
        elif controller is not None and controller.kind == "impulsive":
            # replan only when the previous maneuver scheme has fully executed
            if k % plan_every == 0 and not pending_burns:
                s_hat = (x_hat_native if fmodel.frame == "roe"
                         else roe_from_lvlh(x_hat_lvlh, coe_now))
                burns = controller.plan(t, s_hat, s_des, coe_now)
                pending_burns = [
                    (k + max(1, int(round((tb - t) / cfg.dt))), dv) for tb, dv in burns
                ]
        dv_step = np.zeros(3)
        still_pending = []
        for kb, dv in pending_burns:
            if kb <= k:
                truth.impulse(dv)
                dv_total += float(np.linalg.norm(dv))
                dv_step = dv_step + dv
            else:
                still_pending.append((kb, dv))
        pending_burns = still_pending

        # -- truth propagation over [t, t+dt] ---------------------------------
        truth.step(cfg.dt, u)
        coe_next = truth.chief_mean()

        # -- filter predict + control feed-through ----------------------------
        if not perfect_nav:
            Phi = fmodel.stm(coe_now, cfg.dt)
            Qn = native_Q(fmodel, coe_now, white_accel_Q(q_accel, cfg.dt))
            filt.predict(Phi, Qn)
            G = fmodel.control_input(coe_now, cfg.dt)
            if u is not None:
                filt.x = filt.x + G @ (u * cfg.dt)
            if np.any(dv_step):
                filt.x = filt.x + G @ dv_step

            # -- measurement update -------------------------------------------
            if (k + 1) % cfg.meas_every_n == 0:
                z = meas.sample(truth.rel_lvlh(), rng)
                J = fmodel.to_lvlh_jacobian(coe_next)
                if cfg.filter_kind == "ukf":
                    innov, S = filt.update(z, lambda s: meas.h(J @ s), meas.R)
                else:
                    xh_lvlh = fmodel.to_lvlh(filt.x, coe_next)
                    H = meas.jacobian(xh_lvlh) @ J
                    innov, S = filt.update(z, meas.h(xh_lvlh), H, meas.R)
                nis_list.append(float(innov @ np.linalg.solve(S, innov)))

            # NEES against truth in the native frame
            e_nat = filt.x - truth_native()
            nees_list.append(float(e_nat @ np.linalg.solve(filt.P, e_nat)))

        # -- recording / divergence -------------------------------------------
        x_true_next = truth.rel_lvlh()
        x_ref_next = lvlh_from_roe(s_des, coe_next)
        err_pos = x_true_next[:3] - x_ref_next[:3]
        err_pos_all.append(err_pos)
        if record_history:
            hist["t"].append(t + cfg.dt)
            hist["err_pos"].append(err_pos)
            xh = (filt.x if not perfect_nav else truth_native())
            hist["err_hat"].append(fmodel.to_lvlh(xh, coe_next)[:3] - x_true_next[:3])
            hist["dv_cum"].append(dv_total)
            hist["sroe_err"].append(
                roe_from_lvlh(x_true_next, coe_next) - s_des)
        if not np.all(np.isfinite(x_true_next)) or np.linalg.norm(err_pos) > divergence_limit:
            diverged = True
            break

    # ---- summary -------------------------------------------------------------
    duration_s = (k + 1) * cfg.dt
    summary: dict[str, Any] = {
        **asdict(cfg),
        "dv_total": dv_total,
        "dv_per_year": annualized_dv(dv_total, duration_s),
        "diverged": bool(diverged),
        "steps_run": int(k + 1),
        "mean_nis": float(np.mean(nis_list)) if nis_list else np.nan,
        "mean_nees": float(np.mean(nees_list)) if nees_list else np.nan,
        "saturation_count": getattr(controller, "saturation_count", 0),
    }
    # formation-keeping stats over the post-warm-up window only, so the
    # commissioning transient does not pollute the tightness metric (the same
    # window is used for perfect-nav runs to keep comparisons fair)
    k_warm = min(int(cfg.ctrl_warmup_orbits * T_orbit / cfg.dt), max(len(err_pos_all) - 2, 0))
    summary.update(position_error_stats(np.array(err_pos_all[k_warm:])))
    return SimResult(summary=summary, history={k: np.array(v) for k, v in hist.items()} if record_history else None)
