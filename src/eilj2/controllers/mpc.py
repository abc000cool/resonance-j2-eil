"""Receding-horizon MPC on the KGD ROE plant (CasADi + IPOPT).

Proposal Sec. 1.7 / Phase D: horizon of `horizon_orbits` orbits with
`impulses_per_orbit` delta-v opportunities per orbit; dynamics
s_{k+1} = Phi_k (s_k + Gamma_k dv_k) with Phi from the KGD J2 secular STM
and Gamma from the near-circular GVE. Multiple burn slots per orbit are
essential: with one slot per orbit every impulse occurs at the same argument
of latitude, and the out-of-plane channel is controllable only along a
single fixed direction of the relative-inclination plane.

The cost trades total delta-v against ROE tracking error; the tightness
Pareto is traced by sweeping state_weight/dv_weight. Each planning epoch
solves the program and applies only the first `plan_interval_orbits` worth
of maneuvers (receding horizon).
"""

from __future__ import annotations

import numpy as np

from ..elements import propagate_mean
from ..roe_map import gve_control_matrix
from ..stm.kgd import arg_lat_rate


class MPCController:
    kind = "impulsive"

    def __init__(self, model, horizon_orbits: int = 10,
                 plan_interval_orbits: float = 1.0,
                 dv_weight: float = 1.0, state_weight: float = 1e-3,
                 dv_max: float | None = None, impulses_per_orbit: int = 4):
        import casadi  # deferred: fail loudly only if MPC is actually used

        self._ca = casadi
        self.model = model
        self.H = int(horizon_orbits)
        self.ipo = int(impulses_per_orbit)
        self.plan_interval_orbits = float(plan_interval_orbits)
        self.dv_weight = float(dv_weight)
        self.state_weight = float(state_weight)
        self.dv_max = dv_max
        self.solve_failures = 0

    def plan(self, t: float, s_hat: np.ndarray, s_des: np.ndarray,
             coe_c_mean: np.ndarray) -> list[tuple[float, np.ndarray]]:
        ca = self._ca
        T_orbit = 2.0 * np.pi / arg_lat_rate(coe_c_mean)
        T_seg = T_orbit / self.ipo
        n_seg = self.H * self.ipo

        # precompute segment matrices along the mean chief trajectory
        Phis, Gs = [], []
        coe_k = coe_c_mean.copy()
        for _ in range(n_seg):
            Phis.append(self.model.stm(coe_k, T_seg))
            Gs.append(gve_control_matrix(coe_k))
            coe_k = propagate_mean(coe_k, T_seg)

        opti = ca.Opti()
        dv = opti.variable(3, n_seg)
        s = ca.DM(s_hat)
        cost = 0
        for k in range(n_seg):
            s = ca.mtimes(ca.DM(Phis[k]), s + ca.mtimes(ca.DM(Gs[k]), dv[:, k]))
            cost = cost + self.dv_weight * ca.sumsqr(dv[:, k]) \
                + (self.state_weight / self.ipo) * ca.sumsqr(s - ca.DM(s_des))
            if self.dv_max is not None:
                opti.subject_to(opti.bounded(-self.dv_max, dv[:, k], self.dv_max))
        cost = cost + 10.0 * self.state_weight * ca.sumsqr(s - ca.DM(s_des))
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": 0},
                    {"print_level": 0, "sb": "yes", "max_iter": 300})
        try:
            sol = opti.solve()
            dv_opt = np.asarray(sol.value(dv)).reshape(3, n_seg)
        except RuntimeError:
            self.solve_failures += 1
            return []

        n_apply = max(1, int(round(self.plan_interval_orbits * self.ipo)))
        burns = []
        for k in range(min(n_apply, n_seg)):
            v = dv_opt[:, k]
            if np.linalg.norm(v) > 1e-9:
                burns.append((t + k * T_seg, v.copy()))
        return burns
