"""Formation-keeping controllers (proposal Sec. 1.7).

Two controller kinds share the closed-loop engine:

- ``continuous``: implements ``accel(t, x_hat_lvlh, x_ref_lvlh) -> u`` — an
  LVLH acceleration command held (ZOH) over the next step (LQR).
- ``impulsive``: implements ``plan(t, sroe_hat, sroe_des, coe_c_mean) ->
  list[(t_burn, dv_lvlh)]`` — a schedule of delta-v impulses computed at each
  planning epoch from the current ROE estimate (impulsive Chernick-D'Amico,
  receding-horizon MPC).

All delta-v components are in chief LVLH axes, which coincide with RTN
(R = x radial, T = y along-track, N = z cross-track).
"""

from __future__ import annotations

from .lqr import LQRController

__all__ = ["LQRController"]
