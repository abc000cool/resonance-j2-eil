"""Analytic relative-motion models (STM providers).

Every model implements the `RelativeMotionModel` interface:

- ``frame``: the model's native state representation — ``"lvlh"`` (Cartesian
  relative state [m, m/s] in the chief LVLH frame) or ``"roe"`` (quasi-
  nonsingular ROE scaled by the chief semi-major axis, i.e. a_c * droe, in
  meters for all six components).
- ``stm(coe_c_mean, dt)``: 6x6 state transition matrix propagating the native
  state over dt, evaluated at the chief MEAN elements at segment start.
- ``to_lvlh(state, coe_c_mean)`` / ``from_lvlh(x, coe_c_mean)``: exact-at-
  first-order linear maps between the native state and the LVLH Cartesian
  state (identity for LVLH-native models).
- ``control_input(coe_c_mean, dt)``: 6x3 matrix mapping an impulsive delta-v
  [m/s] in LVLH axes applied at segment START to the native-state change at
  segment END (i.e. Phi @ B_impulse). For ZOH accelerations use
  ``accel_input`` where provided.

The factory `get_model(name, ...)` returns instances by short name:
"cw", "ss", "gim_alfriend" ("ga"), "kgd".
"""

from __future__ import annotations


def get_model(name: str, **kwargs):
    key = name.lower().replace("-", "_")
    if key == "cw":
        from .cw import CWModel

        return CWModel(**kwargs)
    if key in ("ss", "schweighart", "schweighart_sedwick"):
        from .schweighart import SchweighartSedwickModel

        return SchweighartSedwickModel(**kwargs)
    if key in ("ga", "gim_alfriend"):
        from .gim_alfriend import GimAlfriendModel

        return GimAlfriendModel(**kwargs)
    if key == "kgd":
        from .kgd import KGDModel

        return KGDModel(**kwargs)
    raise ValueError(f"unknown relative-motion model: {name!r}")
