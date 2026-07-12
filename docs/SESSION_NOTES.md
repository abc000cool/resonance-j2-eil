# Build-session notes (2026-07-12)

`session-chat-log.jsonl` is the raw Claude Code session transcript (JSONL,
one event per line — user messages, assistant messages, tool calls and
results) of the session in which this entire repository was designed, built,
tested, and first run. It is committed for provenance: every implementation
decision, verified equation source, and debugging step is traceable in it.

## Session summary

1. **Scope decisions** (user-approved): pure-Python J2–J4 truth + Orekit
   cross-validation instead of Basilisk; tiered screening→full Monte-Carlo
   campaign; full 8-week proposal scope (3 controllers × 4 STMs × 3
   measurement architectures, EKF+UKF, both ablations); complete
   reproducibility stack.
2. **Equation provenance**: four research agents fetched and visually
   verified primary sources — Schweighart MIT thesis (2001), Koenig–
   Guffanti–D'Amico JGCD 2017 (Eq. A6 read at 400 dpi from the author PDF),
   Chernick & D'Amico AIAA 2016-5659 + Stanford thesis, and the Brouwer
   mean↔osculating map from Schaub's Basilisk reference code (ISC).
3. **Bugs caught by the verification tests** (all fixed):
   - two transcription errors in secondary-source KGD STM coefficients
     (missing E = 1+η and F = 4+3η factors) — caught by the
     numerical-Jacobian cross-check;
   - Brouwer osc→mean fixed-point collapsing e to 0 at small eccentricity —
     fixed by iterating in nonsingular variables;
   - MPC with one impulse slot per orbit losing out-of-plane
     controllability (all burns at the same argument of latitude) — fixed
     with 4 slots/orbit;
   - saturation-induced runaway of certainty-equivalent LQR under large
     initial δa dispersion — mitigated by realistic init dispersion and
     u_max; noted as paper-discussion material.
4. **First results** (committed under `data/` and
   `paper/figures/generated/`): model validation (KGD 17–26 m < GA ~100 m
   ≪ CW/S-S km-scale over 30 days), perfect-state controller Pareto
   (impulsive 28 m/s/yr loose, MPC ~1 m tight, LQR between), Q calibration
   ladder, 1-day Orekit agreement 0.69 m absolute / 0.1 mm relative.

## Resuming on another machine

```
git clone <this repo> && cd resonance-j2-eil
uv sync --all-extras
uv run pytest -q          # should be all green
```
Then continue from the README's phase list (screening campaign onward).
Progress at the time of this commit: Phases A–D complete (validation, drift
maps, perfect-state Pareto with numerical truth); next step is the screening
campaign (`uv run python -m eilj2.campaign config/screening.yaml`).
