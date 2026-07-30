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

## Filter-consistency diagnosis (2026-07-26 .. 2026-07-29)

The full-tier campaign exposed a filter-consistency problem and three
experiments pinned it down; keep these in mind when writing the paper.

1. **Symptom.** With the open-loop-calibrated `q_accel` (kgd 1e-14), median
   NIS at cdgps sigma = 5 mm is ~8.5e3 against an expected dim(z) = 3, and
   median delta-V *falls* as navigation degrades (spearman -1.00) — the
   Pareto reads backwards. Worst for the most accurate STM (smallest Q).
2. **Fix demonstrated.** `scripts/tune_q_closedloop.py` (closed-loop NIS
   matching) gives kgd q = 3.162e-10 (x3.2e4), ga q = 4.499e-12 (x5.6).
   N=100 rerun (`data/corrected_q`, Fig. 9): NIS 4-6 across the sweep, LQR
   dv and rms both rise with sigma (spearman +1.00); MPC dv flat while rms
   rises — MPC trades accuracy, not fuel.
3. **Not control feed-through.** With control disabled (warmup > duration),
   NIS at the original Q gets ~15x WORSE (8.5e3 -> 1.2e5): the mismatch is
   the STM's secular error, which grows with unconstrained drift; the
   control loop partially masks it. The post-predict `G @ (u dt)` control
   injection is not the driver.
4. **NEES is honest, not buggy.** After tuning, NIS is consistent but NEES
   remains 30-1500 (falling with sigma): the covariance is right in the
   measured subspace and still optimistic in the unobserved directions.
   The ES-EKF NEES/NIS diagnostics themselves are unit-tested on a linear
   system.
5. **Implication.** A white-noise Q cannot represent a secular model error:
   inflating Q buys consistency at the cost of closed-loop performance
   (noise feed-through), and the tuned kgd Q (3.2e-10) lands near ss's
   (5.5e-10), i.e. the deficit is largely common-mode rather than
   STM-fidelity-driven. The principled fix is a bias-augmented /
   Schmidt-Kalman formulation; frame this as a finding, not a flaw.

Orekit cross-validation (Table 2) re-run over the full 30-day duration:
relative position agreement 2.3 mm rms / 6.9 mm max; absolute agreement
9.1 m rms / 19.5 m max, consistent with the documented ~1e-9 gravity-
constant mismatch (quadratic in time). The acceptance criterion should be
stated on RELATIVE motion for this study; absolute drift is reported for
completeness. Artifact: `data/validation/orekit_crossval.csv`.
