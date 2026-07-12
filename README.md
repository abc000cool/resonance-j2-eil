# eilj2 — Estimation-in-the-loop J2 formation-keeping

Reference implementation for the preprint **"Estimation-in-the-loop J2
formation-keeping: a Pareto characterization of relative-navigation accuracy
versus delta-V cost"** (The Resonance Foundation, target: arXiv).

The study extends Koenig & D'Amico (JGCD 2018) along five axes: three
controllers (LQR, MPC, closed-form impulsive Chernick–D'Amico) × three
relative-motion STMs (Schweighart–Sedwick, Gim–Alfriend, Koenig–Guffanti–
D'Amico) × three measurement architectures (CDGPS, RF range, angles-only)
with an **actual error-state EKF/UKF running in the closed loop**, swept over
navigation accuracy to produce a controller-and-plant-conditional Pareto
surface of J2 formation-keeping ΔV.

## What is implemented

| Component | Module | Provenance |
|---|---|---|
| J2–J4 zonal gravity + potential | `eilj2.gravity` | Vallado 4th ed. §8.7; self-consistency-tested (a = ∇U) |
| Elements, Kepler, ROE | `eilj2.elements` | standard; round-trip tested |
| Brouwer mean↔osculating (1st-order J2) | `eilj2.brouwer` | Schaub & Junkins App. F via the Basilisk reference implementation (ISC); nonsingular fixed-point inversion |
| Two-sat numerical truth (DOP853 + fast RK4) | `eilj2.truth` | RK4 verified vs DOP853 to 3 µm relative over an orbit |
| CW STM | `eilj2.stm.cw` | closed form; ODE + truth tested |
| Schweighart–Sedwick LTI plant | `eilj2.stm.schweighart` | MIT thesis (2001) + Bevilacqua et al. (2010) verified coefficients |
| Gim–Alfriend geometric STM | `eilj2.stm.gim_alfriend` | numerically-linearized A·Σ·B composition (see note below) |
| KGD qns-ROE J2 STM (arbitrary e) | `eilj2.stm.kgd` | JGCD 2017 Eq. (A6), visually verified from the author PDF; cross-checked against a numerical secular Jacobian |
| ROE↔LVLH map + GVE input matrix | `eilj2.roe_map` | D'Amico thesis Eq. (2.17); Chernick 2016 Eq. (10) |
| CDGPS / RF range / angles-only models | `eilj2.measurements` | proposal §1.8 accuracy classes |
| ES-EKF (Joseph form) + UKF ablation | `eilj2.filters` | NEES/NIS consistency-tested |
| LQR (+ covariance-aware variant) | `eilj2.controllers.lqr` | scipy CARE |
| Receding-horizon MPC | `eilj2.controllers.mpc` | CasADi + IPOPT on the KGD plant |
| Impulsive Chernick–D'Amico | `eilj2.controllers.impulsive` | AIAA 2016-5659 / JGCD 2018 closed forms, pseudo-state targeting, J2-refined out-of-plane location |
| Closed-loop engine + tiered MC campaign | `eilj2.simulate`, `eilj2.campaign` | Parquet outputs, deterministic seeds, resume |

**Deliberate deviations from the original proposal** (all discussed with the PI):

1. **Truth propagator**: pure-Python J2–J4 numerical truth with an
   **Orekit cross-validation** (`scripts/crossvalidate_orekit.py`) instead of
   Basilisk — Basilisk is not reliably pip-installable on Windows, and for a
   J2-isolation study a zonal-only numerical truth is physically equivalent.
2. **Gim–Alfriend STM**: implemented as the *numerically-linearized* Jacobian
   composition Φ = A(t₁)Σ(t₁,t₀)A(t₀)⁻¹ of the exact nonlinear maps
   (Brouwer + element geometry) rather than a transcription of the closed-form
   entries — mathematically the same first-order-J2 construction, valid for
   arbitrary eccentricity, and far less error-prone (differences are O(J2²)
   + finite-difference noise ~1e-6).
3. Two equation transcription errors in third-party reproductions were caught
   by the numerical-Jacobian cross-checks and corrected against the primary
   sources (KGD Φ[δλ,δa] and Φ[δλ,δix] E/F factors) — see
   `src/eilj2/stm/kgd.py` docstring.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via
`.python-version` (uv downloads it automatically).

```powershell
uv sync --all-extras     # core + Orekit validation extra
uv run pytest -q         # 60+ tests, ~10 s (add -m slow for the 30-orbit truth validations)
```

## Running the study (phase by phase)

Every stage writes Parquet under `data/` and is **resumable** (completed
outputs are skipped; delete them to force re-runs). On Linux/macOS you can
drive everything with `snakemake -c8 figures`; on Windows use the plain
Python driver:

```powershell
uv run python scripts/make_all.py --phase all      # or one phase at a time
```

### Phase B — model validation (Fig. 2) — ~15 min
```powershell
uv run python scripts/run_validation.py
uv run python -m eilj2.figures.fig2_validation
```
Expected ordering of 30-day RMS error: CW ≫ S-S > GA ≈ KGD.

### Phase C — drift maps (Figs. 3–4) — ~30–60 min on 8 cores
```powershell
uv run python scripts/run_drift.py
uv run python -m eilj2.figures.fig3_drift_rates
uv run python -m eilj2.figures.fig4_drift_contour
```

### Phase D — perfect-state Pareto (Fig. 5) — ~20 min (STM truth)
```powershell
uv run python scripts/run_perfect_state.py --truth stm
uv run python -m eilj2.figures.fig5_perfect_pareto
# paper-final: rerun with --truth numerical (hours)
```

### Phase E — the estimation-in-the-loop campaign (Figs. 6–7)
1. Calibrate the filter process noise once:
   ```powershell
   uv run python scripts/calibrate_q.py
   ```
   and put the printed `q_accel` values into the campaign YAMLs.
2. **Screening tier** (STM-surrogate truth, 648 points × 20 seeds; overnight
   on a modern 8-core laptop — KGD/SS points are ~20 s/trial, GA points
   ~6 min/trial):
   ```powershell
   uv run python -m eilj2.campaign config/screening.yaml
   uv run python scripts/find_knees.py data/screening      # Table 3
   uv run python -m eilj2.figures.fig6_dv_vs_sigma
   uv run python -m eilj2.figures.fig7_pareto_surface
   ```
3. **Full tier** (numerical truth, N = 100): *prune* `config/full.yaml` to
   the knee-bracketing sigma points found by screening (instructions in the
   file header), then:
   ```powershell
   uv run python -m eilj2.campaign config/full.yaml
   ```
   Budget ~3–6 min per trial; size the pruned grid to your machine.

### Phase F — ablations (Fig. 8) — overnight (numerical truth)
```powershell
uv run python -m eilj2.campaign config/ablation.yaml
uv run python -m eilj2.figures.fig8_ablation
```

### Cross-validation table (Table 2)
```powershell
uv run python scripts/crossvalidate_orekit.py    # downloads orekit-data on first run
```

### Phase G — paper
`paper/main.tex` contains the full skeleton (abstract, section outlines,
figure hooks that pick up `paper/figures/generated/*.pdf`, `references.bib`
with the anchor bibliography — grep `TODO-verify` before submission).
Build with `latexmk -pdf main.tex` inside `paper/`.

## Repository layout

```
config/          campaign YAMLs (screening / full / ablation / ci_mini)
src/eilj2/       the package (stm/, measurements/, filters/, controllers/, figures/)
scripts/         phase entrypoints + make_all.py driver
tests/           64+ unit/validation tests (pytest; -m slow for truth comparisons)
paper/           LaTeX skeleton + references + generated figures
data/            simulation outputs (gitignored, regenerable)
Snakefile        figure DAG (Linux/macOS); scripts/make_all.py mirrors it
```

## Reproducibility

- `uv.lock` pins the environment; `.python-version` pins CPython 3.12.
- Seeds: every trial derives from `SeedSequence([seed_root, crc32(point), trial])`
  — deterministic, order-independent, recorded in the output Parquet.
- CI (`.github/workflows/ci.yml`): 3.11–3.13 × Linux/macOS/Windows, fast +
  slow tests, an end-to-end mini campaign, and a figure smoke test.
- `Dockerfile` for a hardened Linux build.
- Publishing: push to GitHub → enable Zenodo integration → tag
  `arxiv-submission-v1` (mints the DOI) → update `CITATION.cff` with the DOI
  → submit `paper/` to arXiv with the code/data DOI in the abstract.

## License

MIT. The Brouwer mean↔osculating map is ported from the AVS Lab Basilisk
reference implementation (ISC license, © 2016 Autonomous Vehicle Systems
Lab, University of Colorado Boulder), which is ISC-compatible with MIT.
