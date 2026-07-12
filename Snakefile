# Figure-regeneration DAG (proposal Sec. 3.5): configs -> simulations ->
# parquet -> figures. On Windows use scripts/make_all.py instead (same DAG,
# plain Python).
#
#   snakemake -c8 figures        # everything up to the figures
#   snakemake -c8 fig2           # a single figure and its inputs

FIGDIR = "paper/figures/generated"

rule all:
    input: f"{FIGDIR}/fig1_geometry.pdf", f"{FIGDIR}/fig2_validation.pdf", \
           f"{FIGDIR}/fig3_drift_rates.pdf", f"{FIGDIR}/fig4_drift_contour.pdf", \
           f"{FIGDIR}/fig5_perfect_pareto.pdf", f"{FIGDIR}/fig6_dv_vs_sigma.pdf", \
           f"{FIGDIR}/fig7_pareto_surface.pdf", f"{FIGDIR}/fig8_ablation.pdf"

rule figures:
    input: rules.all.input

# ---------------- simulation stages ----------------

rule validation_data:
    output: "data/validation/model_vs_truth.parquet"
    shell: "python scripts/run_validation.py"

rule drift_data:
    output: "data/drift/drift_rates.parquet", "data/drift/contour.parquet"
    shell: "python scripts/run_drift.py"

rule perfect_state_data:
    output: "data/perfect_state/pareto_stm.parquet"
    shell: "python scripts/run_perfect_state.py --truth stm"

rule screening_campaign:
    output: directory("data/screening")
    shell: "python -m eilj2.campaign config/screening.yaml"

rule ablation_campaign:
    output: directory("data/ablation")
    shell: "python -m eilj2.campaign config/ablation.yaml"

# ---------------- figures ----------------

rule fig1:
    output: f"{FIGDIR}/fig1_geometry.pdf"
    shell: "python -m eilj2.figures.fig1_geometry"

rule fig2:
    input: "data/validation/model_vs_truth.parquet"
    output: f"{FIGDIR}/fig2_validation.pdf"
    shell: "python -m eilj2.figures.fig2_validation"

rule fig3:
    input: "data/drift/drift_rates.parquet"
    output: f"{FIGDIR}/fig3_drift_rates.pdf"
    shell: "python -m eilj2.figures.fig3_drift_rates"

rule fig4:
    input: "data/drift/contour.parquet"
    output: f"{FIGDIR}/fig4_drift_contour.pdf"
    shell: "python -m eilj2.figures.fig4_drift_contour"

rule fig5:
    input: "data/perfect_state/pareto_stm.parquet"
    output: f"{FIGDIR}/fig5_perfect_pareto.pdf"
    shell: "python -m eilj2.figures.fig5_perfect_pareto"

rule fig6:
    input: "data/screening"
    output: f"{FIGDIR}/fig6_dv_vs_sigma.pdf"
    shell: "python -m eilj2.figures.fig6_dv_vs_sigma"

rule fig7:
    input: "data/screening"
    output: f"{FIGDIR}/fig7_pareto_surface.pdf"
    shell: "python -m eilj2.figures.fig7_pareto_surface"

rule fig8:
    input: "data/ablation"
    output: f"{FIGDIR}/fig8_ablation.pdf"
    shell: "python -m eilj2.figures.fig8_ablation"
