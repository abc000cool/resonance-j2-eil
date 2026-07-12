"""Shared figure style and IO helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "paper" / "figures" / "generated"

MODEL_LABELS = {"cw": "CW", "ss": "S–S", "gim_alfriend": "GA", "kgd": "KGD"}
CTRL_LABELS = {"lqr": "LQR", "lqr_cov": "LQR (cov-aware)", "mpc": "MPC",
               "impulsive": "Chernick–D'Amico"}
MEAS_LABELS = {"cdgps": "CDGPS", "rf": "RF range", "angles": "angles-only"}

# colorblind-safe (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#F0E442", "#000000"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    print(f"[fig] wrote {OUT / name}.pdf/.png")


def need(path: Path, hint: str):
    if not Path(path).exists():
        raise SystemExit(f"[fig] missing {path} — generate it first with: {hint}")
