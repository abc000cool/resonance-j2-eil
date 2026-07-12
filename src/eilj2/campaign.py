"""Monte-Carlo campaign driver (proposal Sec. 3.4).

A campaign YAML defines defaults (``base``), a Cartesian ``grid`` of swept
parameters, and seeding/trial counts. Each grid point is run for ``n_seeds``
trials and written to its own Parquet file, keyed by a stable hash of the
point's parameters — re-running a campaign skips completed points, so
interrupted campaigns resume for free.

Grid values may be scalars (swept into one SimConfig field, keyed by the
grid key) or dicts (merged into several fields at once — used to link
measurement kind with its sigma, or geometry family with its size).

Seeding: each trial's RNG derives from SeedSequence([seed_root, point_crc,
trial_index]) — deterministic, order-independent, recorded in the output.

CLI:  python -m eilj2.campaign CONFIG.yaml [--n-jobs N] [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import zlib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from .simulate import SimConfig, run_sim

_SIM_FIELDS = {f.name for f in fields(SimConfig)}


def _expand_grid(grid: dict) -> list[dict]:
    keys = sorted(grid.keys())
    value_lists = []
    for k in keys:
        vals = grid[k]
        if not isinstance(vals, list):
            vals = [vals]
        value_lists.append([(k, v) for v in vals])
    points = []
    for combo in itertools.product(*value_lists):
        p: dict = {}
        for k, v in combo:
            if isinstance(v, dict):
                p.update(v)
            else:
                p[k] = v
        points.append(p)
    return points


def _point_id(point: dict) -> str:
    parts = []
    for k in sorted(point.keys()):
        v = point[k]
        s = f"{v:g}" if isinstance(v, float) else str(v)
        parts.append(f"{k[:12]}-{s}")
    raw = "_".join(parts).replace(" ", "").replace("/", "-")
    crc = zlib.crc32(raw.encode()) & 0xFFFFFFFF
    return f"{raw[:120]}_{crc:08x}"


def _make_config(base: dict, point: dict, seed: int) -> SimConfig:
    merged = {**base, **point, "seed": seed}
    unknown = set(merged) - _SIM_FIELDS
    if unknown:
        raise KeyError(f"unknown SimConfig fields in campaign config: {sorted(unknown)}")
    return SimConfig(**merged)


def _trial_seed(seed_root: int, point: dict, trial: int) -> int:
    crc = zlib.crc32(_point_id(point).encode()) & 0xFFFFFFFF
    ss = np.random.SeedSequence([seed_root, crc, trial])
    return int(ss.generate_state(1)[0])


def _run_trial(base: dict, point: dict, seed: int) -> dict:
    try:
        res = run_sim(_make_config(base, point, seed))
        return res.summary
    except Exception as exc:  # record failures instead of killing the campaign
        return {**base, **point, "seed": seed, "diverged": True,
                "error": f"{type(exc).__name__}: {exc}", "dv_total": np.nan,
                "dv_per_year": np.nan, "rms_pos_err": np.nan}


def run_campaign(config_path: str | Path, out_dir: str | Path | None = None,
                 n_jobs: int = -1) -> Path:
    cfg = yaml.safe_load(Path(config_path).read_text())
    base: dict = cfg.get("base", {})
    grid: dict = cfg.get("grid", {})
    n_seeds: int = int(cfg.get("n_seeds", 10))
    seed_root: int = int(cfg.get("seed_root", 20260712))
    out = Path(out_dir or cfg.get("out_dir", "data/processed"))
    out.mkdir(parents=True, exist_ok=True)

    points = _expand_grid(grid)
    print(f"[campaign] {len(points)} grid points x {n_seeds} seeds "
          f"= {len(points) * n_seeds} trials -> {out}")

    t0 = time.time()
    for idx, point in enumerate(points):
        pid = _point_id(point)
        f_out = out / f"{pid}.parquet"
        if f_out.exists():
            print(f"[campaign] ({idx + 1}/{len(points)}) skip (exists): {pid}")
            continue
        seeds = [_trial_seed(seed_root, point, k) for k in range(n_seeds)]
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_run_trial)(base, point, s) for s in seeds
        )
        df = pd.DataFrame(rows)
        df.to_parquet(f_out, index=False)
        n_div = int(df["diverged"].sum()) if "diverged" in df else 0
        print(f"[campaign] ({idx + 1}/{len(points)}) {pid}: "
              f"{len(df)} trials, {n_div} diverged, "
              f"{time.time() - t0:.0f}s elapsed")
    return out


def load_results(data_dir: str | Path) -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet results under {data_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a Monte-Carlo campaign")
    ap.add_argument("config", help="campaign YAML")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    run_campaign(args.config, args.out, args.n_jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
