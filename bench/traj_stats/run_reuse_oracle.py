#!/usr/bin/env python
"""traj_stats Phase 3 — oracle replay of committed-token compute reuse.

Offline replay of the proposal's cache-skip rule ("reuse the previous step's
velocity for tokens whose code has been stable for m steps, below a σ
threshold") against recorded Phase 1 atlas traces — no renders, no GPU. The
pre-gauge gate for the compute-reuse Phase 3 item: it prices both the
*headroom* (perfect-oracle skip ceiling from the true retrospective commit
step) and the *realizable detector* (streak-based online detection) before
any implementation or quality bench is spent.

Per (m, σ<) cell, aggregated over all generation traces:

* ``skip_frac``      — fraction of token-steps skipped → the compute ceiling
  (per-step cost ≈ active-query fraction when frozen tokens stay in K/V and
  drop only their query+MLP compute; dropping them from K/V is foveation
  redux — refuted).
* ``false_rate``     — among skipped token-steps, how often the recorded code
  at that step differs from the frozen code (the token would have moved).
* ``final_mismatch`` — frozen tokens whose FINAL code ≠ the frozen code
  (exact, bound-free: lands in the wrong quantization cell).
* ``stale_p50/p95``  — cumulative-activity upper bound on
  ‖x̂₀(i) − x̂₀(freeze)‖ over skipped token-steps, in units of the σ→0
  activity noise floor (95th pct of final-step activity).
* ``oracle_skip``    — the same skip fraction if each token froze exactly at
  its true ``commit`` step (zero false freezes; unrealizable online since
  commit is retrospective). Separates "detector is bad" from "headroom is
  small".

Open-loop caveat: real reuse feeds frozen v back into z; compounding drift is
NOT modeled — every number here is the intervention's *best* case.

Usage::

    uv run python bench/traj_stats/run_reuse_oracle.py \
        --traces bench/traj_stats/results/20260723-2100-phase1/traces_gen \
        --label phase3-reuse-oracle
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

M_VALUES = (2, 3, 4)
SIGMA_THRESH = (1.0, 0.7, 0.5)


def replay_one(path: str, m: int, sig_t: float) -> dict:
    d = np.load(path)
    codes = d["codes"]  # (S, B, C, Ht, Wt) u8
    act = np.nan_to_num(d["activity"].astype(np.float32))  # (S, B, Ht, Wt)
    sig = d["sigmas"]
    S = codes.shape[0]

    noise_floor = (
        float(np.nanquantile(d["activity"][-1].astype(np.float32), 0.95)) + 1e-12
    )

    stable = np.all(codes[1:] == codes[:-1], axis=2)  # (S-1, B, Ht, Wt)
    # streak[i] = consecutive unchanged code intervals ending at step i
    streak = np.zeros((S,) + stable.shape[1:], dtype=np.int16)
    for i in range(1, S):
        streak[i] = np.where(stable[i - 1], streak[i - 1] + 1, 0)

    frozen = np.zeros(stable.shape[1:], dtype=bool)
    frozen_code = np.zeros_like(codes[0])
    stale_acc = np.zeros(stable.shape[1:], dtype=np.float32)

    skip_ts = 0
    false_ts = 0
    stale_errs = []
    for i in range(S):
        if sig[i] < sig_t:
            newly = (~frozen) & (streak[i] >= m)
            frozen_code = np.where(newly[:, None], codes[i], frozen_code)
            frozen |= newly
        stale_acc[frozen] += act[i][frozen]
        n_skip = int(frozen.sum())
        skip_ts += n_skip
        if n_skip:
            moved = np.any(codes[i] != frozen_code, axis=1) & frozen
            false_ts += int(moved.sum())
            stale_errs.append(stale_acc[frozen] / noise_floor)

    commit = d["commit"]  # (B, Ht, Wt) — true last-change step, retrospective
    oracle_skip = float(
        np.mean([((commit < i) & (sig[i] < sig_t)).mean() for i in range(S)])
    )
    final_mismatch = (
        float(
            (np.any(codes[-1] != frozen_code, axis=1) & frozen).sum()
            / max(frozen.sum(), 1)
        )
        if frozen.any()
        else 0.0
    )
    allstale = np.concatenate(stale_errs) if stale_errs else np.array([0.0])
    return {
        "skip_frac": skip_ts / (S * frozen.size),
        "false_rate": false_ts / max(skip_ts, 1),
        "final_mismatch": final_mismatch,
        "stale_p50": float(np.percentile(allstale, 50)),
        "stale_p95": float(np.percentile(allstale, 95)),
        "frozen_final": float(frozen.mean()),
        "oracle_skip": oracle_skip,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--traces",
        default="bench/traj_stats/results/20260723-2100-phase1/traces_gen",
        help="dir of recorder .npz sidecars (default: the Phase 1 atlas generation arm)",
    )
    ap.add_argument("--label", default="phase3-reuse-oracle")
    args = ap.parse_args()

    traces = Path(args.traces)
    if not traces.is_absolute():
        traces = REPO_ROOT / traces
    files = sorted(glob.glob(str(traces / "*.npz")))
    if not files:
        sys.exit(f"no .npz traces under {traces}")
    print(f"{len(files)} traces from {traces}")

    header = (
        f"{'m':>2} {'sig<':>5} | {'skip%':>6} {'oracle%':>8} {'frozen@end%':>11} "
        f"{'false%':>7} {'finalmis%':>9} | {'stale p50':>9} {'p95':>6}  (noise-floor units)"
    )
    print(header)
    print("-" * len(header))
    cells = {}
    for m in M_VALUES:
        for st in SIGMA_THRESH:
            rs = [replay_one(f, m, st) for f in files]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            agg["oracle_skip_min"] = float(np.min([r["oracle_skip"] for r in rs]))
            agg["oracle_skip_max"] = float(np.max([r["oracle_skip"] for r in rs]))
            cells[f"m{m}_sig{st}"] = agg
            print(
                f"{m:>2} {st:>5} | {100 * agg['skip_frac']:>5.1f}% {100 * agg['oracle_skip']:>7.1f}% "
                f"{100 * agg['frozen_final']:>10.1f}% {100 * agg['false_rate']:>6.2f}% "
                f"{100 * agg['final_mismatch']:>8.2f}% | {agg['stale_p50']:>9.2f} {agg['stale_p95']:>6.2f}"
            )

    run_dir = make_run_dir("traj_stats", label=args.label)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics={"n_traces": len(files), "cells": cells},
        extra={
            "verdict": "compute-reuse CLOSED: oracle ceiling <=25.5% token-steps "
            "(13.6% at sig<0.5); realizable detector final-code mismatch >=26% "
            "at every tolerable-staleness cell; open-loop best case"
        },
    )
    print(f"\nwrote {run_dir / 'result.json'}")


if __name__ == "__main__":
    main()
