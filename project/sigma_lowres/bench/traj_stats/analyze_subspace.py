#!/usr/bin/env python
"""traj_stats Phase 1 addendum — channel *subspace* concentration + boundary
predictability (offline; no GPU, reads existing traces + cached latents).

Two questions the marginal statistics of the atlas couldn't answer:

1. **Subspace**: the atlas measured per-channel (axis-aligned) entropy skew.
   Here: the channel *covariance* spectrum of token-level normalized x̂₀ —
   effective rank (participation ratio) of the 16-dim channel space, for the
   static corpus, generated final x̂₀, and mid-trajectory (σ≈0.5) — plus the
   principal angles between their top-8 eigenspaces (is it the SAME
   subspace?).
2. **Boundary predictability**: can you predict *which* tokens commit late
   from static covariates (final hf = spatial detail, early guide = prompt
   pressure)? Per-trace Spearman of commit-knot vs each covariate.

Usage::

    uv run python project/sigma_lowres/bench/traj_stats/analyze_subspace.py
    uv run python project/sigma_lowres/bench/traj_stats/analyze_subspace.py --out subspace.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from library.inference.traj_stats import (  # noqa: E402
    QWEN_LATENTS_MEAN,
    QWEN_LATENTS_STD,
)

DEFAULT_TOK_TRACES = (
    "project/sigma_lowres/bench/traj_stats/results/20260723-2120-phase2/traces_baseline"
)
DEFAULT_GEN_TRACES = (
    "project/sigma_lowres/bench/traj_stats/results/20260723-2100-phase1/traces_gen"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--static_sample", type=int, default=120)
    p.add_argument(
        "--tok_traces",
        default=DEFAULT_TOK_TRACES,
        help="trace dir WITH the tok array (post-Phase-2 sidecars)",
    )
    p.add_argument(
        "--gen_traces",
        default=DEFAULT_GEN_TRACES,
        help="generation traces for the commit-predictability part",
    )
    p.add_argument("--out", default=None)
    return p.parse_args()


def _static_tokens(n: int) -> np.ndarray:
    mean = np.asarray(QWEN_LATENTS_MEAN, np.float32)[:, None, None]
    std = np.asarray(QWEN_LATENTS_STD, np.float32)[:, None, None]
    paths = sorted((REPO_ROOT / "post_image_dataset/lora").glob("*/*_anima.npz"))
    rng = random.Random(0)
    rng.shuffle(paths)
    toks = []
    for p in paths[:n]:
        d = np.load(p)
        key = next(k for k in d.files if k.startswith("latents_"))
        lat = d[key].astype(np.float32)
        c, h, w = lat.shape
        h -= h % 2
        w -= w % 2
        xn = (lat[:, :h, :w] - mean) / std
        toks.append(
            xn.reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4)).reshape(c, -1)
        )
    return np.concatenate(toks, axis=1)


def _cov(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return (x @ x.T) / x.shape[1]


def _spectrum(cov: np.ndarray) -> dict:
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    cum = np.cumsum(ev) / ev.sum()
    return {
        "effective_rank": round(float(ev.sum() ** 2 / (ev**2).sum()), 3),
        "top4_var": round(float(cum[3]), 4),
        "top8_var": round(float(cum[7]), 4),
        "eigenvalues": np.round(ev, 4).tolist(),
    }


def _top_eigvecs(cov: np.ndarray, k: int = 8) -> np.ndarray:
    w, v = np.linalg.eigh(cov)
    return v[:, np.argsort(w)[::-1][:k]]


def _traj_tokens(traces_dir: str, at_sigma: float | None) -> np.ndarray:
    """Token-level x̂₀ from tok-carrying traces; final knot if at_sigma None."""
    cols = []
    for f in sorted(glob.glob(str(REPO_ROOT / traces_dir / "*.npz"))):
        d = np.load(f)
        if "tok" not in d.files:
            raise SystemExit(f"{f} has no tok array (pre-Phase-2 sidecar)")
        order = np.argsort(-d["sigmas"])
        tok = d["tok"].astype(np.float32)[order]
        j = (
            tok.shape[0] - 1
            if at_sigma is None
            else int(np.argmin(np.abs(d["sigmas"][order] - at_sigma)))
        )
        cols.append(tok[j].reshape(tok.shape[2], -1))
    if not cols:
        raise SystemExit(f"no traces in {traces_dir}")
    return np.concatenate(cols, axis=1)


def commit_predictability(traces_dir: str) -> dict:
    rows = []
    for f in sorted(glob.glob(str(REPO_ROOT / traces_dir / "*.npz"))):
        d = np.load(f)
        order = np.argsort(-d["sigmas"])
        codes = d["codes"][order]
        s = codes.shape[0]
        changed = (codes[1:] != codes[:-1]).any(axis=2)
        commit = (
            (changed * np.arange(1, s)[:, None, None, None])
            .max(axis=0)
            .ravel()
            .astype(np.float32)
        )
        hf_fin = d["hf"].astype(np.float32)[order][-1].ravel()
        guide_early = np.nanmean(
            d["guide"].astype(np.float32)[order][:6], axis=0
        ).ravel()
        rows.append(
            (
                sstats.spearmanr(commit, hf_fin).statistic,
                sstats.spearmanr(commit, guide_early).statistic,
            )
        )
    r = np.asarray(rows)
    return {
        "n_traces": len(rows),
        "commit_vs_final_hf_spearman": {
            "mean": round(float(r[:, 0].mean()), 3),
            "min": round(float(r[:, 0].min()), 3),
            "max": round(float(r[:, 0].max()), 3),
        },
        "commit_vs_early_guide_spearman": {
            "mean": round(float(np.nanmean(r[:, 1])), 3),
            "min": round(float(np.nanmin(r[:, 1])), 3),
            "max": round(float(np.nanmax(r[:, 1])), 3),
        },
    }


def main() -> None:
    args = parse_args()

    xs = _static_tokens(args.static_sample)
    cov_s = _cov(xs)
    out = {"static": {"n_tokens": int(xs.shape[1]), **_spectrum(cov_s)}}

    for name, at in (("generated_final", None), ("generated_mid_sigma0.5", 0.5)):
        xg = _traj_tokens(args.tok_traces, at)
        cov_g = _cov(xg)
        angles = np.linalg.svd(
            _top_eigvecs(cov_s).T @ _top_eigvecs(cov_g), compute_uv=False
        )
        out[name] = {
            "n_tokens": int(xg.shape[1]),
            **_spectrum(cov_g),
            "top8_alignment_to_static": np.round(angles, 3).tolist(),
        }

    out["commit_predictability"] = commit_predictability(args.gen_traces)

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
