#!/usr/bin/env python
"""traj_stats Phase 2 — the trajectory-intactness gauge.

For a candidate intervention X and a baseline, both recorded with
``--traj_stats`` at matched (prompt, seed, steps, size), compute

    D(X) = per-σ divergence profile between X's traces and baseline's traces

reported as *curves*, never one scalar (``_archive/proposals/traj_latent_stats.md``
Phase 2). The design premise: a process-breaking intervention (foveation) can
look fine on the final image while its *trace* flatlines — the gauge must
rediscover that from traces alone.

Per matched pair, per σ-knot (σ-descending canonical order):

* ``x0_rmse``       — token-wise RMSE of the raw normalized x̂₀ maps (``tok``)
* ``code_mismatch`` — fraction of tokens whose k-bit code differs
* ``dE``            — E_X(σ) − E_base(σ), τ from the *baseline* activity floor
* ``hf_ratio``      — mean hf_X / mean hf_base (the trace that flatlined in P4t)
* ``commit_shift``  — commit-CDF_X(σ) − commit-CDF_base(σ)

plus two structure scalars at the σ→0 end (final knot):

* ``hf_flatline_frac``     — fraction of tokens with hf_X < 0.5·hf_base
* ``flatline_clustering``  — P(4-neighbor also flatlined | token flatlined)
  divided by the flatline base rate. ≈1 for spatially random deficits,
  ≫1 for a contiguous region (foveation's periphery).

Verdict bands (**bench-calibrated, provisional** — set by
``run_gauge_calibration.py``; re-tune there, not here). Calibration put one
exemplar in each band:

* ``intact``          — divergence within the process-transparent envelope
  (exemplar: SMC-CFG)
* ``perturbed``       — trace moved beyond it, but no structural damage
  (exemplar: Spectrum — its forecast steps are process-visible even though
  its output is quality-neutral by its own bench)
* ``process-broken``  — structural damage: a commit-CDF hole, an hf blow-up
  (≥3 knots at ≥3× — foveation's blocky group-shared periphery), or a
  spatially-clustered hf flatline (exemplar: foveation σ_c=0.75)

Usage::

    uv run python project/sigma_lowres/bench/traj_stats/gauge.py --baseline <dir> --candidate <dir>
    uv run python project/sigma_lowres/bench/traj_stats/gauge.py ... --out gauge.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Verdict thresholds — CALIBRATED by run_gauge_calibration.py
# (results/20260723-2120-phase2/), provisional until a second calibration run
# disagrees. Calibration exemplars, one per band: SMC-CFG = "intact"
# (process-transparent), Spectrum = "perturbed" (quality-neutral but its
# forecast steps are process-VISIBLE — dE dips to −0.65 in the forecast
# band; quality-neutral ≠ process-transparent), foveation σ_c=0.75 =
# "process-broken" (commit-CDF hole 0.17 + hf blow-up ~30×).
# ---------------------------------------------------------------------------
#
# The verdict is driven by DISTRIBUTIONAL metrics only (E, hf-ratio,
# commit-CDF): any intervention that touches the combine (SMC) or forecasts
# steps (Spectrum) produces chaotic *pointwise* trajectory divergence —
# large x0_rmse / code_mismatch — while leaving the process statistics
# intact. Pointwise curves are reported as descriptive context, never
# verdict inputs; P4t itself was a distributional signature (periphery hf
# flatline + commit hole), which is what "process-broken" tests for.
BANDS = {
    # structural damage ⇒ process-broken
    "hf_flatline_frac_broken": 0.20,  # ≥20 % of tokens hf-flatlined ...
    "flatline_clustering_broken": 2.0,  # ... AND spatially contiguous
    "commit_shift_broken": 0.15,  # |commit-CDF hole| anywhere
    # hf blow-up: |hf_ratio − 1| ≥ dev on ≥ knots consecutive-free count.
    # Foveation's group-shared periphery is piecewise-constant → cell
    # boundaries dump Laplacian energy (measured ~30× at σ_c; the *flatline*
    # only exists in the final image after the bicubic readout). Calibration:
    # foveation 14+ qualifying knots, Spectrum 0 (max dev 0.34).
    "hf_ratio_dev_broken": 2.0,
    "hf_blowup_knots_broken": 3,
    # distributional perturbation envelope (any exceeded ⇒ perturbed)
    "dE_intact": 0.15,  # max |dE| over knots
    "commit_shift_intact": 0.08,  # max |Δcdf| over knots
    "hf_ratio_dev_intact": 0.30,  # max |hf_ratio − 1| over knots
}


def _load_canonical(npz_path: str) -> dict:
    """Load one trace, canonicalized to σ-descending knot order."""
    d = np.load(npz_path)
    sig = d["sigmas"]
    order = np.argsort(-sig, kind="stable")
    out = {
        "path": npz_path,
        "sigmas": sig[order],
        "activity": d["activity"].astype(np.float32)[order],
        "hf": d["hf"].astype(np.float32)[order],
        "codes": d["codes"][order],
        "header": json.loads(str(d["header_json"])),
    }
    out["tok"] = d["tok"].astype(np.float32)[order] if "tok" in d.files else None
    return out


def _commit_cdf(codes: np.ndarray) -> np.ndarray:
    """commit-CDF from stored per-step codes (σ-descending order)."""
    s = codes.shape[0]
    if s < 2:
        return np.ones(s)
    changed = (codes[1:] != codes[:-1]).any(axis=2)  # (S-1, B, Ht, Wt)
    idx = np.arange(1, s, dtype=np.int32)[:, None, None, None]
    commit = (changed * idx).max(axis=0)
    n = commit.size
    return np.asarray([(commit <= i).sum() / n for i in range(s)])


def _activity_floor_tau(trace: dict, q: float = 0.95) -> float:
    """τ from the smallest-σ knot with real activity values."""
    act = trace["activity"]
    for j in range(act.shape[0] - 1, -1, -1):
        if not np.all(np.isnan(act[j])):
            return float(np.nanquantile(act[j], q))
    return float("nan")


def _e_curve(trace: dict, tau: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean((trace["activity"] > tau).astype(np.float32), axis=(1, 2, 3))


def _clustering(flat: np.ndarray) -> float:
    """P(4-neighbor flatlined | token flatlined) / base rate. flat: (B,H,W) bool."""
    base = float(flat.mean())
    if base <= 0.0 or base >= 1.0:
        return 1.0
    agree = hits = 0.0
    for dy, dx in ((0, 1), (1, 0)):
        a = flat[:, : flat.shape[1] - dy, : flat.shape[2] - dx]
        b = flat[:, dy:, dx:]
        agree += float((a & b).sum())
        hits += float(a.sum())
    if hits == 0:
        return 1.0
    return (agree / hits) / base


def compare_pair(base: dict, cand: dict) -> dict:
    """One matched (baseline, candidate) pair → per-knot divergence profile."""
    assert np.allclose(base["sigmas"], cand["sigmas"], atol=1e-6), (
        f"σ-schedule mismatch: {base['path']} vs {cand['path']}"
    )
    s = base["sigmas"].shape[0]

    if base["tok"] is not None and cand["tok"] is not None:
        diff = cand["tok"] - base["tok"]
        x0_rmse = np.sqrt(np.mean(diff**2, axis=(1, 2, 3, 4)))
    else:  # pre-tok sidecars: coarse fallback via dequantized codes
        k = base["header"]["k"]
        width = 8.0 / (1 << k)
        dq = cand["codes"].astype(np.float32) - base["codes"].astype(np.float32)
        x0_rmse = np.sqrt(np.mean((dq * width) ** 2, axis=(1, 2, 3, 4)))

    code_mismatch = (cand["codes"] != base["codes"]).any(axis=2).mean(axis=(1, 2, 3))

    tau = _activity_floor_tau(base)
    d_e = _e_curve(cand, tau) - _e_curve(base, tau)

    with np.errstate(invalid="ignore"):
        hf_b = np.nanmean(base["hf"], axis=(1, 2, 3))
        hf_c = np.nanmean(cand["hf"], axis=(1, 2, 3))
    hf_ratio = hf_c / np.where(hf_b > 0, hf_b, np.nan)

    commit_shift = _commit_cdf(cand["codes"]) - _commit_cdf(base["codes"])

    # final-knot structural check: the P4t signature
    flat = cand["hf"][-1] < 0.5 * base["hf"][-1]  # (B, Ht, Wt) bool
    return {
        "sigmas": base["sigmas"],
        "x0_rmse": x0_rmse,
        "code_mismatch": code_mismatch,
        "dE": d_e,
        "hf_ratio": hf_ratio,
        "commit_shift": commit_shift,
        "hf_flatline_frac": float(flat.mean()),
        "flatline_clustering": _clustering(flat),
        "n_knots": s,
    }


def pair_traces(baseline_dir: str, candidate_dir: str) -> list[tuple[dict, dict]]:
    """Match traces across dirs by (prompt_hash, seed) from the npz header."""

    def index(d):
        out = {}
        for p in sorted(Path(d).glob("*.npz")):
            t = _load_canonical(str(p))
            key = (t["header"].get("prompt_hash"), t["header"].get("seed"))
            out.setdefault(key, []).append(t)
        return out

    bi, ci = index(baseline_dir), index(candidate_dir)
    pairs = []
    for key in sorted(set(bi) & set(ci), key=str):
        for b, c in zip(bi[key], ci[key]):
            pairs.append((b, c))
    unmatched = sorted(set(bi) ^ set(ci), key=str)
    if unmatched:
        print(
            f"gauge: {len(unmatched)} unmatched (prompt,seed) keys skipped",
            file=sys.stderr,
        )
    if not pairs:
        raise SystemExit("gauge: no matched (prompt_hash, seed) pairs")
    return pairs


def verdict_for(agg: dict, bands: dict = BANDS) -> tuple[str, list[str]]:
    """Aggregate profile → (band, reasons). Structure first, then envelope."""
    reasons = []
    if (
        agg["hf_flatline_frac_mean"] >= bands["hf_flatline_frac_broken"]
        and agg["flatline_clustering_mean"] >= bands["flatline_clustering_broken"]
    ):
        reasons.append(
            f"clustered hf flatline: frac={agg['hf_flatline_frac_mean']:.3f}, "
            f"clustering={agg['flatline_clustering_mean']:.2f}x"
        )
    if agg["commit_shift_max_abs"] >= bands["commit_shift_broken"]:
        reasons.append(f"commit-CDF hole: {agg['commit_shift_max_abs']:.3f}")
    if agg["hf_blowup_knots"] >= bands["hf_blowup_knots_broken"]:
        reasons.append(
            f"hf blow-up: {agg['hf_blowup_knots']} knots with "
            f"|hf_ratio-1| >= {bands['hf_ratio_dev_broken']}"
        )
    if reasons:
        return "process-broken", reasons
    for metric, bound in (
        ("dE_max_abs", bands["dE_intact"]),
        ("commit_shift_max_abs", bands["commit_shift_intact"]),
        ("hf_ratio_dev_max", bands["hf_ratio_dev_intact"]),
    ):
        if agg[metric] > bound:
            reasons.append(f"{metric}={agg[metric]:.3f} > {bound}")
    return ("perturbed", reasons) if reasons else ("intact", reasons)


def gauge(baseline_dir: str, candidate_dir: str) -> dict:
    """The whole gauge: pair, compare, aggregate, verdict."""
    pairs = pair_traces(baseline_dir, candidate_dir)
    profiles = [compare_pair(b, c) for b, c in pairs]
    sig = profiles[0]["sigmas"]

    def mean_curve(key: str) -> np.ndarray:
        return np.nanmean(np.stack([p[key] for p in profiles]), axis=0)

    curves = {
        key: mean_curve(key).round(5).tolist()
        for key in ("x0_rmse", "code_mismatch", "dE", "hf_ratio", "commit_shift")
    }
    agg = {
        "n_pairs": len(profiles),
        "x0_rmse_max": round(float(np.nanmax(curves["x0_rmse"])), 5),
        "code_mismatch_max": round(float(np.nanmax(curves["code_mismatch"])), 5),
        "dE_max_abs": round(float(np.nanmax(np.abs(curves["dE"]))), 5),
        "commit_shift_max_abs": round(
            float(np.nanmax(np.abs(curves["commit_shift"]))), 5
        ),
        "hf_ratio_min": round(float(np.nanmin(curves["hf_ratio"])), 5),
        "hf_ratio_dev_max": round(
            float(np.nanmax(np.abs(np.asarray(curves["hf_ratio"]) - 1.0))), 5
        ),
        "hf_blowup_knots": int(
            np.nansum(
                np.abs(np.asarray(curves["hf_ratio"]) - 1.0)
                >= BANDS["hf_ratio_dev_broken"]
            )
        ),
        "hf_flatline_frac_mean": round(
            float(np.mean([p["hf_flatline_frac"] for p in profiles])), 5
        ),
        "flatline_clustering_mean": round(
            float(np.mean([p["flatline_clustering"] for p in profiles])), 5
        ),
    }
    band, reasons = verdict_for(agg)
    return {
        "verdict": band,
        "reasons": reasons,
        "aggregate": agg,
        "curves": {"sigmas": sig.round(6).tolist(), **curves},
        "bands": BANDS,
        "baseline": str(baseline_dir),
        "candidate": str(candidate_dir),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--baseline", required=True, help="dir of baseline trace npz")
    p.add_argument("--candidate", required=True, help="dir of candidate trace npz")
    p.add_argument("--out", default=None, help="write the full profile json here")
    args = p.parse_args()
    res = gauge(args.baseline, args.candidate)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    print(
        json.dumps(
            {key: res[key] for key in ("verdict", "reasons", "aggregate")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
