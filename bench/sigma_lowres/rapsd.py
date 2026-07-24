#!/usr/bin/env python
"""sigma_lowres Measurement A — latent RAPSD vs flow-matching noise floor.

Replicates SwD's spectral analysis (arXiv:2503.16397 §3, Fig 1) for Anima's
Qwen-VAE latents: under ``(1-σ)·x0 + σ·ε`` noising, which spatial frequencies
survive above the noise floor at each σ?

Per radial-frequency bin f (cycles/latent-pixel, Nyquist = 0.5) with clean
RAPSD P(f) and unit-variance white noise (analytic PSD = 1):

    signal-vs-noise crossover:  (1-σ)²·P(f) = σ²  ⇔  σ_eq(f) = √P(f) / (1+√P(f))

σ_eq(f) is the σ at which frequency f sinks into the noise. The predicted
safe-demotion boundary for tier ``e`` (native 1024) is σ*(e) = σ_eq at the
demoted grid's Nyquist 0.5·e/1024 — above σ*(e), everything the demoted grid
cannot represent is already noise-masked (H1), and the gradient probe
(Measurement B) should see the demotion gap collapse (H2/H3).

Probe-set selection is shared with ``bench/tier_routing`` so Measurement B
runs on the same images.

Usage::

    uv run python bench/sigma_lowres/rapsd.py --label phase0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.tier_routing.redundancy import score_corpus, select_probe_set  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NATIVE_EDGE = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--num_images", type=int, default=40)
    p.add_argument("--tier", type=int, default=NATIVE_EDGE)
    p.add_argument("--demote_edges", default="896,768")
    p.add_argument("--nbins", type=int, default=64, help="radial frequency bins")
    p.add_argument("--artists", default=None, help="csv restriction on the corpus")
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument("--score_limit", type=int, default=None)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    return p.parse_args()


def rapsd(lat: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged PSD of a ``(C, H, W)`` latent.

    Returns ``(bin_centers, P)`` with P normalized so unit-variance white
    noise gives P ≈ 1 in every bin (i.e. |FFT|²/(H·W), channel-averaged).
    Frequencies in cycles/latent-pixel; rectangular grids are handled by
    binning on the true 2D radius, cropped at the 0.5 Nyquist circle.
    """
    c, h, w = lat.shape
    power = np.abs(np.fft.fft2(lat.astype(np.float64))) ** 2 / (h * w)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx**2 + fy**2)
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    keep = (idx >= 0) & (idx < nbins)
    pw = power.reshape(c, -1).mean(axis=0)[keep]
    idx = idx[keep]
    sums = np.bincount(idx, weights=pw, minlength=nbins)
    counts = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid="ignore"):
        prof = sums / counts
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


def sigma_eq(p: np.ndarray) -> np.ndarray:
    """σ at which (1-σ)²·P == σ²·1 per bin: σ_eq = √P / (1+√P)."""
    root = np.sqrt(np.maximum(p, 0.0))
    return root / (1.0 + root)


def band_snr(centers: np.ndarray, p: np.ndarray, f_lo: float, sigma: float) -> float:
    """Mean signal/noise power ratio in the above-Nyquist band (f > f_lo)."""
    m = centers > f_lo
    if not m.any() or sigma <= 0:
        return float("inf")
    return float(((1.0 - sigma) ** 2 * np.nanmean(p[m])) / sigma**2)


def main() -> None:
    args = parse_args()
    edges = [int(e) for e in args.demote_edges.split(",") if e]

    artists = args.artists.split(",") if args.artists else None
    records = score_corpus(artists=artists, k=args.quant_k, limit=args.score_limit)
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    log.info(
        f"probe set: {len(probe)} images, {len({r.artist for r in probe})} artists"
    )

    profiles = []
    per_image: list[dict] = []
    for r in probe:
        d = np.load(r.npz_path)
        lat_key = next(k for k in d.files if k.startswith("latents_"))
        lat = d[lat_key].astype(np.float32)  # (16, H_lat, W_lat)
        centers, prof = rapsd(lat, args.nbins)
        profiles.append(prof)
        row = {
            "artist": r.artist,
            "stem": r.stem,
            "lat_var": round(float(lat.var()), 5),
        }
        for e in edges:
            f_nyq = 0.5 * e / args.tier
            i = int(np.searchsorted(centers, f_nyq))
            i = min(i, args.nbins - 1)
            row[f"sigma_star_{e}"] = round(float(sigma_eq(prof[i : i + 1])[0]), 5)
        per_image.append(row)

    centers = np.linspace(0.0, 0.5, args.nbins + 1)
    centers = 0.5 * (centers[:-1] + centers[1:])
    p_mean = np.nanmean(np.stack(profiles), axis=0)
    seq = sigma_eq(p_mean)

    run_dir = make_run_dir("sigma_lowres", args.label)

    headline: dict = {
        "n_images": len(probe),
        "nbins": args.nbins,
        "lat_var_mean": round(float(np.mean([r["lat_var"] for r in per_image])), 5),
        # spectrum shape summary: P at a few normalized frequencies
        "P_f0.05": round(float(np.interp(0.05, centers, p_mean)), 4),
        "P_f0.125": round(float(np.interp(0.125, centers, p_mean)), 4),
        "P_f0.25": round(float(np.interp(0.25, centers, p_mean)), 4),
        "P_f0.5": round(float(p_mean[-1]), 4),
    }
    sigma_grid = [round(s, 2) for s in np.arange(0.05, 1.0, 0.05)]
    for e in edges:
        f_nyq = 0.5 * e / args.tier
        stars = np.array([r[f"sigma_star_{e}"] for r in per_image])
        headline[f"f_nyquist_{e}"] = round(f_nyq, 4)
        headline[f"sigma_star_{e}"] = round(float(np.interp(f_nyq, centers, seq)), 4)
        headline[f"sigma_star_{e}_p10"] = round(float(np.percentile(stars, 10)), 4)
        headline[f"sigma_star_{e}_p90"] = round(float(np.percentile(stars, 90)), 4)
        headline[f"band_snr_{e}"] = {
            f"{s:.2f}": round(band_snr(centers, p_mean, f_nyq, s), 4)
            for s in sigma_grid
        }

    curves = {
        "freq": [round(float(f), 5) for f in centers],
        "P_mean": [round(float(v), 6) for v in p_mean],
        "sigma_eq": [round(float(v), 5) for v in seq],
    }
    (run_dir / "curves.json").write_text(json.dumps(curves, indent=1))
    rows_path = run_dir / "per_image.jsonl"
    with rows_path.open("w") as f:
        for row in per_image:
            f.write(json.dumps(row) + "\n")

    # ---- Fig-1-style plot -------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    show_sigmas = [0.2, 0.4, 0.6, 0.8]
    ax1.loglog(centers, p_mean, color="purple", lw=2, label="clean latents P(f)")
    for s in show_sigmas:
        ax1.loglog(
            centers,
            (1 - s) ** 2 * p_mean + s**2,
            lw=1,
            alpha=0.8,
            label=f"noisy σ={s}",
        )
        ax1.axhline(s**2, ls=":", lw=0.6, color="gray")
    for e in edges:
        ax1.axvline(0.5 * e / args.tier, ls="--", lw=1, label=f"{e} Nyquist")
    ax1.set_xlabel("spatial frequency (cycles / latent px)")
    ax1.set_ylabel("power")
    ax1.set_title("Anima latent RAPSD vs FM noise")
    ax1.legend(fontsize=7)

    ax2.plot(centers, seq, color="purple", lw=2)
    for e in edges:
        f_nyq = 0.5 * e / args.tier
        s_star = float(np.interp(f_nyq, centers, seq))
        ax2.axvline(f_nyq, ls="--", lw=1)
        ax2.axhline(s_star, ls=":", lw=0.8)
        ax2.annotate(f"σ*({e})={s_star:.2f}", (f_nyq, s_star), fontsize=8)
    ax2.set_xlabel("spatial frequency (cycles / latent px)")
    ax2.set_ylabel("σ_eq(f)")
    ax2.set_title("σ where each frequency sinks into noise")
    fig.tight_layout()
    plot_path = run_dir / "rapsd.png"
    fig.savefig(plot_path, dpi=140)

    log.info(
        json.dumps(
            {k: v for k, v in headline.items() if "band_snr" not in str(k)},
            indent=2,
            default=str,
        )
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path, run_dir / "curves.json", plot_path],
        extra={"probe_set": [f"{r.artist}/{r.stem}" for r in probe]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
