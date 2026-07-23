"""tier_routing Phase 3a — per-image latent redundancy scoring + probe-set selection.

The per-image form of the Phase 1 static column
(``bench/traj_stats/run_atlas.py::run_static_arm``): identical k-bit quantizer,
identical double normalization (``(cached_latent - mean) / std`` applied on top
of the already-VAE-normalized cache — that is the established Phase 1
convention, not a bug). Design: ``_archive/proposals/traj_latent_stats.md`` §Phase 3a.

Pure numpy + bucket math; no torch model code here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from library.datasets.buckets import (  # noqa: E402
    EDGE_TOKEN_BANDS,
    freefit_band_for_edge,
    freefit_bucket,
)
from library.inference.traj_stats import (  # noqa: E402
    QWEN_LATENTS_MEAN,
    QWEN_LATENTS_STD,
)

CACHE_ROOT = "post_image_dataset/lora"
RESIZED_ROOT = "post_image_dataset/resized"


def quantize_tokens(lat: np.ndarray, k: int = 4) -> np.ndarray:
    """``(16, H, W)`` cached latent → ``(16, H//2, W//2)`` uint8 token codes.

    Exact Phase 1 static-arm recipe: second per-channel normalization, 2×2
    token mean-pool, uniform k-bit codes over [-4, 4].
    """
    mean = np.asarray(QWEN_LATENTS_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(QWEN_LATENTS_STD, dtype=np.float32)[:, None, None]
    levels = 1 << k
    c, h, w = lat.shape
    h -= h % 2
    w -= w % 2
    xn = (lat[:, :h, :w] - mean) / std
    tok = xn.reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4))
    return np.clip(
        np.floor((np.clip(tok, -4.0, 4.0) + 4.0) / 8.0 * levels), 0, levels - 1
    ).astype(np.uint8)


@dataclass
class ImageRecord:
    stem: str
    artist: str
    npz_path: Path
    png_path: Path
    txt_path: Path
    te_path: Path
    cached_px: tuple[int, int]  # (W, H) of the resized/cached image
    tokens: int  # DiT patch-token count of the cached bucket
    tier: int | None  # inferred EDGE_TOKEN_BANDS tier (None = out of band)
    modal_share: float  # modal joint-code share (high = redundant)
    unique_frac: float  # unique joint-code fraction (low = redundant)

    @property
    def redundancy(self) -> float:
        """The routing scalar: 1 - unique-code fraction (high = flat image)."""
        return 1.0 - self.unique_frac

    def complete(self) -> bool:
        return self.png_path.exists() and self.te_path.exists()


def _infer_tier(tokens: int) -> int | None:
    for edge in sorted(EDGE_TOKEN_BANDS):
        lo, hi = freefit_band_for_edge(edge)
        if lo <= tokens <= hi:
            return edge
    return None


def score_image(npz_path: Path, k: int = 4) -> ImageRecord:
    d = np.load(npz_path)
    lat_key = next(key for key in d.files if key.startswith("latents_"))
    lat = d[lat_key].astype(np.float32)  # (16, H_lat, W_lat)

    codes = quantize_tokens(lat, k)
    c = codes.shape[0]
    flat = codes.reshape(c, -1).T  # (n_tokens, 16) joint codes
    _, counts = np.unique(flat, axis=0, return_counts=True)
    n_tok = flat.shape[0]

    # filename embeds the *pixel* resolution ({stem}_{Wpix}x{Hpix}_anima.npz),
    # the latent key the *latent* one — strip by pattern, not by the key
    stem = re.sub(r"_\d+x\d+_anima\.npz$", "", npz_path.name)
    artist = npz_path.parent.name
    resized_dir = REPO_ROOT / RESIZED_ROOT / artist
    h_lat, w_lat = lat.shape[1], lat.shape[2]
    return ImageRecord(
        stem=stem,
        artist=artist,
        npz_path=npz_path,
        png_path=resized_dir / f"{stem}.png",
        txt_path=resized_dir / f"{stem}.txt",
        te_path=npz_path.parent / f"{stem}_anima_te.safetensors",
        cached_px=(w_lat * 8, h_lat * 8),
        tokens=(h_lat // 2) * (w_lat // 2),
        tier=_infer_tier((h_lat // 2) * (w_lat // 2)),
        modal_share=float(counts.max() / n_tok),
        unique_frac=float(len(counts) / n_tok),
    )


def score_corpus(
    artists: list[str] | None = None, k: int = 4, limit: int | None = None
) -> list[ImageRecord]:
    root = REPO_ROOT / CACHE_ROOT
    if artists:
        paths = sorted(p for a in artists for p in (root / a).glob("*_anima.npz"))
    else:
        paths = sorted(root.glob("*/*_anima.npz"))
    if limit:
        paths = paths[:limit]
    return [score_image(p, k) for p in paths]


def select_probe_set(
    records: list[ImageRecord],
    n: int,
    tier: int = 1024,
    max_per_artist: int | None = None,
) -> list[ImageRecord]:
    """Stratified selection spanning the redundancy range at one tier.

    Sorts complete tier-matched records by redundancy and takes evenly spaced
    picks, so the probe set covers the whole spectrum instead of the bulk of
    the distribution. ``max_per_artist`` caps concentration (the next-ranked
    neighbor substitutes when an artist is full).
    """
    pool = sorted(
        (r for r in records if r.tier == tier and r.complete()),
        key=lambda r: r.redundancy,
    )
    if len(pool) <= n:
        return pool
    picks: list[ImageRecord] = []
    per_artist: dict[str, int] = {}
    taken: set[int] = set()
    targets = np.linspace(0, len(pool) - 1, n).round().astype(int)
    for t in targets:
        # walk outward from the stratum target until an admissible record
        for off in range(len(pool)):
            for idx in (t + off, t - off):
                if idx < 0 or idx >= len(pool) or idx in taken:
                    continue
                r = pool[idx]
                if max_per_artist and per_artist.get(r.artist, 0) >= max_per_artist:
                    continue
                picks.append(r)
                taken.add(idx)
                per_artist[r.artist] = per_artist.get(r.artist, 0) + 1
                break
            else:
                continue
            break
    return sorted(picks, key=lambda r: r.redundancy)


def demoted_bucket(record: ImageRecord, edge: int) -> tuple[int, int]:
    """Free-fit pixel bucket ``(W', H')`` for retraining ``record`` at ``edge``.

    Computed from the cached native-tier pixel size, i.e. the demoted arm is a
    pure down-resize of the exact pixels the native cache encodes — isolating
    the resolution effect from crop-anchor differences (ship-time 3b resizes
    from source instead).
    """
    w, h = record.cached_px
    return freefit_bucket(w, h, freefit_band_for_edge(edge))
