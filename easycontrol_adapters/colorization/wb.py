"""Target white-balance for the colorize task.

The colorize training-target pool (the shared color corpus) is corpus-wide
warm/desaturated — median per-image mean HSV saturation 0.182, bright-pixel
white-point R−B +0.089 on average, 84.7% of targets warm-dominant (measured
2026-07-09 over the 2,951 staged pairs; see queued.md). With
``caption_dropout_rate = 0.8`` collapsing most gradient into one shared
unconditional colorization mode, the adapter faithfully reproduces that
corpus-mean sepia tone on *every* cond. Tag filtering can't fix it (the tagger
emits no tone tags for this corpus, and the low-sat tail is 33–94% of the data
depending on threshold), so the fix is *correction*: neutralize each target's
white-point at prep time and encode the corrected image into a
colorize-specific target-latent cache (subset ``latent_cache_dir``).

Pure numpy on uint8 RGB ``(H, W, 3)`` arrays; the prep stage wires these into
``library.preprocess.cache_latents``'s ``image_transform`` hook.
"""

from __future__ import annotations

import numpy as np

# Rec.709 luma weights — matches the vision-side convention elsewhere in the repo.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Below this mean HSV saturation a target is effectively monochrome/sepia-core:
#: white-balancing it just yields flat gray — a bad colorize target. The prep
#: stage drops these (removes their cond latent so the loader pairs them out).
DEFAULT_DROP_SAT = 0.10

#: Per-channel gain clamp. The measured corpus cast needs ~1.1–1.2x; anything
#: past this is a heavily stylized palette we shouldn't "correct" away.
DEFAULT_MAX_GAIN = 1.35

#: Luma quantile that defines the bright (paper/highlight) region used to
#: estimate the white-point.
BRIGHT_QUANTILE = 0.92

#: Bright pixels must also clear this absolute luma (uint8 scale) — a dark
#: image's brightest 8% is not paper. Below it we fall back to gray-world.
BRIGHT_LUMA_FLOOR = 160.0

#: Minimum fraction of pixels the bright mask must cover to be trusted.
MIN_BRIGHT_FRAC = 0.005


def mean_saturation(img: np.ndarray) -> float:
    """Mean HSV saturation of a uint8 RGB image, in [0, 1].

    Same metric as the 2026-07-09 corpus stats (S = (max−min)/max, 0 where
    max = 0), so thresholds carry over. Call on a thumbnail for speed — the
    statistic is scale-stable."""
    f = img.astype(np.float32) / 255.0
    mx = f.max(axis=-1)
    mn = f.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(sat.mean())


def whitepoint_gains(
    img: np.ndarray,
    *,
    max_gain: float = DEFAULT_MAX_GAIN,
    strength: float = 1.0,
) -> np.ndarray:
    """Per-channel gains that neutralize the image's white-point.

    Estimates the cast from the bright (paper/highlight) region — pixels at or
    above the :data:`BRIGHT_QUANTILE` luma quantile AND :data:`BRIGHT_LUMA_FLOOR`
    — and returns gains scaling each channel mean to their common mean (so
    bright-region R≈G≈B). Images without a trustworthy bright region (dark or
    tiny highlight coverage) fall back to gray-world over the whole image.
    Gains are clamped to ``[1/max_gain, max_gain]`` and softened by
    ``gains ** strength`` (``strength < 1`` = partial correction)."""
    f = img.astype(np.float32)
    luma = f @ _LUMA
    thresh = max(float(np.quantile(luma, BRIGHT_QUANTILE)), BRIGHT_LUMA_FLOOR)
    mask = luma >= thresh
    if mask.mean() < MIN_BRIGHT_FRAC:
        mask = np.ones(luma.shape, dtype=bool)  # gray-world fallback
    means = np.maximum(f[mask].mean(axis=0), 1e-3)
    gains = means.mean() / means
    gains = np.clip(gains, 1.0 / max_gain, max_gain)
    if strength != 1.0:
        gains = gains**strength
    return gains.astype(np.float32)


def apply_whitebalance(
    img: np.ndarray,
    *,
    max_gain: float = DEFAULT_MAX_GAIN,
    strength: float = 1.0,
) -> np.ndarray:
    """White-balance a uint8 RGB image via :func:`whitepoint_gains`."""
    gains = whitepoint_gains(img, max_gain=max_gain, strength=strength)
    out = img.astype(np.float32) * gains
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)
