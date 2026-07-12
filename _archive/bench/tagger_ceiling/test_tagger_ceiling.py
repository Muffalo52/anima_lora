"""Invariant tests for the tagger ceiling / attribution probe helpers.

Pins the two pure helpers the verdict rests on: the fixed mean|max|cls pool
shape/order, and :func:`classify_ceiling`'s self-checking attribution logic
(the oracle-validity sanity gate + floor/control lift thresholds).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.tagger_ceiling.run_bench import classify_ceiling, fixed_pool


# ── fixed_pool ────────────────────────────────────────────────────────────


def test_fixed_pool_shape_and_order():
    B, T, D = 2, 5, 4
    tok = torch.randn(B, T, D)
    out = fixed_pool(tok)
    assert out.shape == (B, 3 * D)
    assert torch.allclose(out[:, :D], tok.mean(dim=1))
    assert torch.allclose(out[:, D : 2 * D], tok.amax(dim=1))
    assert torch.allclose(out[:, 2 * D :], tok[:, 0])


def test_fixed_pool_rejects_non_3d():
    try:
        fixed_pool(torch.randn(4, 8))
    except ValueError:
        return
    raise AssertionError("fixed_pool must reject non-[B,T,D] input")


# ── classify_ceiling ──────────────────────────────────────────────────────


def _row(slice_, dep, orc, n_sup=20, sup=100):
    return {
        "slice": slice_,
        "n_supported": n_sup,
        "support_sum": sup,
        "deployed_ap": dep,
        "oracle_ap": orc,
    }


def test_head_headroom_when_oracle_lifts_floor():
    rows = [
        _row("pose", 0.10, 0.30),  # floor, big lift
        _row("expression", 0.15, 0.28),  # floor, big lift
        _row("skin", 0.40, 0.41),  # control, matched
    ]
    v = classify_ceiling(rows)
    assert v["verdict"] == "HEAD_HEADROOM"
    assert v["oracle_valid"] is True
    assert v["floor_median_lift"] > 0.05


def test_feature_ceiling_when_floor_unmoved_but_oracle_valid():
    rows = [
        _row("pose", 0.10, 0.11),  # floor, flat
        _row("expression", 0.15, 0.14),  # floor, flat
        _row("skin", 0.40, 0.42),  # control, matched → oracle well-fit
        _row("eye_color", 0.50, 0.52),  # control, matched
    ]
    v = classify_ceiling(rows)
    assert v["verdict"] == "FEATURE_CEILING"
    assert v["oracle_valid"] is True


def test_inconclusive_when_oracle_underpowered():
    # Oracle is globally worse than deployed even on control slices → the null
    # on floor slices is just undertraining, not a feature ceiling.
    rows = [
        _row("pose", 0.10, 0.09),  # floor, flat
        _row("skin", 0.40, 0.20),  # control, oracle FAR below → gate fails
        _row("eye_color", 0.50, 0.30),  # control, oracle below
    ]
    v = classify_ceiling(rows)
    assert v["oracle_valid"] is False
    assert v["verdict"] == "INCONCLUSIVE"


def test_low_support_slices_ignored():
    rows = [
        _row("pose", 0.10, 0.30, n_sup=2),  # too thin — dropped
        _row("skin", 0.40, 0.41),
    ]
    v = classify_ceiling(rows)
    assert v["n_floor_slices"] == 0
