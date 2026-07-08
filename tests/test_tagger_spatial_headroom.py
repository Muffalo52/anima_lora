"""Invariant tests for the tagger spatial-headroom Phase-1 levers.

Pins the pure/structural pieces the retrain rests on
(docs/proposal/tagger_spatial_head_headroom.md):

* the spatial/rest parameter partition (``spatial_param_names``),
* the spatial mean-AP selection metric (``spatial_mean_ap``),
* ``_build_optimizer`` param-grouping (default-inert vs split),
* the refit-stage freeze invariant: with the non-spatial params frozen, an
  optimizer step over the spatial branch leaves every identity/core/rating/
  people weight bit-identical (so those near-solved slices cannot regress).
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
from scripts.anima_tagger.train_cached import _build_optimizer
from scripts.anima_tagger.train_common import spatial_mean_ap, spatial_param_names


def _make_model(pool_kind_aux: str = "map") -> AnimaTaggerHead:
    cfg = AnimaTaggerConfig(
        d_in=16,
        n_tags=10,
        d_in_aux=24,
        n_ratings=3,
        n_people_counts=4,
        d_hidden=8,
        pool_kind="mean",
        pool_kind_aux=pool_kind_aux,
        pool_n_queries_aux=2,
        pool_n_heads_aux=2,
        pool_use_cls_aux=False,
        pool_use_mean_aux=False,
        tag_indices_core=[0, 1, 2],
        tag_indices_spatial=[3, 4, 5, 6, 7, 8, 9],
    )
    return AnimaTaggerHead(cfg)


def _base_args(**over) -> Namespace:
    args = Namespace(lr=2e-4, weight_decay=0.01, lr_spatial=None, wd_spatial=None)
    for k, v in over.items():
        setattr(args, k, v)
    return args


# ── partition ──────────────────────────────────────────────────────────────


def test_spatial_partition_is_disjoint_and_covers_spatial_modules():
    m = _make_model(pool_kind_aux="map")
    spatial = spatial_param_names(m)
    allnames = {n for n, _ in m.named_parameters()}
    rest = allnames - spatial

    # Every spatial param is under exactly the three spatial submodules...
    assert spatial == {
        n
        for n in allnames
        if n.startswith(("pool_spatial.", "trunk_spatial.", "tag_head_spatial."))
    }
    # ...and pool_spatial IS present with the map pool.
    assert any(n.startswith("pool_spatial.") for n in spatial)
    # The rest are the identity/core/rating/people params — no leakage either way.
    assert not any(
        n.startswith(("pool_spatial.", "trunk_spatial.", "tag_head_spatial."))
        for n in rest
    )
    for expect in ("rating_head", "people_head", "tag_head_core", "trunk_core"):
        assert any(n.startswith(expect) for n in rest)
    assert spatial and rest and (spatial | rest) == allnames


# ── spatial_mean_ap ──────────────────────────────────────────────────────────


def test_spatial_mean_ap_perfect_and_restriction():
    m = _make_model()
    spatial_idx = torch.tensor(m.cfg.tag_indices_spatial)
    B, n = 8, m.cfg.n_tags
    logits = torch.full((B, n), -10.0)
    mh = torch.zeros(B, n)
    # A spatial tag ranked perfectly → AP 1; a core tag is NOT counted.
    mh[:4, 3] = 1
    logits[:4, 3] = 10.0
    mh[:4, 0] = 1  # core positive, but core is off the spatial index
    logits[:4, 0] = -10.0  # would tank AP if (wrongly) included
    assert abs(spatial_mean_ap(logits, mh, spatial_idx) - 1.0) < 1e-6


def test_spatial_mean_ap_ignores_unsupported_tags():
    m = _make_model()
    spatial_idx = torch.tensor(m.cfg.tag_indices_spatial)
    B, n = 6, m.cfg.n_tags
    logits = torch.randn(B, n)
    mh = torch.zeros(B, n)  # no positives anywhere → all-NaN → nanmean is nan
    val = spatial_mean_ap(logits, mh, spatial_idx)
    assert val != val  # nan


# ── optimizer param-grouping ─────────────────────────────────────────────────


def test_build_optimizer_default_is_single_group():
    m = _make_model()
    opt = _build_optimizer(m, _base_args(), spatial_names=spatial_param_names(m))
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 2e-4
    assert opt.param_groups[0]["weight_decay"] == 0.01


def test_build_optimizer_splits_spatial_group():
    m = _make_model()
    spatial = spatial_param_names(m)
    opt = _build_optimizer(
        m, _base_args(lr_spatial=1e-3, wd_spatial=0.0), spatial_names=spatial
    )
    assert len(opt.param_groups) == 2
    rest_g, spatial_g = opt.param_groups
    assert rest_g["lr"] == 2e-4 and rest_g["weight_decay"] == 0.01
    assert spatial_g["lr"] == 1e-3 and spatial_g["weight_decay"] == 0.0
    # The split routes exactly the spatial params (by identity) into group 2.
    id_to_name = {id(p): n for n, p in m.named_parameters()}
    got = {id_to_name[id(p)] for p in spatial_g["params"]}
    assert got == spatial


def test_build_optimizer_partial_override_inherits_core_wd():
    m = _make_model()
    # Only lr_spatial set → spatial wd inherits --weight_decay.
    opt = _build_optimizer(
        m, _base_args(lr_spatial=5e-4), spatial_names=spatial_param_names(m)
    )
    assert len(opt.param_groups) == 2
    assert opt.param_groups[1]["lr"] == 5e-4
    assert opt.param_groups[1]["weight_decay"] == 0.01


# ── refit-stage freeze invariant ─────────────────────────────────────────────


def test_refit_freeze_leaves_nonspatial_bit_identical():
    torch.manual_seed(0)
    m = _make_model(pool_kind_aux="mean")
    spatial = spatial_param_names(m)

    # Mirror the refit setup: freeze everything but the spatial branch.
    for name, p in m.named_parameters():
        p.requires_grad_(name in spatial)
    refit_params = [p for n, p in m.named_parameters() if n in spatial]
    assert all(p.requires_grad for p in refit_params)
    assert all(not p.requires_grad for n, p in m.named_parameters() if n not in spatial)

    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.AdamW(refit_params, lr=1e-2)

    feat_core = torch.randn(4, m.cfg.d_in)
    feat_spatial = torch.randn(4, m.cfg.d_in_aux)
    target = torch.randint(0, 2, (4, m.cfg.n_tags)).float()
    tag_logits, _rate, _people = m(feat_core, feat_spatial)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(tag_logits, target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    after = {n: p.detach() for n, p in m.named_parameters()}
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    # Non-spatial params are bit-identical; at least one spatial param moved.
    assert changed and changed <= spatial
    assert not (changed & ({n for n in before} - spatial))
