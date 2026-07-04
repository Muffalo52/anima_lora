"""Invariant tests for the label-embedding tag head.

Synthetic tensors only — no encoders, no KB, no text-embedding model. Pins:
config round-trip + pre-label-embed dict compatibility, forward shape parity
with the linear head, vocab-order → side-local embedding slicing, state-dict
self-containment (checkpoints carry the embeddings), and the frozen/trainable
parameterization split.
"""

from __future__ import annotations

import pytest
import torch

from library.captioning.anima_tagger_model import (
    AnimaTaggerConfig,
    AnimaTaggerHead,
)


def _routing(n_tags: int, n_core: int):
    # Interleaved on purpose — catches any code that assumes contiguous sides.
    core = list(range(0, 2 * n_core, 2))
    spatial = sorted(set(range(n_tags)) - set(core))
    return core, spatial


def _cfg(n_tags: int = 24, n_core: int = 6, **kw) -> AnimaTaggerConfig:
    core, spatial = _routing(n_tags, n_core)
    base = dict(
        d_in=32,
        n_tags=n_tags,
        d_in_aux=16,
        d_hidden=48,
        pool_kind="mean",
        pool_kind_aux="mean",
        tag_indices_core=core,
        tag_indices_spatial=spatial,
    )
    base.update(kw)
    return AnimaTaggerConfig(**base)


def _le_cfg(**kw) -> AnimaTaggerConfig:
    return _cfg(tag_head_kind="label_embed", d_label_emb=20, **kw)


# ── Config ────────────────────────────────────────────────────────────────


def test_config_roundtrip_label_embed():
    cfg = _le_cfg(label_emb_trainable=True)
    d = cfg.to_dict()
    assert d["tag_head_kind"] == "label_embed"
    assert d["d_label_emb"] == 20
    assert d["label_emb_trainable"] is True
    cfg2 = AnimaTaggerConfig.from_dict(d)
    assert cfg2.to_dict() == d


def test_pre_label_embed_dict_defaults_to_linear():
    # A v2-era config dict (no tag_head_kind keys) must load as the linear head.
    d = _cfg().to_dict()
    for k in ("tag_head_kind", "d_label_emb", "label_emb_trainable"):
        d.pop(k)
    cfg = AnimaTaggerConfig.from_dict(d)
    assert cfg.tag_head_kind == "linear"
    assert cfg.d_label_emb == 0
    model = AnimaTaggerHead(cfg)
    assert isinstance(model.tag_head_core, torch.nn.Linear)


def test_label_embed_requires_d_label_emb():
    with pytest.raises(ValueError, match="d_label_emb"):
        _cfg(tag_head_kind="label_embed")


def test_unknown_head_kind_rejected():
    with pytest.raises(ValueError, match="tag_head_kind"):
        _cfg(tag_head_kind="mlp")


# ── Forward parity + slicing ──────────────────────────────────────────────


def _full_emb(cfg: AnimaTaggerConfig) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(cfg.n_tags, cfg.d_label_emb, generator=g)


def test_forward_shape_matches_linear_head():
    torch.manual_seed(0)
    B = 3
    feat = torch.randn(B, 32)
    feat_aux = torch.randn(B, 16)

    lin = AnimaTaggerHead(_cfg())
    le = AnimaTaggerHead(_le_cfg())
    le.load_label_embeddings(_full_emb(le.cfg))

    for model in (lin, le):
        tags, rating, people = model(feat, feat_aux)
        assert tags.shape == (B, 24)
        assert rating.shape == (B, 3)
        assert people is None
        assert torch.isfinite(tags).all()


def test_embedding_rows_slice_by_routing_partition():
    cfg = _le_cfg()
    model = AnimaTaggerHead(cfg)
    full = _full_emb(cfg)
    model.load_label_embeddings(full)
    # Side-local row k must be the vocab row the partition scatters back from.
    assert torch.equal(model.tag_head_core.emb, full[model.tag_idx_core])
    assert torch.equal(model.tag_head_spatial.emb, full[model.tag_idx_spatial])


def test_load_shape_and_kind_guards():
    cfg = _le_cfg()
    model = AnimaTaggerHead(cfg)
    with pytest.raises(ValueError, match="shape"):
        model.load_label_embeddings(torch.zeros(cfg.n_tags + 1, cfg.d_label_emb))
    lin = AnimaTaggerHead(_cfg())
    with pytest.raises(ValueError, match="linear"):
        lin.load_label_embeddings(torch.zeros(24, 20))


def test_identical_embeddings_give_identical_logit_gap():
    # Two tags with the same description embedding differ only by bias —
    # the sharing property the head exists for.
    cfg = _le_cfg()
    model = AnimaTaggerHead(cfg).eval()
    full = _full_emb(cfg)
    i, j = cfg.tag_indices_spatial[0], cfg.tag_indices_spatial[1]
    full[j] = full[i]
    model.load_label_embeddings(full)
    with torch.no_grad():
        tags, _, _ = model(torch.randn(4, 32), torch.randn(4, 16))
    head = model.tag_head_spatial
    li = head.bias[(model.tag_idx_spatial == i).nonzero().item()]
    lj = head.bias[(model.tag_idx_spatial == j).nonzero().item()]
    assert torch.allclose(tags[:, i] - tags[:, j], (li - lj).expand(4), atol=1e-5)


# ── State-dict self-containment + parameterization ────────────────────────


def test_state_dict_carries_embeddings_frozen_and_trainable():
    for trainable in (False, True):
        cfg = _le_cfg(label_emb_trainable=trainable)
        model = AnimaTaggerHead(cfg)
        model.load_label_embeddings(_full_emb(cfg))
        sd = model.state_dict()
        assert "tag_head_core.emb" in sd and "tag_head_spatial.emb" in sd

        fresh = AnimaTaggerHead(AnimaTaggerConfig.from_dict(cfg.to_dict()))
        fresh.load_state_dict(sd)  # no KB / embed_tags needed at load time
        assert torch.equal(fresh.tag_head_core.emb, model.tag_head_core.emb)

        emb_is_param = isinstance(fresh.tag_head_core.emb, torch.nn.Parameter)
        assert emb_is_param == trainable


def test_prior_bias_init():
    cfg = _le_cfg()
    model = AnimaTaggerHead(cfg)
    prior = torch.full((cfg.n_tags,), 0.5)
    prior[cfg.tag_indices_core[0]] = 0.0  # clamps, must stay finite
    model.init_tag_bias_from_prior(prior)
    assert torch.isfinite(model.tag_head_core.bias).all()
    # p=0.5 → logit 0.
    k = (model.tag_idx_spatial == cfg.tag_indices_spatial[0]).nonzero().item()
    assert torch.allclose(model.tag_head_spatial.bias[k], torch.tensor(0.0), atol=1e-6)
    lin = AnimaTaggerHead(_cfg())
    with pytest.raises(ValueError, match="linear"):
        lin.init_tag_bias_from_prior(prior)
