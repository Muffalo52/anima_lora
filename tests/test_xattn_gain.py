"""Invariants for the Block._xattn_gain cross-attn residual gain.

The buffer backs the frontload_text_boost arm-(b) bench
(bench/frontload_text_boost/). Two things must hold:

1. gain = 1.0 (the default) is EXACT identity — the training path and every
   checkpoint rendered before the buffer existed are bit-unchanged.
2. gain = g scales exactly the cross-attn residual (not self-attn, not MLP),
   and set_xattn_gain reaches every block.
"""

import torch
import torch.nn as nn

from library.anima.models import Block
from library.inference.adapters import set_xattn_gain


class _StubAttn(nn.Module):
    """Deterministic stand-in for Attention — avoids the dispatcher on CPU."""

    def __init__(self, val: float):
        super().__init__()
        self.val = val

    def forward(self, x, attn_params, ctx, rope_cos_sin=None):
        return torch.full_like(x, self.val)


class _ZeroMLP(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def _make_block(cross_val: float = 1.0) -> Block:
    torch.manual_seed(0)
    b = Block(x_dim=8, context_dim=8, num_heads=2)
    b.self_attn = _StubAttn(0.0)
    b.cross_attn = _StubAttn(cross_val)
    b.mlp = _ZeroMLP()
    return b.eval()


def _run(block: Block) -> torch.Tensor:
    torch.manual_seed(1)
    x = torch.randn(2, 1, 2, 2, 8)
    emb = torch.randn(2, 1, 8)
    ctx = torch.randn(2, 4, 8)
    with torch.no_grad():
        return block._forward(x, emb, ctx, attn_params=None)


def test_default_gain_is_exact_identity():
    b = _make_block()
    out_default = _run(b)
    b._xattn_gain.fill_(1.0)
    assert torch.equal(_run(b), out_default)


def test_gain_scales_only_cross_residual():
    b = _make_block(cross_val=1.0)
    out_g1 = _run(b)
    b._xattn_gain.fill_(0.0)
    out_g0 = _run(b)  # cross branch fully removed
    b._xattn_gain.fill_(2.0)
    out_g2 = _run(b)
    # residual doubles (×2 exact on the product; the surrounding residual
    # add/subtract reintroduces rounding, so allclose not bit-equal)
    assert torch.allclose(
        out_g2 - out_g0, 2.0 * (out_g1 - out_g0), atol=1e-6, rtol=1e-5
    )
    # with the cross stub emitting a nonzero residual, gain must change output
    assert not torch.equal(out_g1, out_g0)


def test_set_xattn_gain_reaches_all_blocks():
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_make_block() for _ in range(3)])

    m = _Model()
    set_xattn_gain(m, 1.5)
    for blk in m.blocks:
        assert float(blk._xattn_gain) == 1.5
    set_xattn_gain(m, 1.0)
    for blk in m.blocks:
        assert float(blk._xattn_gain) == 1.0


def test_gain_buffer_not_in_state_dict():
    b = _make_block()
    assert "_xattn_gain" not in b.state_dict()


def test_side_channels_resolve_boost_off_at_identity():
    """--xattn_boost 1.0 (the default) must resolve to None (off) so every
    denoise-loop runner can gate on ``xattn_boost is not None``."""
    from types import SimpleNamespace

    from library.inference.sampler_context import SamplerSideChannels

    off = SamplerSideChannels.from_args(SimpleNamespace())
    assert off.xattn_boost is None
    assert off.xattn_boost_band == 0.85

    on = SamplerSideChannels.from_args(
        SimpleNamespace(xattn_boost=2.0, xattn_boost_band=0.9)
    )
    assert on.xattn_boost == 2.0
    assert on.xattn_boost_band == 0.9


def test_cli_defaults_are_identity():
    """inference.py must default to boost-off; the flag names are load-bearing
    (the XATTN_BOOST env lever and the bench build argv from them)."""
    from library.inference.args import build_parser

    args, _ = build_parser().parse_known_args(
        ["--prompt", "x", "--text_encoder", "te", "--save_path", "out"]
    )
    assert args.xattn_boost == 1.0
    assert args.xattn_boost_band == 0.85
