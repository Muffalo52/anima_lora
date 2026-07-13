"""G3 unit gates for the artist-atlas pack (docs/proposal/artist_atlas_pack.md).

CPU-only, synthetic state dicts. Invariants pinned:
- packed expert e reconstructs ingredient e's fused ΔW exactly (alpha scaling,
  inv_scale folding, and BOTH attn-save shapes: pre-fused cloned q/k/v downs
  AND per-component soup-style downs → block-diagonal fuse + zero-pad align)
- the packed file loads through create_network_from_weights (three-axis
  stamps: independent_A / no per-layer router / router_source="none") with the
  detected expert count (regression: stacked-experts from_weights used to fall
  back to num_experts=4)
- one-hot gate == solo ingredient forward (G1 at module level); zero gate ==
  base forward (G2); unset gates default to uniform 1/E = ingredient average
  (the documented atlas gotcha — callers must set gates per request)
- non-plain (hydra-shaped) ingredients are refused loudly
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from networks.lora_anima.factory import create_network_from_weights
from networks.lora_anima.loading import (
    _refuse_split_stacked_experts_keys,
    _stack_lora_ups,
)
from scripts.atlas.pack import (
    atlas_metadata,
    default_tag,
    pack_atlas_file,
    pack_state_dicts,
)

IN, HEAD, RANK = 8, 6, 3
E = 2  # experts / ingredients

_QKV = "lora_unet_blocks_0_self_attn_qkv_proj"
_MLP = "lora_unet_blocks_0_mlp_layer1"


class _SelfAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(IN, 3 * HEAD, bias=False)


# Class name must be exactly "Block" — LoRANetwork.ANIMA_TARGET_REPLACE_MODULE
# matches on __class__.__name__.
class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttn()
        self.mlp_layer1 = nn.Linear(IN, HEAD, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block()])


def _split_attn_sd(
    seed: int, alpha: float = 90.0, cloned_downs: bool = False, inv_scale: bool = False
) -> dict:
    """On-disk plain-LoRA sd: split q/k/v (out HEAD each) + one mlp module.

    ``cloned_downs=True`` mimics a direct train.py save (one fused runtime
    module split at save → identical downs); ``False`` mimics a soup output
    (per-component SVD → distinct downs).
    """
    g = torch.Generator().manual_seed(seed)
    sd: dict = {}
    shared_down = torch.randn(RANK, IN, generator=g)
    inv = (torch.rand(IN, generator=g) + 0.5) if inv_scale else None
    for comp in ("q", "k", "v"):
        m = f"lora_unet_blocks_0_self_attn_{comp}_proj"
        sd[f"{m}.lora_down.weight"] = (
            shared_down.clone() if cloned_downs else torch.randn(RANK, IN, generator=g)
        )
        sd[f"{m}.lora_up.weight"] = torch.randn(HEAD, RANK, generator=g)
        sd[f"{m}.alpha"] = torch.tensor(alpha)
        if inv is not None:
            sd[f"{m}.inv_scale"] = inv.clone()
    sd[f"{_MLP}.lora_down.weight"] = torch.randn(RANK, IN, generator=g)
    sd[f"{_MLP}.lora_up.weight"] = torch.randn(HEAD, RANK, generator=g)
    sd[f"{_MLP}.alpha"] = torch.tensor(alpha)
    return sd


def _component_delta_w(sd: dict, module: str) -> torch.Tensor:
    """ΔW of one on-disk module as a loaded LoRAModule applies it (raw input)."""
    down = sd[f"{module}.lora_down.weight"].to(torch.float32)
    up = sd[f"{module}.lora_up.weight"].to(torch.float32)
    scale = float(sd[f"{module}.alpha"]) / down.shape[0]
    inv = sd.get(f"{module}.inv_scale")
    if inv is not None:
        down = down * inv.to(torch.float32).unsqueeze(0)
    return up @ down * scale


def _fused_qkv_delta_w(sd: dict) -> torch.Tensor:
    """Ground-truth fused ΔW (3·HEAD, IN): per-component ΔWs stacked over out."""
    return torch.cat(
        [
            _component_delta_w(sd, f"lora_unet_blocks_0_self_attn_{c}_proj")
            for c in ("q", "k", "v")
        ],
        dim=0,
    )


def _ingredients() -> list[dict]:
    # Ingredient 0: soup-style (distinct downs → block-diag fuse, rank 3·RANK,
    # with inv_scale). Ingredient 1: pre-fused style (cloned downs, rank RANK)
    # → exercises the zero-pad rank alignment across experts.
    return [
        _split_attn_sd(1, alpha=90.0, cloned_downs=False, inv_scale=True),
        _split_attn_sd(2, alpha=48.0, cloned_downs=True),
    ]


def _atlas_runtime_sd(sds: list[dict]) -> dict:
    """Pack → on-disk split → back through the loader's stack + refuse path."""
    from networks.lora_modules.stacked_experts import StackedExpertsLoRAModule

    atlas = pack_state_dicts(sds, out_dtype=torch.float32)
    disk = StackedExpertsLoRAModule.build_moe_state_dict(atlas, torch.float32)
    disk = _stack_lora_ups(disk)
    return _refuse_split_stacked_experts_keys(disk)


def test_pack_roundtrip_delta_w_exact():
    sds = _ingredients()
    runtime = _atlas_runtime_sd(sds)

    for module, ref_fn in (
        (_QKV, _fused_qkv_delta_w),
        (_MLP, lambda sd: _component_delta_w(sd, _MLP)),
    ):
        down = runtime[f"{module}.lora_down_weight"]  # (E, R, in)
        up = runtime[f"{module}.lora_up_weight"]  # (E, out, R)
        alpha = float(runtime[f"{module}.alpha"])
        assert down.shape[0] == E and up.shape[0] == E
        assert alpha == down.shape[1]  # scale 1
        for e, sd in enumerate(sds):
            delta = up[e].to(torch.float32) @ down[e].to(torch.float32)
            torch.testing.assert_close(delta, ref_fn(sd), rtol=1e-5, atol=1e-5)


def test_qkv_rank_is_blockdiag_vs_prefused():
    """Soup-style ingredient fuses to 3·RANK; pre-fused stays RANK; the stack
    pads to the max — pinning the exact-fuse geometry, not an SVD approximation."""
    runtime = _atlas_runtime_sd(_ingredients())
    down = runtime[f"{_QKV}.lora_down_weight"]
    assert down.shape == (E, 3 * RANK, IN)
    # Pre-fused expert 1 occupies only the first RANK rows; padding is zero.
    assert torch.all(down[1, RANK:] == 0)


def test_atlas_loads_and_forward_gates_exact(tmp_path):
    sds = _ingredients()
    ckpts = []
    for i, sd in enumerate(sds):
        p = tmp_path / f"anima_soup_artist{i}.safetensors"
        save_file(sd, str(p))
        ckpts.append(str(p))
    out = tmp_path / "anima_atlas_moe.safetensors"
    pack_atlas_file(ckpts, str(out), out_dtype=torch.float32)

    dit = _TinyDiT()
    network, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=dit,
        for_inference=True,
    )
    # Regression pin: expert count comes from the checkpoint, not the default 4.
    assert network.cfg.num_experts == E
    assert network.cfg.use_moe_style == "independent_A"
    assert network.cfg.route_per_layer is False
    assert network.cfg.router_source == "none"
    assert network.global_router is None

    network.apply_to([], dit, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    assert not info.missing_keys, info.missing_keys
    assert not info.unexpected_keys, info.unexpected_keys
    network.eval().requires_grad_(False)

    x = torch.randn(1, 5, IN)
    qkv = dit.blocks[0].self_attn.qkv_proj
    base = torch.nn.functional.linear(x, qkv.weight.detach().clone())

    # G1 (one-hot == solo ingredient) per expert.
    for e, sd in enumerate(sds):
        gate = torch.zeros(1, E)
        gate[0, e] = 1.0
        network.set_routing_weights(gate)
        expected = base + x @ _fused_qkv_delta_w(sd).T
        torch.testing.assert_close(qkv(x), expected, rtol=1e-4, atol=1e-4)

    # G2 (zero gate == base model).
    network.set_routing_weights(torch.zeros(1, E))
    torch.testing.assert_close(qkv(x), base, rtol=1e-6, atol=1e-6)

    # Documented gotcha: unset/cleared gates default to uniform 1/E — the
    # ingredient average, NOT the base model.
    network.clear_routing_weights()
    avg = base + x @ torch.stack([_fused_qkv_delta_w(sd) for sd in sds]).mean(0).T
    torch.testing.assert_close(qkv(x), avg, rtol=1e-4, atol=1e-4)


def test_pack_refuses_nonplain():
    sd = _split_attn_sd(3)
    sd[f"{_MLP}.lora_up_weight"] = torch.randn(3, HEAD, RANK)  # hydra-shaped
    with pytest.raises(ValueError, match="only plain LoRA"):
        pack_state_dicts([sd, _split_attn_sd(4)])


def test_metadata_and_tags():
    assert default_tag("output/ckpt/soup/anima_soup_hews.safetensors") == "hews"
    md = atlas_metadata(["a/anima_soup_x.safetensors"] * 2, ["x", "y"])
    assert md["ss_use_moe_style"] == "independent_A"
    assert md["ss_router_source"] == "none"
    with pytest.raises(ValueError, match="Duplicate"):
        pack_atlas_file(["a.safetensors", "b.safetensors"], "o", tags=["x", "x"])
