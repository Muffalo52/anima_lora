"""Unit tests for the ΔW LoRA soup builders + `make soup` pool resolution.

CPU-only, synthetic state dicts. Invariants pinned:
- the soup is exact at the ΔW level (alpha scaling + inv_scale folding included)
- SVD truncation is lossless when ingredients share a rank-r subspace
- non-plain checkpoints (hydra/registers/mismatched recipes) are refused loudly
- the default uncond pool is the round-robin artist shard containing the target
"""

from __future__ import annotations

import pytest
import torch

from scripts.soup.build import soup_state_dicts, truncated_soup
from scripts.soup.pipeline import pool_pattern, resolve_pool, uncond_name

OUT, IN, RANK = 12, 10, 4


def _delta_w(sd: dict, module: str) -> torch.Tensor:
    """ΔW as a loaded LoRAModule would apply it (raw-input path, alpha/r scale)."""
    down = sd[f"{module}.lora_down.weight"].to(torch.float32)
    up = sd[f"{module}.lora_up.weight"].to(torch.float32)
    scale = float(sd[f"{module}.alpha"]) / down.shape[0]
    inv = sd.get(f"{module}.inv_scale")
    if inv is not None:
        down = down * inv.to(torch.float32).unsqueeze(0)
    return up @ down * scale


def _make_sd(
    seed: int,
    modules=("lora_unet_blocks_0_attn_q_proj", "lora_unet_blocks_1_mlp_layer2"),
    alpha: float = 90.0,
    inv_scale: bool = False,
    down: torch.Tensor | None = None,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd: dict = {}
    for m in modules:
        sd[f"{m}.lora_down.weight"] = (
            down.clone() if down is not None else torch.randn(RANK, IN, generator=g)
        )
        sd[f"{m}.lora_up.weight"] = torch.randn(OUT, RANK, generator=g)
        sd[f"{m}.alpha"] = torch.tensor(alpha)
        if inv_scale:
            sd[f"{m}.inv_scale"] = torch.rand(IN, generator=g) + 0.5
    return sd


class TestSoupStateDicts:
    def test_delta_w_is_exact_average(self):
        sds = [_make_sd(0), _make_sd(1, inv_scale=True), _make_sd(2, alpha=32.0)]
        soup = soup_state_dicts(sds)
        for module in (
            "lora_unet_blocks_0_attn_q_proj",
            "lora_unet_blocks_1_mlp_layer2",
        ):
            want = torch.stack([_delta_w(sd, module) for sd in sds]).mean(0)
            torch.testing.assert_close(_delta_w(soup, module), want)
            # rank-concat: soup rank = sum of ingredient ranks, alpha = rank
            assert soup[f"{module}.lora_down.weight"].shape[0] == 3 * RANK
            assert float(soup[f"{module}.alpha"]) == 3 * RANK
            assert f"{module}.inv_scale" not in soup

    def test_weights_used_as_is(self):
        sds = [_make_sd(0), _make_sd(1)]
        soup = soup_state_dicts(sds, weights=[0.75, 0.25])
        module = "lora_unet_blocks_0_attn_q_proj"
        want = 0.75 * _delta_w(sds[0], module) + 0.25 * _delta_w(sds[1], module)
        torch.testing.assert_close(_delta_w(soup, module), want)

    def test_refuses_single_ingredient(self):
        with pytest.raises(ValueError, match=">= 2"):
            soup_state_dicts([_make_sd(0)])

    def test_refuses_hydra_suffix(self):
        sd = _make_sd(0)
        sd["lora_unet_blocks_0_attn_q_proj.lora_up_experts.0.weight"] = torch.zeros(1)
        with pytest.raises(ValueError, match="refused"):
            soup_state_dicts([sd, _make_sd(1)])

    def test_refuses_register_tokens(self):
        sd = _make_sd(0)
        sd["register_tokens"] = torch.zeros(4, 8)
        with pytest.raises(ValueError, match="not a plain-LoRA module key"):
            soup_state_dicts([sd, _make_sd(1)])

    def test_refuses_mismatched_module_sets(self):
        with pytest.raises(ValueError, match="module sets differ"):
            soup_state_dicts(
                [_make_sd(0), _make_sd(1, modules=("lora_unet_blocks_0_attn_q_proj",))]
            )


class TestTruncatedSoup:
    def test_shapes_alpha_and_rank(self):
        soup, energy = truncated_soup([_make_sd(0), _make_sd(1)], rank=RANK)
        module = "lora_unet_blocks_0_attn_q_proj"
        assert soup[f"{module}.lora_down.weight"].shape == (RANK, IN)
        assert soup[f"{module}.lora_up.weight"].shape == (OUT, RANK)
        assert float(soup[f"{module}.alpha"]) == RANK
        assert set(energy) == {
            "lora_unet_blocks_0_attn_q_proj",
            "lora_unet_blocks_1_mlp_layer2",
        }

    def test_lossless_on_shared_subspace(self):
        # Ingredients share the same down basis (the weight_svd-pinned-init
        # regime) -> the averaged ΔW is at most rank RANK -> truncation exact.
        down = torch.randn(RANK, IN, generator=torch.Generator().manual_seed(99))
        sds = [_make_sd(i, down=down) for i in range(3)]
        soup, energy = truncated_soup(sds, rank=RANK)
        assert all(e == pytest.approx(1.0, abs=1e-5) for e in energy.values())
        module = "lora_unet_blocks_0_attn_q_proj"
        want = torch.stack([_delta_w(sd, module) for sd in sds]).mean(0)
        torch.testing.assert_close(_delta_w(soup, module), want, rtol=1e-4, atol=1e-4)

    def test_truncation_is_best_rank_r(self):
        sds = [_make_sd(0), _make_sd(1), _make_sd(2)]
        soup, energy = truncated_soup(sds, rank=RANK)
        module = "lora_unet_blocks_0_attn_q_proj"
        avg = torch.stack([_delta_w(sd, module) for sd in sds]).mean(0)
        # Eckart-Young: residual energy == dropped singular-value energy.
        s = torch.linalg.svdvals(avg)
        want_energy = float((s[:RANK] ** 2).sum() / (s**2).sum())
        assert energy[module] == pytest.approx(want_energy, abs=1e-5)
        resid = torch.linalg.norm(avg - _delta_w(soup, module)) ** 2
        assert float(resid) == pytest.approx(float((s[RANK:] ** 2).sum()), rel=1e-4)


class TestPoolResolution:
    def _resized(self, tmp_path, artists):
        for a in artists:
            (tmp_path / a).mkdir()
        return str(tmp_path)

    def test_shard_contains_target(self, tmp_path):
        artists = [f"artist_{i:02d}" for i in range(10)]
        resized = self._resized(tmp_path, artists)
        pool = resolve_pool("artist_05", None, 4, resized_dir=resized)
        assert "artist_05" in pool
        # round-robin shard k=2 (index 5 % 4 + 1) of 4 over the sorted list
        assert pool == artists[1::4]

    def test_explicit_pool_adds_target(self, tmp_path):
        pool = resolve_pool("t", ["b", "a"], 4, resized_dir=str(tmp_path))
        assert pool == ["a", "b", "t"]

    def test_unknown_target_raises(self, tmp_path):
        resized = self._resized(tmp_path, ["a"])
        with pytest.raises(SystemExit, match="not an artist directory"):
            resolve_pool("missing", None, 4, resized_dir=resized)

    def test_shard_n_clamped_to_artist_count(self, tmp_path):
        resized = self._resized(tmp_path, ["a", "b"])
        assert resolve_pool("b", None, 16, resized_dir=resized) == ["b"]

    def test_pattern_and_name_determinism(self):
        assert pool_pattern(["a", "b"]) == "a/*|b/*"
        assert pool_pattern(["x[1]"]) == "x[[]1]/*"  # glob-escaped
        n1 = uncond_name(["b", "a"], 0.5, 2)
        assert n1 == uncond_name(["a", "b"], 0.5, 2)  # order-invariant
        assert n1.endswith("_r0p5_e2")
        assert n1 != uncond_name(["a", "c"], 0.5, 2)
