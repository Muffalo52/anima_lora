"""
Tests for optimizer-relevant parameter attributes on sd_scripts network modules.

Verifies that ``tag_lora_module_params()`` (from ``networks.network_base``)
correctly sets the following attributes on ``nn.Parameter`` objects so that
Advanced_Optimizers can identify each parameter's role:

    _is_dora_scale  -> DoRA magnitude scale
    _is_oft         -> OFT skew-symmetric blocks
    _is_lora_A      -> LoRA down/A factor
    _is_lora_B      -> LoRA up/B factor
    is_hidden       -> generic 2D hidden-layer weight
    is_vector       -> logically-vector parameter (multi-dim)
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

# Ensure the sd_scripts root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.network_base import tag_lora_module_params


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLoRAModule(nn.Module):
    """Minimal module that mimics sd_scripts LoRAModule structure."""
    def __init__(self, in_dim=64, out_dim=32, lora_dim=4):
        super().__init__()
        self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)


class FakeOFTModule(nn.Module):
    """Minimal module that mimics sd_scripts OFTModule structure."""
    def __init__(self, num_blocks=8, block_size=4):
        super().__init__()
        self.oft_blocks = nn.Parameter(torch.zeros(num_blocks, block_size, block_size))


class FakeDoRALoRAModule(nn.Module):
    """Minimal module with lora_down, lora_up, and dora_scale."""
    def __init__(self, in_dim=64, out_dim=32, lora_dim=4):
        super().__init__()
        self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        self.dora_scale = nn.Parameter(torch.ones(out_dim))


class FakeSplitLoRAModule(nn.Module):
    """Minimal module with ModuleList lora_down/lora_up (flux split dims)."""
    def __init__(self, in_dim=64, split_dims=(16, 16), lora_dim=4):
        super().__init__()
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_dim, bias=False) for _ in split_dims])
        self.lora_up = nn.ModuleList([nn.Linear(lora_dim, d, bias=False) for d in split_dims])


class FakeFluxOFTModule(nn.Module):
    """Minimal module with oft_blocks as ParameterList (flux OFT)."""
    def __init__(self):
        super().__init__()
        self.oft_blocks = nn.ParameterList([
            nn.Parameter(torch.zeros(8, 4, 4)),
            nn.Parameter(torch.zeros(8, 4, 4)),
        ])


# ---------------------------------------------------------------------------
# 1. Standard LoRA: lora_down -> A, lora_up -> B
# ---------------------------------------------------------------------------

class TestLoRATagging:
    def test_lora_down_is_lora_a(self):
        mod = FakeLoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_down.weight, "is_hidden", False) is True

    def test_lora_up_is_lora_b(self):
        mod = FakeLoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True
        assert getattr(mod.lora_up.weight, "is_hidden", False) is True

    def test_no_dora_scale(self):
        mod = FakeLoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert not hasattr(mod, "dora_scale")


# ---------------------------------------------------------------------------
# 2. OFT: oft_blocks tagged
# ---------------------------------------------------------------------------

class TestOFTTagging:
    def test_oft_blocks_tagged(self):
        mod = FakeOFTModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert getattr(mod.oft_blocks, "_is_oft", False) is True

    def test_oft_not_lora(self):
        mod = FakeOFTModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert getattr(mod.oft_blocks, "_is_lora_A", False) is False
        assert getattr(mod.oft_blocks, "_is_lora_B", False) is False


# ---------------------------------------------------------------------------
# 3. DoRA: dora_scale tagged
# ---------------------------------------------------------------------------

class TestDoRATagging:
    def test_dora_scale_tagged(self):
        mod = FakeDoRALoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert getattr(mod.dora_scale, "_is_dora_scale", False) is True
        assert getattr(mod.dora_scale, "is_vector", False) is True

    def test_lora_factors_still_tagged_with_dora(self):
        mod = FakeDoRALoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.lora_up.weight, "_is_lora_B", False) is True


# ---------------------------------------------------------------------------
# 4. Split LoRA (ModuleList): lora_down.0 -> A, lora_up.0 -> B, etc.
# ---------------------------------------------------------------------------

class TestSplitLoRATagging:
    def test_split_down_tagged(self):
        mod = FakeSplitLoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        for i, sub in enumerate(mod.lora_down):
            assert getattr(sub.weight, "_is_lora_A", False) is True
            assert getattr(sub.weight, "is_hidden", False) is True

    def test_split_up_tagged(self):
        mod = FakeSplitLoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        for i, sub in enumerate(mod.lora_up):
            assert getattr(sub.weight, "_is_lora_B", False) is True
            assert getattr(sub.weight, "is_hidden", False) is True


# ---------------------------------------------------------------------------
# 5. Flux OFT (ParameterList): all blocks tagged
# ---------------------------------------------------------------------------

class TestFluxOFTTagging:
    def test_all_blocks_tagged(self):
        mod = FakeFluxOFTModule().to(DEVICE)
        tag_lora_module_params(mod)
        for p in mod.oft_blocks:
            assert getattr(p, "_is_oft", False) is True


# ---------------------------------------------------------------------------
# 6. Idempotent
# ---------------------------------------------------------------------------

class TestIdempotent:
    def test_double_call_safe(self):
        mod = FakeDoRALoRAModule().to(DEVICE)
        tag_lora_module_params(mod)
        tag_lora_module_params(mod)
        assert getattr(mod.lora_down.weight, "_is_lora_A", False) is True
        assert getattr(mod.dora_scale, "_is_dora_scale", False) is True


# ---------------------------------------------------------------------------
# 7. Fallback: all 2D params get is_hidden
# ---------------------------------------------------------------------------

class TestFallbackHidden:
    def test_2d_params_is_hidden(self):
        """Every 2D trainable parameter should have is_hidden=True."""
        for mod_class in (FakeLoRAModule, FakeOFTModule, FakeDoRALoRAModule, FakeSplitLoRAModule):
            mod = mod_class().to(DEVICE)
            tag_lora_module_params(mod)
            for p in mod.parameters():
                if p.ndim >= 2:
                    has_role = (
                        getattr(p, "_is_oft", False)
                        or getattr(p, "_is_lora_A", False)
                        or getattr(p, "_is_lora_B", False)
                        or getattr(p, "_is_dora_scale", False)
                        or getattr(p, "is_hidden", False)
                    )
                    assert has_role, (
                        f"{mod_class.__name__}: 2D param with shape {tuple(p.shape)} "
                        f"has no role attribute"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])