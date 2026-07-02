"""Register-token adapter invariants (networks/methods/register.py).

DSR-style registers (arXiv:2605.05206) on a frozen DiT. What must hold:

- **Insertion depth** (DSR Tab. 9 "starting block"): with ``insert_block=b``,
  blocks ``0..b-1`` run at the bare seq and blocks ``b..`` at ``seq+K`` — the
  registers enter at block b, not the stack entry — and the K rows are stripped
  before unpatchify (output shape matches the base forward).
- ``insert_block=0`` keeps the entry-insertion geometry (the RQ3 arm).
- **Step-0 no-op controls**: the QKV surface is zero-init, so arm L
  (``num_registers=0``) is bit-exact to the base forward.
- ``remove()`` restores the base forward bit-exactly.
- **Save/load roundtrip**: ``create_network_from_weights`` rebuilds
  ``insert_block`` and the lora-mode ``scale`` (= alpha/rank) from ``ss_*``
  metadata — a checkpoint trained with alpha != rank must reload at the trained
  strength, and a pre-``ss_insert_block`` checkpoint must default to entry
  insertion (0), not the new default (8).
"""

from __future__ import annotations

import torch

from library.anima.models import Anima
from networks.methods.register import RegisterNetwork, create_network_from_weights


def _tiny_anima(num_blocks: int = 4) -> Anima:
    model = Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    return model.eval().requires_grad_(False)


def _inputs(latent_h: int = 32, latent_w: int = 32):
    torch.manual_seed(0)
    x = torch.randn(1, 16, 1, latent_h, latent_w)
    timesteps = torch.tensor([0.5])
    crossattn_emb = torch.randn(1, 8, 64)
    return x, timesteps, crossattn_emb


def _net(unet, **kw) -> RegisterNetwork:
    defaults = dict(
        num_registers=4,
        arm="B",
        qkv_mode="unfrozen",
        target_blocks="2-3",
        insert_block=2,  # constructor default (8) targets the 28-block DiT
    )
    defaults.update(kw)
    return RegisterNetwork(unet, **defaults)


@torch.no_grad()
def test_mid_stack_insertion_and_strip():
    model = _tiny_anima()
    inp = _inputs()
    base = model.forward_mini_train_dit(*inp)

    net = _net(model, insert_block=2)
    net.apply_to(None, model)
    seq_seen = {}
    spies = [
        blk.register_forward_hook(
            lambda m, a, out, bi=bi: seq_seen.__setitem__(bi, out.shape[2])
        )
        for bi, blk in enumerate(model.blocks)
    ]
    out = model.forward_mini_train_dit(*inp)
    for h in spies:
        h.remove()

    L = (32 // 2) * (32 // 2)  # patch_spatial=2
    assert seq_seen == {0: L, 1: L, 2: L + 4, 3: L + 4}
    assert out.shape == base.shape  # registers stripped before unpatchify
    assert net.last_reg_ratio is not None and net.last_patch_sink_ratio is not None

    net.remove()
    assert torch.equal(model.forward_mini_train_dit(*inp), base)


@torch.no_grad()
def test_entry_insertion_geometry():
    model = _tiny_anima()
    inp = _inputs()
    base = model.forward_mini_train_dit(*inp)

    net = _net(model, insert_block=0)
    net.apply_to(None, model)
    seq_seen = {}
    spies = [
        blk.register_forward_hook(
            lambda m, a, out, bi=bi: seq_seen.__setitem__(bi, out.shape[2])
        )
        for bi, blk in enumerate(model.blocks)
    ]
    out = model.forward_mini_train_dit(*inp)
    for h in spies:
        h.remove()
    net.remove()

    L = (32 // 2) * (32 // 2)
    assert seq_seen == {bi: L + 4 for bi in range(4)}
    assert out.shape == base.shape


@torch.no_grad()
def test_arm_L_is_step0_noop():
    model = _tiny_anima()
    inp = _inputs()
    base = model.forward_mini_train_dit(*inp)
    net = _net(model, num_registers=0, insert_block=2)
    net.apply_to(None, model)
    assert torch.equal(model.forward_mini_train_dit(*inp), base)
    net.remove()


def test_gradients_reach_registers_through_frozen_prefix():
    model = _tiny_anima()
    inp = _inputs()
    net = _net(model, insert_block=2)
    net.apply_to(None, model)
    model.forward_mini_train_dit(*inp).sum().backward()
    assert net.register.grad is not None and net.register.grad.abs().sum() > 0
    for surface in net.qkv.values():
        assert all(p.grad is not None for p in surface.parameters())
    net.remove()


def test_down_init_weight_svd():
    import pytest

    model = _tiny_anima()
    inp = _inputs()
    with torch.no_grad():
        base = model.forward_mini_train_dit(*inp)

    net = _net(
        model, qkv_mode="lora", lora_rank=4, down_init="weight_svd", num_registers=0
    )
    # top-r right-singular directions → orthonormal rows
    for bi in net.target_blocks:
        down = net.qkv[str(bi)].down
        gram = down @ down.T
        assert torch.allclose(gram, torch.eye(4), atol=1e-4)
        # aligned with the base weight's principal input subspace: projecting W
        # onto the r directions must capture more energy than r random ones
        w = model.blocks[bi].self_attn.qkv_proj.weight.float()
        svd_energy = (w @ down.T).norm()
        rand = torch.linalg.qr(torch.randn(w.shape[1], 4)).Q.T
        assert svd_energy > (w @ rand.T).norm()
    # up is zero-init either way → still a step-0 no-op
    net.apply_to(None, model)
    with torch.no_grad():
        assert torch.equal(model.forward_mini_train_dit(*inp), base)
    net.remove()

    with pytest.raises(ValueError, match="down_init"):
        _net(model, qkv_mode="lora", down_init="nope")
    with pytest.raises(ValueError, match="only applies"):
        _net(model, qkv_mode="unfrozen", down_init="weight_svd")


def test_from_weights_restores_scale_and_insert_block(tmp_path):
    from safetensors.torch import save_file

    model = _tiny_anima()
    # lora mode with alpha != rank → scale 0.5 must survive the roundtrip
    net = _net(model, qkv_mode="lora", lora_rank=4, lora_alpha=2.0, insert_block=1)
    assert net.scale == 0.5
    path = str(tmp_path / "adapter.safetensors")
    save_file(net.state_dict(), path, metadata=net.metadata_fields())

    net2 = create_network_from_weights(1.0, path, None, None, _tiny_anima())
    assert net2.scale == 0.5
    assert net2.insert_block == 1
    assert net2.qkv_mode == "lora"
    assert net2.K == 4

    # pre-insert_block checkpoint (no ss_insert_block stamp) → entry insertion
    meta_legacy = {
        k: v for k, v in net.metadata_fields().items() if k != "ss_insert_block"
    }
    legacy = str(tmp_path / "legacy.safetensors")
    save_file(net.state_dict(), legacy, metadata=meta_legacy)
    net3 = create_network_from_weights(1.0, legacy, None, None, _tiny_anima())
    assert net3.insert_block == 0
