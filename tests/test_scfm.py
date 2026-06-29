"""SCFM (velocity-space self-distillation) wiring + invariant tests.

Covers the load-bearing pieces of the Phase-1 port (docs/proposal/turbo_scfm.md):

  - Config resolution + the ``[scfm]`` knob precedence, and the validation gates
    that protect the EMA-as-fake-stack contract (rank match, grid >= steps,
    incompatible objectives force-inert, dual_ema/per_step_expert/ortho refused).
  - The EMA bookkeeping math on ``TurboDMDNetwork`` (``reset_ema`` copies θ←θ,
    ``update_ema`` computes μθ⁻+(1−μ)θ), exercised on lightweight stand-in stacks
    so no real DiT is needed; plus the shape-mismatch guard.
  - The masked / unmasked velocity-MSE helper used as the SCFM student loss.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from networks.methods.turbo_dmd import TurboDMDNetwork
from scripts.distill_turbo.config import build_argparser, resolve_config
from scripts.distill_turbo.scfm import _velocity_mse


def _resolve(cli: list[str] | None = None, toml: dict | None = None):
    args = build_argparser().parse_args(cli or [])
    return resolve_config(args, toml or {})


# ── config resolution ─────────────────────────────────────────────────────────


def test_scfm_defaults_and_inert_when_dpdmd():
    # Off-path: the scfm knobs resolve to defaults but base_loss stays dpdmd.
    c = _resolve()
    assert c.base_loss == "dpdmd"
    assert c.scfm_k_ratio == 0.4
    assert c.scfm_ema_mu == 0.999
    assert c.scfm_ema_restart == 1000
    assert c.scfm_dual_ema is False
    assert c.scfm_n_consistency_grid == 8


def test_scfm_toml_and_cli_precedence():
    toml = {
        "scfm": {
            "k_ratio": 0.5,
            "ema_mu": 0.99,
            "ema_restart": 500,
            "n_consistency_grid": 12,
        }
    }
    c = _resolve(["--base_loss", "scfm"], toml=toml)
    assert c.base_loss == "scfm"
    assert c.scfm_k_ratio == 0.5
    assert c.scfm_ema_mu == 0.99
    assert c.scfm_ema_restart == 500
    assert c.scfm_n_consistency_grid == 12
    # CLI sentinels win over TOML.
    c = _resolve(
        ["--base_loss", "scfm", "--scfm_k_ratio", "0.25", "--scfm_ema_mu", "0.9"],
        toml=toml,
    )
    assert c.scfm_k_ratio == 0.25
    assert c.scfm_ema_mu == 0.9


def test_grad_accum_defaults_to_one():
    assert _resolve().gradient_accumulation_steps == 1


def test_grad_accum_toml_and_cli_under_scfm():
    c = _resolve(
        ["--base_loss", "scfm"], toml={"optim": {"gradient_accumulation_steps": 4}}
    )
    assert c.gradient_accumulation_steps == 4
    # CLI sentinel wins over TOML.
    c = _resolve(
        ["--base_loss", "scfm", "--gradient_accumulation_steps", "8"],
        toml={"optim": {"gradient_accumulation_steps": 4}},
    )
    assert c.gradient_accumulation_steps == 8


def test_grad_accum_rejected_for_dpdmd():
    # Only the scfm loop honors accumulation; dpdmd/dmd must fail loud.
    with pytest.raises(ValueError, match="only.*scfm"):
        _resolve(["--base_loss", "dpdmd", "--gradient_accumulation_steps", "4"])
    with pytest.raises(ValueError, match="only.*scfm"):
        _resolve(["--base_loss", "dmd", "--gradient_accumulation_steps", "2"])


def test_term_b_point_default_and_override():
    assert _resolve().scfm_term_b_point == "renoise"  # paper-faithful default
    c = _resolve(["--base_loss", "scfm"], toml={"scfm": {"term_b_point": "rollout"}})
    assert c.scfm_term_b_point == "rollout"
    # CLI wins over TOML.
    c = _resolve(
        ["--base_loss", "scfm", "--scfm_term_b_point", "renoise"],
        toml={"scfm": {"term_b_point": "rollout"}},
    )
    assert c.scfm_term_b_point == "renoise"


def test_term_b_point_invalid_via_toml_raises():
    # argparse `choices` guards the CLI; route a bad value through TOML to hit the
    # resolve_config guard.
    with pytest.raises(ValueError, match="term_b_point"):
        _resolve(["--base_loss", "scfm"], toml={"scfm": {"term_b_point": "wat"}})


def test_scfm_forces_other_objectives_inert():
    # The shipped turbo.toml carries gan/f_div for the dpdmd default; under scfm
    # they must be force-inert (warned), not error.
    toml = {
        "gan": {"weight_gen": 0.03},
        "f_distill": {"f_div": "kl"},
        "repa": {"weight": 0.05},
        "softrank": {"weight": 0.05},
        "mean_var": {"weight": 0.02},
    }
    c = _resolve(["--base_loss", "scfm"], toml=toml)
    assert c.gan_loss_weight_gen == 0.0
    assert c.f_div == "rkl"
    assert c.repa_weight == 0.0
    assert c.softrank_weight == 0.0
    assert c.mean_var_weight == 0.0


@pytest.mark.parametrize(
    "cli",
    [
        ["--base_loss", "scfm", "--fake_rank", "48", "--student_rank", "96"],  # rank
        [
            "--base_loss",
            "scfm",
            "--scfm_n_consistency_grid",
            "2",
            "--student_steps",
            "4",
        ],
        ["--base_loss", "scfm", "--scfm_k_ratio", "1.0"],  # k_ratio out of (0,1)
        ["--base_loss", "scfm", "--scfm_k_ratio", "0.0"],
        ["--base_loss", "scfm", "--scfm_dual_ema"],  # Phase-2, not implemented
        ["--base_loss", "scfm", "--per_step_expert"],  # single field only
    ],
)
def test_scfm_validation_raises(cli):
    with pytest.raises(ValueError):
        _resolve(cli)


def test_scfm_ortho_init_refused():
    with pytest.raises(ValueError):
        _resolve(
            ["--base_loss", "scfm"], toml={"network": {"student_ortho_init": True}}
        )


def test_base_loss_unknown_raises():
    # argparse `choices` rejects a bad CLI value; route it through TOML to reach
    # the resolve_config guard.
    with pytest.raises(ValueError):
        _resolve(toml={"base_loss": "bogus"})


# ── EMA bookkeeping (math) ─────────────────────────────────────────────────────


class _StubTurbo(TurboDMDNetwork):
    """Minimal stand-in exposing ``.student`` / ``.fake`` as nn.Modules so the
    inherited EMA methods (which only touch those two) run without a real DiT.

    Skips ``TurboDMDNetwork.__init__`` (which needs a DiT) deliberately.
    """

    def __init__(self, student: nn.Module, fake: nn.Module):
        self.student = student
        self.fake = fake


def _two_stacks(seed_s: float, seed_f: float):
    s = nn.Linear(4, 4, bias=False)
    f = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        s.weight.fill_(seed_s)
        f.weight.fill_(seed_f)
    return _StubTurbo(s, f)


def test_reset_ema_copies_student_to_fake():
    stub = _two_stacks(seed_s=2.0, seed_f=-1.0)
    TurboDMDNetwork.reset_ema(stub)
    assert torch.allclose(stub.fake.weight, stub.student.weight)
    assert torch.allclose(stub.fake.weight, torch.full((4, 4), 2.0))


def test_update_ema_is_decayed_average():
    mu = 0.9
    stub = _two_stacks(seed_s=1.0, seed_f=5.0)  # θ=1, θ⁻=5
    TurboDMDNetwork.update_ema(stub, mu)
    # θ⁻ ← μ·5 + (1−μ)·1 = 0.9·5 + 0.1·1 = 4.6
    assert torch.allclose(stub.fake.weight, torch.full((4, 4), 4.6), atol=1e-6)
    # student is untouched by the EMA update.
    assert torch.allclose(stub.student.weight, torch.full((4, 4), 1.0))


def test_update_ema_does_not_build_graph():
    stub = _two_stacks(seed_s=1.0, seed_f=2.0)
    stub.student.weight.requires_grad_(True)
    TurboDMDNetwork.update_ema(stub, 0.99)
    # no_grad decorator ⇒ the EMA target never carries a grad_fn.
    assert stub.fake.weight.grad_fn is None


def test_ema_pairing_shape_mismatch_raises():
    stub = _StubTurbo(nn.Linear(4, 4, bias=False), nn.Linear(4, 8, bias=False))
    with pytest.raises(RuntimeError):
        TurboDMDNetwork.reset_ema(stub)


# ── velocity-MSE loss helper ───────────────────────────────────────────────────


def test_velocity_mse_unmasked_matches_mse():
    v = torch.randn(2, 16, 8, 8)
    t = torch.randn(2, 16, 8, 8)
    got = _velocity_mse(v, t, None)
    assert torch.allclose(got, ((v - t) ** 2).mean())


def test_velocity_mse_masked_is_region_mean():
    B, C, H, W = 1, 16, 4, 4
    v = torch.zeros(B, C, H, W)
    t = torch.full((B, C, H, W), 3.0)  # sqerr = 9 everywhere
    mask = torch.zeros(B, 1, H, W)
    mask[..., :2, :] = 1.0  # keep half the spatial cells
    got = _velocity_mse(v, t, mask)
    # Per-kept-element mean = 9 (constant error), independent of mask fraction.
    assert torch.allclose(got, torch.tensor(9.0), atol=1e-5)
