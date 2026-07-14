"""Re-noising, τ samplers, and shared loop-side helpers.

These were the first copy of the distillation per-step primitives; they have
since been promoted to ``library/training`` and ``library/datasets`` so the
other distillation loops (``scripts/distill_mod``)
share one implementation. This module is now a thin compatibility shim — the
turbo loop imports the same names from here as before.
"""

from __future__ import annotations

import torch

from library.datasets.cache import make_cached_collate as make_collate
from library.training.forward import PadCache, renoise
from library.training.forward import sample_sigma as sample_t
from library.training.schedulers import make_warmup_cosine_scheduler

__all__ = [
    "renoise",
    "sample_t",
    "sample_t_routed",
    "make_scheduler",
    "PadCache",
    "make_collate",
]


def sample_t_routed(
    B: int,
    *,
    turbo,
    fake_tau_banks: int,
    fake_tau_boundary: float,
    distribution: str,
    sigmoid_scale: float,
    device,
    dtype,
) -> torch.Tensor:
    """Draw τ for a fake update/query and arm the owner fake bank (τ-split critic).

    ``fake_tau_banks <= 1`` → a plain on-device :func:`sample_t`, byte-identical
    to the pre-bank loop (same RNG stream, no extra work). With 2 banks the draw
    happens on CPU (B=1, enforced at config resolve) so the routing comparison
    costs no GPU sync; the owner bank is selected via ``turbo.set_fake_bank``
    (bank 0 owns τ < boundary) and τ is then moved to ``device``. Callers must
    invoke this BEFORE the fake forward the τ conditions.
    """
    if fake_tau_banks <= 1:
        return sample_t(
            B,
            distribution=distribution,
            sigmoid_scale=sigmoid_scale,
            device=device,
            dtype=dtype,
        )
    tau_cpu = sample_t(
        B,
        distribution=distribution,
        sigmoid_scale=sigmoid_scale,
        device="cpu",
        dtype=torch.float32,
    )
    turbo.set_fake_bank(0 if float(tau_cpu) < fake_tau_boundary else 1)
    return tau_cpu.to(device=device, dtype=dtype)


def make_scheduler(opt, total_steps: int, lr: float):
    """Warmup (2% of ``total_steps``, ≥1 step) → cosine annealing to ``0.1·lr``.

    Cosine is the only shape: the ``lr_schedule="constant"`` variant was tried
    (superturbo_B2) and closed — it never settles (full-magnitude updates to the
    last step, ~5-10x the annealed per-step displacement) and rendered worse
    than the cosine twin. The tail is settling, not dead time.
    """
    warmup_steps = max(1, int(0.02 * total_steps))
    return make_warmup_cosine_scheduler(
        opt,
        total_steps,
        lr,
        warmup_steps=warmup_steps,
        eta_min_ratio=0.1,
    )
