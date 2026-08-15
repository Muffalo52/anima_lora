"""Probe bank for the set-level attention loss — real DiT K/V, no DiT load.

Two verified facts about how the DiT consumes the adapter output fix the shape
of this loss (``library/anima/models.py``):

1. The DiT's cross-attention applies **no RoPE to the context** (rope is gated
   on ``is_selfattn``, :385), so the text side is consumed
   **permutation-invariantly** — position-wise matching asks for more than the
   model can observe.
2. Padded positions are zeroed after the adapter, never masked out
   (``crossattn_emb[~mask] = 0``, inference/text.py:229) — the CLAUDE.md
   attention-sink invariant. With ``k_norm(0) = 0`` those ``S - N`` rows are
   sink mass at logit 0 carrying zero value, so the **number of real tokens is
   itself part of the conditioning**, and teacher and student do not have the
   same number of them.

So the faithful objective is: match the attention *readout* over
``{N real vectors} ∪ {S - N zeros}``. The sink term is folded into the softmax
denominator analytically rather than materializing 512-row tensors.

The K/V projections and the q/k RMSNorm gains are read straight out of the DiT
safetensors (``net.blocks.<i>.cross_attn.*``) — no model instantiation. Probe
queries are seeded random directions scaled to the real query RMS (the q_norm
gain), because that scale is what sets the softmax temperature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

RMS_EPS = 1e-6  # matches models.py::RMSNorm(eps=1e-6) on q_norm/k_norm


@dataclass
class BlockProbe:
    block: int
    k_weight: torch.Tensor  # [inner, ctx_dim]
    v_weight: torch.Tensor  # [inner, ctx_dim]
    k_gain: torch.Tensor  # [head_dim]
    queries: torch.Tensor  # [heads, n_query, head_dim] — already q_norm'ed


def _rms_norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * gain


def build_bank(
    dit_path: str,
    blocks: tuple[int, ...],
    *,
    n_query: int = 64,
    seed: int = 0,
    device=None,
    dtype=torch.float32,
) -> list[BlockProbe]:
    from safetensors import safe_open

    probes: list[BlockProbe] = []
    g = torch.Generator().manual_seed(seed)
    with safe_open(dit_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = "net." if any(k.startswith("net.blocks.") for k in keys) else ""
        for b in blocks:
            base = f"{prefix}blocks.{b}.cross_attn."
            missing = [
                s
                for s in (
                    "k_proj.weight",
                    "v_proj.weight",
                    "k_norm.weight",
                    "q_norm.weight",
                )
                if base + s not in keys
            ]
            if missing:
                raise KeyError(f"block {b}: missing {missing} in {dit_path}")
            kw = f.get_tensor(base + "k_proj.weight").to(dtype)
            vw = f.get_tensor(base + "v_proj.weight").to(dtype)
            k_gain = f.get_tensor(base + "k_norm.weight").to(dtype)
            q_gain = f.get_tensor(base + "q_norm.weight").to(dtype)
            head_dim = k_gain.shape[0]
            heads = kw.shape[0] // head_dim
            q = torch.randn(heads, n_query, head_dim, generator=g, dtype=torch.float32)
            q = _rms_norm(q, q_gain.float()).to(dtype)
            probes.append(
                BlockProbe(
                    block=b,
                    k_weight=kw.to(device),
                    v_weight=vw.to(device),
                    k_gain=k_gain.to(device),
                    queries=q.to(device),
                )
            )
            logger.info(
                "attn bank: block %d — %d heads × %d dim, %d probe queries",
                b,
                heads,
                head_dim,
                n_query,
            )
    return probes


def readout(
    x: torch.Tensor,
    mask: torch.Tensor,
    probe: BlockProbe,
    *,
    seq_total: int = 512,
) -> torch.Tensor:
    """Cross-attention readout of ``x`` under ``probe``: ``[B, heads, n_query, head_dim]``.

    ``mask`` marks the real (non-pad) rows of ``x``; the remaining
    ``seq_total - mask.sum()`` rows the DiT would see are zeros, folded into
    the denominator as ``n_sink * exp(0)``.
    """
    head_dim = probe.k_gain.shape[0]
    B, L, _ = x.shape
    xf = x.float()
    k = (xf @ probe.k_weight.float().T).view(B, L, -1, head_dim)
    v = (xf @ probe.v_weight.float().T).view(B, L, -1, head_dim)
    k = _rms_norm(k, probe.k_gain.float())

    q = probe.queries.float()  # [H, M, hd]
    # [B, H, M, L]
    logits = torch.einsum("hmd,blhd->bhml", q, k) / (head_dim**0.5)
    valid = mask.bool()[:, None, None, :]
    logits = logits.masked_fill(~valid, float("-inf"))

    n_sink = (seq_total - mask.sum(dim=1)).clamp(min=0).float()  # [B]
    # Stable softmax including the zero-logit sink block.
    m = torch.maximum(
        logits.masked_fill(~valid, float("-inf")).amax(dim=-1),
        torch.zeros(1, device=x.device),
    )  # [B, H, M]
    w = torch.exp(logits - m[..., None]).masked_fill(~valid, 0.0)
    den = w.sum(dim=-1) + n_sink[:, None, None] * torch.exp(-m)
    out = torch.einsum("bhml,blhd->bhmd", w, v)
    return out / den[..., None].clamp_min(1e-12)
