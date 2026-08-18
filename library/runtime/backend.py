"""Runtime accelerator compatibility helpers."""

from __future__ import annotations

from typing import Any


def is_rocm(torch_module: Any) -> bool:
    """Return whether *torch_module* is a ROCm/HIP PyTorch build."""
    return getattr(getattr(torch_module, "version", None), "hip", None) is not None


def needs_rocm_attention_fallback(
    requested: str | None, torch_module: Any
) -> bool:
    """Return whether an explicit Flash request must fall back on ROCm."""
    return requested == "flash" and is_rocm(torch_module)


def resolve_attention_mode(requested: str | None, torch_module: Any) -> str:
    """Use PyTorch SDPA when the CUDA-only Flash backend is selected on ROCm."""
    mode = requested or "torch"
    if needs_rocm_attention_fallback(requested, torch_module):
        return "torch"
    return mode
