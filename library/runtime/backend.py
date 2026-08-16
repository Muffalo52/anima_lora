"""Runtime accelerator compatibility helpers."""

from __future__ import annotations

from typing import Any


def is_rocm(torch_module: Any) -> bool:
    """Return whether *torch_module* is a ROCm/HIP PyTorch build."""
    return getattr(getattr(torch_module, "version", None), "hip", None) is not None


def resolve_attention_mode(requested: str | None, torch_module: Any) -> str:
    """Use PyTorch SDPA when the CUDA-only Flash backend is selected on ROCm."""
    mode = requested or "torch"
    if mode == "flash" and is_rocm(torch_module):
        return "torch"
    return mode
