#!/usr/bin/env python3
"""Small post-install check for the supported Windows ROCm path."""

from __future__ import annotations

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access an AMD GPU")
    if torch.version.hip is None:
        raise RuntimeError("installed PyTorch is not a ROCm build")

    # Exercise device allocation, compile, and backward rather than accepting
    # an import-only success, which misses runtime/device/Triton failures.
    @torch.compile
    def compiled_loss(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean()

    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    compiled_loss(x).backward()
    torch.cuda.synchronize()
    print("ROCm tensor/compile/backward smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
