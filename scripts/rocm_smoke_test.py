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

    # Exercise device allocation, compile, SDPA, and backward rather than
    # accepting an import-only success, which misses runtime/device/Triton
    # failures and the attention path used by Anima on ROCm.
    @torch.compile
    def compiled_loss(value: torch.Tensor) -> torch.Tensor:
        return value.square().mean()

    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    compiled_loss(x).backward()

    q = torch.randn(
        2, 4, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    attention = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    attention.float().square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(attention).all() or not torch.isfinite(q.grad).all():
        raise RuntimeError("ROCm PyTorch SDPA produced non-finite values")

    print("ROCm tensor/compile/SDPA/backward smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
