#!/usr/bin/env python3
"""Artist atlas — pack N plain artist LoRAs into one StackedExperts checkpoint.

Offline assembly, no training (docs/proposal/artist_atlas_pack.md): each
ingredient fills one expert slot of the independent-A StackedExperts layout
(``lora_down_weight (E, R, in)`` / ``lora_up_weight (E, out, R)``), routing is
a per-request gate written into ``_routing_weights`` (one-hot = solo artist,
zero = base model, convex blend = style mix). ``router_source="none"`` is
stamped so no GlobalRouter is built at load — nothing sets gates per step, and
the loaded default is **uniform 1/E** (= the ingredient average), so callers
must set gates explicitly.

Exactness contract: expert *e* hard-selected must reproduce ingredient *e*'s
ΔW bit-for-bit (up to the save dtype). Two ingredient shapes exist per attn
group on disk, and both are matched to the fused runtime exactly the way the
plain-LoRA loader does it (``_refuse_unfused_attn_lora_keys``):

  * pre-fused saves (direct train.py artifacts): q/k/v downs are clones of one
    fused rank-r module → fused rank stays r;
  * per-component saves (soup outputs — the soup SVD runs per split module, so
    q/k/v downs differ): block-diagonal fuse → fused rank n·r.

Ranks are then aligned across ingredients per module by zero-padding to the
max (zero rows in down / zero cols in up — an exact no-op), NOT by SVD
truncation. alpha is set to the padded rank (scale 1); alpha/inv_scale of the
ingredients are folded into the factors first (mirrors the soup builder).

Plain-LoRA ingredients only — hydra/chimera/stacked/ortho are refused loudly,
same rule as ``scripts/soup/build.py``.

Usage
-----
    python -m scripts.atlas.pack \
        --ckpts output/ckpt/soup/anima_soup_a.safetensors \
                output/ckpt/soup/anima_soup_b.safetensors \
        --tags artist_a artist_b \
        --out output/ckpt/anima_atlas2_moe.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from networks.lora_anima.loading import _refuse_unfused_attn_lora_keys
from networks.lora_modules.stacked_experts import StackedExpertsLoRAModule
from scripts.soup.build import _effective_factors, _group_modules


def _fused_effective_modules(
    sd: dict, path: str
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Plain-LoRA state dict → ``{fused_module: (down (R, in), up (out, R))}``.

    ΔW = up @ down exactly (alpha and inv_scale folded, fp32), with the attn
    q/k/v groups fused to the runtime layout via the same code path the
    plain-LoRA loader uses, so the atlas reproduces exactly what loading the
    ingredient solo would apply.
    """
    grouped = _group_modules(sd, path)  # validates plain-LoRA suffixes
    flat: dict = {}
    for module, mod in grouped.items():
        down, up = _effective_factors(mod, module, path)
        flat[f"{module}.lora_down.weight"] = down
        flat[f"{module}.lora_up.weight"] = up
        # Scale is folded into up → alpha = rank keeps the runtime scale at 1,
        # and equal alphas across q/k/v let the loader's pre-fused detection
        # (cloned downs) keep rank r instead of inflating to n·r.
        flat[f"{module}.alpha"] = torch.tensor(float(down.shape[0]))
    _refuse_unfused_attn_lora_keys(flat)

    fused: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for module, mod in _group_modules(flat, path).items():
        down = mod["lora_down.weight"]
        up = mod["lora_up.weight"]
        alpha = mod.get("alpha")
        rank = down.shape[0]
        # Both refuse branches preserve scale 1 (pre-fused keeps alpha=r; the
        # block-diagonal path folds per-component scales and sets alpha=n·r).
        if alpha is not None and float(alpha) != float(rank):
            raise AssertionError(
                f"{path}: {module} fused to alpha {float(alpha)} != rank {rank} — "
                "scale folding drifted; refusing to pack an inexact expert."
            )
        fused[module] = (down, up)
    return fused


def pack_state_dicts(
    sds: List[dict],
    *,
    paths: Optional[List[str]] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> dict:
    """Build the runtime-form atlas state dict from N plain-LoRA state dicts.

    Returns ``{module}.lora_down_weight (E, R, in)`` / ``.lora_up_weight
    (E, out, R)`` / ``.alpha = R`` keys (expert order = ingredient order).
    Use :meth:`StackedExpertsLoRAModule.build_moe_state_dict` to expand to the
    canonical on-disk split layout.
    """
    n = len(sds)
    if n < 2:
        raise ValueError("Need >= 2 ingredient checkpoints to pack an atlas.")
    paths = paths or [f"<sd {i}>" for i in range(n)]
    fused = [_fused_effective_modules(sd, p) for sd, p in zip(sds, paths)]

    module_names = set(fused[0])
    for f, p in zip(fused[1:], paths[1:]):
        if set(f) != module_names:
            missing = module_names.symmetric_difference(f)
            raise ValueError(
                f"Ingredient module sets differ ({p}): {sorted(missing)[:5]}... — "
                "all ingredients must come from the same recipe."
            )

    dtype = out_dtype or sds[0][next(iter(sds[0]))].dtype
    atlas: dict = {}
    for module in sorted(module_names):
        factors = [f[module] for f in fused]
        in_dim = factors[0][0].shape[1]
        out_dim = factors[0][1].shape[0]
        for (down, up), p in zip(factors, paths):
            if down.shape[1] != in_dim or up.shape[0] != out_dim:
                raise ValueError(
                    f"{p}: {module} has (in, out) = ({down.shape[1]}, "
                    f"{up.shape[0]}), expected ({in_dim}, {out_dim}) — "
                    "ingredients target different base models."
                )
        rank = max(down.shape[0] for down, _ in factors)
        down_stack = torch.zeros(n, rank, in_dim, dtype=torch.float32)
        up_stack = torch.zeros(n, out_dim, rank, dtype=torch.float32)
        for e, (down, up) in enumerate(factors):
            r = down.shape[0]
            down_stack[e, :r] = down
            up_stack[e, :, :r] = up
        atlas[f"{module}.lora_down_weight"] = down_stack.to(dtype)
        atlas[f"{module}.lora_up_weight"] = up_stack.to(dtype)
        # alpha = padded rank → runtime scale 1 (ingredient scales are folded).
        atlas[f"{module}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
    return atlas


def default_tag(ckpt: str) -> str:
    """Ingredient filename → artist tag (strip the soup/atlas name prefixes)."""
    stem = Path(ckpt).stem
    for prefix in ("anima_soup_", "anima_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def atlas_metadata(ckpts: List[str], tags: List[str]) -> Dict[str, str]:
    """Three-axis stamps (no-router StackedExperts cell) + atlas provenance."""
    return {
        "ss_network_module": "networks.lora_anima",
        "ss_use_moe_style": "independent_A",
        "ss_route_per_layer": "False",
        "ss_router_source": "none",
        "ss_network_dim": "Dynamic",
        "ss_network_alpha": "Dynamic",
        "ss_num_experts": str(len(ckpts)),
        "ss_atlas_tags": json.dumps({t: i for i, t in enumerate(tags)}),
        "ss_atlas_ingredients": json.dumps([Path(c).name for c in ckpts]),
    }


def pack_atlas_file(
    ckpts: List[str],
    out: str,
    tags: Optional[List[str]] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Load ingredient .safetensors, pack, write the on-disk moe-form atlas."""
    from safetensors.torch import load_file, save_file

    if tags is None:
        tags = [default_tag(c) for c in ckpts]
    if len(tags) != len(ckpts):
        raise ValueError(f"{len(tags)} tags for {len(ckpts)} checkpoints.")
    if len(set(tags)) != len(tags):
        raise ValueError(f"Duplicate atlas tags: {tags}.")

    sds = [load_file(c) for c in ckpts]
    atlas = pack_state_dicts(sds, paths=list(ckpts))
    sd_disk = StackedExpertsLoRAModule.build_moe_state_dict(atlas, out_dtype)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd_disk, str(out_path), metadata=atlas_metadata(ckpts, tags))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Per-expert artist tags, same order as --ckpts (default: derived "
        "from the ingredient filenames).",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pack_atlas_file(args.ckpts, args.out, args.tags)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
