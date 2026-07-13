#!/usr/bin/env python3
"""Artist-atlas Phase 0 — G1/G2 exactness gates on the real DiT.

Proposal: docs/proposal/artist_atlas_pack.md. The pack itself is
``scripts/atlas/pack.py``; the CPU-level ΔW/forward gates live in
``tests/test_atlas_pack.py``. This bench runs the full-model version: sample
final latents with the packed atlas under explicit gates and compare against
the solo ingredient checkpoints, same seed / prompt / steps.

Gates (correctness checks, not quality judgments):
  G1  one-hot expert e reproduces solo ingredient e's final latents within
      bf16 tolerance (reported as RMSE relative to the solo-vs-base RMSE —
      the adapter's own effect size; pass = ratio < --g1_ratio).
  G2  zero gate is bit-identical to the base model (expert contribution is
      exactly zero by construction; assert anyway).
  Pin cleared gates = uniform 1/E = the ingredient average (the documented
      atlas gotcha: nothing sets gates per step with router_source="none",
      so a caller that forgets to gate gets the average, NOT the base).

Usage
-----
    python bench/atlas/run_bench.py \
        --atlas output/ckpt/anima_atlas5_moe.safetensors \
        --tags hews sincos --steps 10 --label phase0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._anima import build_anima  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.memorization.probe import _Acc, build_training_args  # noqa: E402
from library.anima import training as anima_train_utils  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402
from library.training.validation import (  # noqa: E402
    _build_val_crossattn_emb,
    _load_safetensors,
)

SOUP_DIR = REPO_ROOT / "output" / "ckpt" / "soup"


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", default="output/ckpt/anima_atlas5_moe.safetensors")
    ap.add_argument(
        "--tags",
        nargs="+",
        default=["hews", "sincos"],
        help="Atlas tags to gate-test (each needs its solo ingredient in "
        "output/ckpt/soup/anima_soup_<tag>.safetensors).",
    )
    ap.add_argument("--method", default="soup")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--size", type=int, nargs=2, default=(1024, 1024))
    ap.add_argument(
        "--te",
        default=None,
        help="Cached TE .safetensors to condition on (default: first cache of "
        "the first tested tag).",
    )
    ap.add_argument(
        "--g1_ratio",
        type=float,
        default=0.05,
        help="G1 passes when RMSE(atlas_onehot, solo) < ratio * RMSE(solo, base).",
    )
    ap.add_argument(
        "--control",
        action="store_true",
        default=True,
        help="Also pack a duplicate-ingredient control atlas (one-hot == solo "
        "by construction) to measure the pure kernel-path noise floor.",
    )
    ap.add_argument("--no-control", dest="control", action="store_false")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from safetensors import safe_open

    with safe_open(args.atlas, framework="pt") as f:
        meta = dict(f.metadata() or {})
    tag_map = json.loads(meta["ss_atlas_tags"])
    for tag in args.tags:
        if tag not in tag_map:
            raise SystemExit(f"tag {tag!r} not in atlas tags {sorted(tag_map)}")

    te_path = args.te
    if te_path is None:
        te_dir = REPO_ROOT / "post_image_dataset" / "lora" / args.tags[0]
        te_path = str(sorted(te_dir.glob("*_anima_te.safetensors"))[0])
    print(f"conditioning TE: {te_path}", flush=True)

    targs = build_training_args(args.method, args.preset, {})
    device = torch.device("cuda")
    shim = _Acc(device, torch.bfloat16)
    flow_shift = float(getattr(targs, "discrete_flow_shift", 3.0))
    te_sd = _load_safetensors(te_path)
    height, width = int(args.size[0]), int(args.size[1])

    # Fixed single-forward probe input: separates per-forward exactness from
    # trajectory chaos (bf16 kernel-order noise compounds over Euler steps).
    probe_sigmas = (0.9, 0.5, 0.1)
    g = torch.manual_seed(args.seed + 1)
    z_probe = torch.randn(
        1, 16, 1, height // 8, width // 8, dtype=torch.float32, generator=g
    )

    @torch.no_grad()
    def run(adapter: str | None, gates: dict[str, torch.Tensor] | None = None):
        """Build DiT(+adapter), sample + σ-probe once per gate config.

        Returns ``{gate_key: {"lat": final latents, "v": {σ: velocity}}}``.
        ``gates=None`` → single ungated config keyed "". ``gates`` values of
        None mean "clear to the uniform default".
        """
        bundle = build_anima(targs, adapter=adapter, train_mode=False)
        dit = bundle.anima
        out: dict[str, dict] = {}
        try:
            with shim.autocast():
                if hasattr(dit, "prepare_block_swap_before_forward"):
                    dit.prepare_block_swap_before_forward()
                emb = _build_val_crossattn_emb(dit, te_sd, shim)
                z = z_probe.to(device, dtype=dit.dtype)
                pad = torch.zeros(
                    1, 1, height // 8, width // 8, dtype=dit.dtype, device=device
                )
                for key, gate in (gates or {"": None}).items():
                    if gate is not None:
                        bundle.network.set_routing_weights(gate.to(device))
                    elif bundle.network is not None and hasattr(
                        bundle.network, "clear_routing_weights"
                    ):
                        bundle.network.clear_routing_weights()
                    lat = anima_train_utils.sample_image_latents(
                        accelerator=shim,
                        dit=dit,
                        height=height,
                        width=width,
                        crossattn_emb=emb,
                        sample_steps=args.steps,
                        guidance_scale=args.cfg,
                        flow_shift=flow_shift,
                        seed=args.seed,
                        show_progress=False,
                    )
                    vs = {}
                    for s in probe_sigmas:
                        t = torch.tensor([s], dtype=dit.dtype, device=device)
                        v = dit(z, t, emb.to(dit.dtype), padding_mask=pad)
                        vs[s] = v.detach().float().cpu()
                    out[key] = {"lat": lat.detach().float().cpu(), "v": vs}
                    print(f"  sampled [{adapter or 'base'}] gate={key!r}", flush=True)
        finally:
            del bundle, dit
            clean_memory_on_device(device)
        return out

    E = len(tag_map)
    print(f"atlas: {args.atlas} (E={E}, tags={tag_map})", flush=True)

    print("run: base (no adapter)", flush=True)
    base = run(None)[""]

    print("run: atlas (zero / uniform / one-hots)", flush=True)
    gates: dict[str, torch.Tensor] = {"zero": torch.zeros(1, E)}
    gates["uniform"] = None  # cleared → default 1/E
    for tag in args.tags:
        onehot = torch.zeros(1, E)
        onehot[0, tag_map[tag]] = 1.0
        gates[f"onehot:{tag}"] = onehot
    atlas_out = run(args.atlas, gates)

    solo_out: dict[str, dict] = {}
    for tag in args.tags:
        solo = SOUP_DIR / f"anima_soup_{tag}.safetensors"
        print(f"run: solo {tag} ({solo.name})", flush=True)
        solo_out[tag] = run(str(solo))[""]

    # Control: a duplicate-ingredient atlas whose one-hot is mathematically
    # identical to the solo checkpoint — its deviation is the pure bf16
    # kernel-path noise floor (stacked einsum vs plain fused GEMM), the fair
    # yardstick for the real atlas's G1 ratio.
    ctl_tag = args.tags[0]
    ctl_solo = str(SOUP_DIR / f"anima_soup_{ctl_tag}.safetensors")
    ctl_path = None
    if args.control:
        from scripts.atlas.pack import pack_atlas_file  # noqa: PLC0415

        run_dir_early = make_run_dir("atlas", args.label)
        ctl_path = run_dir_early / f"control_atlas2_{ctl_tag}_moe.safetensors"
        pack_atlas_file(
            [ctl_solo, ctl_solo], str(ctl_path), tags=[f"{ctl_tag}:0", f"{ctl_tag}:1"]
        )
        print(f"run: control atlas (duplicate {ctl_tag})", flush=True)
        ctl_gate = torch.zeros(1, 2)
        ctl_gate[0, 0] = 1.0
        ctl_out = run(str(ctl_path), {"onehot:ctl": ctl_gate})["onehot:ctl"]
    else:
        run_dir_early = make_run_dir("atlas", args.label)
        ctl_out = None

    # ----- gates -----
    metrics: dict[str, object] = {}
    g2_max = _max_abs(atlas_out["zero"]["lat"], base["lat"])
    metrics["g2_zero_gate_max_abs_vs_base"] = g2_max
    metrics["g2_pass"] = g2_max == 0.0
    # G2 must hold per-forward too (exact-zero contribution).
    metrics["g2_fwd_max_abs"] = max(
        _max_abs(atlas_out["zero"]["v"][s], base["v"][s]) for s in base["v"]
    )

    # Control yardstick first: the duplicate-ingredient atlas is exactly the
    # solo model expressed through the stacked kernel path, so its ratios ARE
    # the numeric noise floor. G1 passes when a tag's per-forward ratio is
    # indistinguishable from that floor (≤ 1.5×); without --control, fall back
    # to the fixed --g1_ratio threshold.
    ctl_fwd_max = None
    if ctl_out is not None:
        s_out = solo_out.get(ctl_tag) or run(ctl_solo)[""]
        metrics["control_traj_ratio"] = _rmse(ctl_out["lat"], s_out["lat"]) / max(
            _rmse(s_out["lat"], base["lat"]), 1e-12
        )
        ctl_ratios = []
        for s in base["v"]:
            dv = _rmse(ctl_out["v"][s], s_out["v"][s])
            dv_anchor = _rmse(s_out["v"][s], base["v"][s])
            ctl_ratios.append(dv / max(dv_anchor, 1e-12))
            metrics[f"control_fwd_ratio_sigma{s}"] = ctl_ratios[-1]
        ctl_fwd_max = max(ctl_ratios)

    g1_all = True
    for tag in args.tags:
        a_lat = atlas_out[f"onehot:{tag}"]["lat"]
        s_lat = solo_out[tag]["lat"]
        d_atlas = _rmse(a_lat, s_lat)
        d_anchor = _rmse(s_lat, base["lat"])
        ratio = d_atlas / d_anchor if d_anchor > 0 else float("inf")
        metrics[f"g1_{tag}_rmse_atlas_vs_solo"] = d_atlas
        metrics[f"g1_{tag}_rmse_solo_vs_base"] = d_anchor
        metrics[f"g1_{tag}_traj_ratio"] = ratio
        # Per-forward exactness (no chaos accumulation) — the real G1 signal.
        fwd_ratios = {}
        for s in base["v"]:
            dv = _rmse(atlas_out[f"onehot:{tag}"]["v"][s], solo_out[tag]["v"][s])
            dv_anchor = _rmse(solo_out[tag]["v"][s], base["v"][s])
            fwd_ratios[s] = dv / dv_anchor if dv_anchor > 0 else float("inf")
            metrics[f"g1_{tag}_fwd_ratio_sigma{s}"] = fwd_ratios[s]
        if ctl_fwd_max is not None:
            ok = max(fwd_ratios.values()) <= 1.5 * ctl_fwd_max
        else:
            ok = max(fwd_ratios.values()) < args.g1_ratio
        g1_all &= ok
        metrics[f"g1_{tag}_pass"] = ok
    metrics["g1_pass"] = g1_all

    if len(args.tags) >= 2:
        a, b = args.tags[:2]
        metrics["cross_expert_rmse"] = _rmse(
            atlas_out[f"onehot:{a}"]["lat"], atlas_out[f"onehot:{b}"]["lat"]
        )
    metrics["uniform_rmse_vs_base"] = _rmse(atlas_out["uniform"]["lat"], base["lat"])

    print("\n===== atlas Phase 0 gates =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"G1 {'PASS' if metrics['g1_pass'] else 'FAIL'}  ")
    print(f"G2 {'PASS' if metrics['g2_pass'] else 'FAIL'}")

    run_dir = run_dir_early
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        device=device,
        extra={"atlas_tags": tag_map, "te_path": str(te_path)},
    )
    print(f"result: {run_dir}/result.json")


if __name__ == "__main__":
    main()
