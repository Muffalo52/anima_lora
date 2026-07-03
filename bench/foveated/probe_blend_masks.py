"""Phase-2f probe — diverse masks on the BLEND mechanisms (fovblend / fov1a).

P2t settled the trigger: region-independent, generic SEA-vs-window timing. The
remaining 1a mechanisms are the blend ones — ``fovblend`` (window schedule; on
actual forwards below σ_c the emitted features are mask-blended: mask = actual,
complement = forecast; forecaster history stays anchored to full actuals) and
``fov1a`` (same blend on the SEA schedule; per P2t the trigger region is
irrelevant, so fov1a arms here use the production-shaped **global** trigger).

Why diverse masks here matter: fovblend's oracle-rect neutrality (+0.0001, run
1744-p1a) is the foundation of Phase-3 partial recompute. But the P2 merge run
falsified scatter via *attention context* (isolated fovea cells surrounded by
merged tokens). Blend has no such coupling at emit — ``final_layer`` is
per-token — so shape-independence is plausible but unproven, and it decides
whether the Phase-3 recompute region is freely choosable or inherits the
compact-blob constraint.

Arms (blend σ_c=0.5, the Spectrum-composed knee): ``full``, ``spec``, then
``fovblend_X`` / ``fov1a_X`` for X ∈ {rect, miss, multi3, scatter} (identical
placements to the P2/P2t runs). Readouts vs full, with spec's same-region values
as the neutrality reference: own-mask RMSE (actual-emitted region), complement
RMSE (forecast-emitted region), fixed subject-box RMSE, forwards, refreshes.

Gate: neutrality is mask-independent if every arm's own/complement RMSE ≈ spec's
same-region RMSE (1a standard: |Δ| ≲ 0.003) — then Phase-3 region choice is
shape-free. A scatter-only failure = compactness constraint carries over.
Caveat unchanged from 1a: fits are still anchored to full actuals; the
no-complement-update degradation remains Phase 3's own pre-gate.

Usage:
  uv run python -m bench.foveated.probe_blend_masks
  uv run python -m bench.foveated.probe_blend_masks --seeds 40 41 42 --sigma_c 0.75
"""

from __future__ import annotations

import argparse
import logging
import sys
import zlib

import numpy as np
import torch
from PIL import Image

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result
from bench.foveated.probe_foveated_spectrum import spectrum_arm
from bench.foveated.probe_mask_shapes import (
    _cells_px,
    _multi_rect_cells,
    _rect_cells,
    _rect_miss_cells,
    _scatter_cells,
)
from bench.foveated.probe_spectrum_compose import _errmap, _rmse_region
from bench.foveated.probe_velocity_foveation import (
    _grid,
    _outline,
    _stack,
    _to_pil,
)

log = logging.getLogger("bench.foveated.blend")
logging.basicConfig(level=logging.INFO, format="%(message)s")

REGIONS = ["rect", "rect_miss", "multi3", "scatter"]
SHORT = {"rect": "rect", "rect_miss": "miss", "multi3": "multi3", "scatter": "scatter"}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Spectrum knobs — same aggressive point as 0b/1a/2t.
    ap.add_argument("--window_size", type=float, default=3.0)
    ap.add_argument("--flex_window", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=int, default=4)
    ap.add_argument("--sigma_c", type=float, default=0.5)
    ap.add_argument("--fovea_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--num_rects", type=int, default=3)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--label", default="p2f")
    args = ap.parse_args()

    device = torch.device(args.device)

    import inference as inference_mod
    from anima_lora import load_vae
    from diffusers.utils.torch_utils import randn_tensor
    from library.inference import sampling as inference_utils
    from library.inference.models import load_dit_model
    from library.inference.output import decode_latent
    from library.inference.text import (
        MAX_CROSSATTN_TOKENS,
        ensure_text_strategies,
        prepare_text_inputs,
    )
    from networks.spectrum_sea import (
        l1rel,
        solve_delta_for_refresh_ratio,
        window_decision_fraction,
    )

    infer_argv = [
        "--dit", args.dit,
        "--text_encoder", args.text_encoder,
        "--vae", args.vae,
        "--vae_chunk_size", "64",
        "--vae_disable_cache",
        "--attn_mode", "flash",
        "--lora_multiplier", "1.0",
        "--prompt", args.prompt,
        "--negative_prompt", args.negative_prompt,
        "--image_size", str(args.height), str(args.width),
        "--infer_steps", str(args.infer_steps),
        "--flow_shift", str(args.flow_shift),
        "--guidance_scale", str(args.guidance_scale),
        "--seed", str(args.seeds[0]),
        "--device", str(device),
        "--save_path", "output/tests",  # required by parse_args; probe writes its own
    ]  # fmt: skip
    _saved_argv = sys.argv
    try:
        sys.argv = ["inference.py", *infer_argv]
        iargs = inference_mod.parse_args()
    finally:
        sys.argv = _saved_argv
    iargs.lora_weight = None  # BARE DiT
    iargs.sampler = "euler"

    ensure_text_strategies(args.text_encoder, MAX_CROSSATTN_TOKENS)
    log.info("Loading bare DiT (no LoRA, eager) ...")
    anima = load_dit_model(iargs, device, torch.bfloat16)
    context, context_null = prepare_text_inputs(
        iargs, device, anima, shared_models=None
    )
    embed = context["embed"][0].to(device, torch.bfloat16)
    neg_embed = context_null["embed"][0].to(device, torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info("Loading VAE ...")
    vae = load_vae(args.vae, device="cpu", spatial_chunk_size=64)
    vae.to(torch.bfloat16)
    vae.eval()

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps = timesteps.to(device, torch.bfloat16)
    sigmas = sigmas.to(device)
    num_steps = len(timesteps)

    h_lat, w_lat = args.height // 8, args.width // 8
    if h_lat % args.pool or w_lat % args.pool:
        raise SystemExit(f"latent {h_lat}x{w_lat} not divisible by pool={args.pool}")
    hp, wp = h_lat // args.pool, w_lat // args.pool
    p = anima.patch_spatial  # latent px per token edge
    tok_per_cell = args.pool // p
    px_per_cell = args.pool * 8

    oracle = _rect_cells(hp, wp, args.fovea_frac, *args.fovea_center)
    n_oracle = int(oracle.sum())

    def build_cells(name: str, seed: int) -> torch.Tensor:
        """Same names + seed scheme as probe_mask_shapes → identical regions."""
        gen = torch.Generator().manual_seed(
            seed * 1000 + zlib.crc32(name.encode()) % 997
        )
        if name == "rect":
            return oracle
        if name == "rect_miss":
            return _rect_miss_cells(hp, wp, args.fovea_frac, oracle, gen)
        if name == "multi3":
            return _multi_rect_cells(hp, wp, args.fovea_frac, args.num_rects, gen)
        if name == "scatter":
            return _scatter_cells(hp, wp, n_oracle, gen)
        raise SystemExit(f"unknown region {name}")

    def cells_to_tok5(cells: torch.Tensor) -> torch.Tensor:
        tok = cells.repeat_interleave(tok_per_cell, 0).repeat_interleave(
            tok_per_cell, 1
        )
        return tok.float().view(1, 1, *tok.shape, 1).to(device)

    # Global SEA trigger for the fov1a arms (P2t: region-independent, so use the
    # production-shaped whole-latent distance).
    def global_dist(now, prev):
        return l1rel(now, prev)

    # Fixed evaluation region: the oracle-rect subject box.
    subject_px = np.kron(oracle.numpy(), np.ones((px_per_cell,) * 2, dtype=bool))
    box = np.nonzero(subject_px)
    subject_box = (box[1].min(), box[0].min(), box[1].max() + 1, box[0].max() + 1)

    stop_at = num_steps - 3
    target_ratio = window_decision_fraction(
        num_steps, args.warmup_steps, stop_at, args.window_size, args.flex_window
    )
    common = dict(
        warmup=args.warmup_steps,
        window_size=args.window_size,
        flex_window=args.flex_window,
    )
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    run_dir = make_run_dir("foveated", label=args.label)

    sea_delta: float | None = None
    per_seed = []
    for seed in args.seeds:
        log.info(f"\n=== seed {seed} ===")
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        cells_by_region = {r: build_cells(r, seed) for r in REGIONS}
        images: dict[str, Image.Image] = {}
        row = {"seed": seed, "arms": {}}

        def run(name, **kw):
            log.info(f"  {name} ...")
            lat, info = spectrum_arm(
                anima, init, embed, neg_embed, timesteps, sigmas,  # noqa: B023
                args.guidance_scale, pad, device, **common, **kw,
            )  # fmt: skip
            img = _to_pil(decode_latent(vae, lat, device))
            img.save(run_dir / f"seed{seed}_{name}.png")  # noqa: B023
            images[name] = img  # noqa: B023
            row["arms"][name] = {  # noqa: B023
                "actual_forwards": info["actual_forwards"],
                "refresh_steps": info["refresh_steps"],
                "nan_inf": bool(torch.isnan(lat).any() or torch.isinf(lat).any()),
                "latent_std": float(lat.float().std()),
            }
            return info

        run("full", schedule="all")
        spec_info = run("spec", schedule="window", trace_fns={"global": global_dist})
        if sea_delta is None:
            sea_delta = solve_delta_for_refresh_ratio(
                spec_info["traces"]["global"], target_ratio
            )
            log.info(
                f"  global-SEA delta={sea_delta:.4g} "
                f"(target refresh_ratio {target_ratio:.2f})"
            )
        for r in REGIONS:
            tok5 = cells_to_tok5(cells_by_region[r])
            run(
                f"fovblend_{SHORT[r]}",
                schedule="window",
                blend=True,
                mask_tok5=tok5,
                sigma_c=args.sigma_c,
            )
            run(
                f"fov1a_{SHORT[r]}",
                schedule="sea",
                sea_delta=sea_delta,
                sea_dist_fn=global_dist,
                blend=True,
                mask_tok5=tok5,
                sigma_c=args.sigma_c,
            )

        base = images["full"]
        spec_img = images["spec"]
        errmaps = [(_errmap(spec_img, base), "|spec - full|")]
        arm_names = ["spec"]
        row["arms"]["spec"]["subject_rmse_vs_full"] = _rmse_region(
            spec_img, base, subject_px
        )
        for r in REGIONS:
            own_px = _cells_px(cells_by_region[r], px_per_cell)
            spec_own = _rmse_region(spec_img, base, own_px)
            spec_comp = _rmse_region(spec_img, base, ~own_px)
            for mech in ("fovblend", "fov1a"):
                name = f"{mech}_{SHORT[r]}"
                arm_names.append(name)
                img, m = images[name], row["arms"][name]
                m["own_rmse_vs_full"] = _rmse_region(img, base, own_px)
                m["comp_rmse_vs_full"] = _rmse_region(img, base, ~own_px)
                m["own_delta_vs_spec"] = m["own_rmse_vs_full"] - spec_own
                m["comp_delta_vs_spec"] = m["comp_rmse_vs_full"] - spec_comp
                m["subject_rmse_vs_full"] = _rmse_region(img, base, subject_px)
                errmaps.append((_errmap(img, base), f"|{name} - full|"))
                log.info(
                    f"    {name}: fwd={m['actual_forwards']}  "
                    f"own={m['own_rmse_vs_full']:.4f} ({m['own_delta_vs_spec']:+.4f})  "
                    f"comp={m['comp_rmse_vs_full']:.4f} "
                    f"({m['comp_delta_vs_spec']:+.4f})  "
                    f"subject={m['subject_rmse_vs_full']:.4f}"
                )
        per_seed.append(row)

        names = ["full", "spec", *[f"fovblend_{SHORT[r]}" for r in REGIONS]]
        full_row = _grid(
            [
                (
                    _outline(images[n], subject_box),
                    f"{n} ({row['arms'][n]['actual_forwards']} fwd)",
                )
                for n in names
            ]
        )
        names1a = ["full", "spec", *[f"fov1a_{SHORT[r]}" for r in REGIONS]]
        row1a = _grid(
            [
                (
                    _outline(images[n], subject_box),
                    f"{n} ({row['arms'][n]['actual_forwards']} fwd)",
                )
                for n in names1a
            ]
        )
        crop_row = _grid([(images[n].crop(subject_box), f"subject {n}") for n in names])
        _stack([full_row, row1a, crop_row]).save(run_dir / f"compare_seed{seed}.png")
        _grid(errmaps[:5]).save(run_dir / f"errmap_blend_seed{seed}.png")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate + gate ──
    blend_arms = [
        f"{mech}_{SHORT[r]}" for r in REGIONS for mech in ("fovblend", "fov1a")
    ]

    def _mean(name, key):
        return float(np.mean([r["arms"][name][key] for r in per_seed]))

    agg = {
        n: {
            "own_rmse": _mean(n, "own_rmse_vs_full"),
            "own_delta_vs_spec": _mean(n, "own_delta_vs_spec"),
            "comp_delta_vs_spec": _mean(n, "comp_delta_vs_spec"),
            "subject_rmse": _mean(n, "subject_rmse_vs_full"),
            "fwd": _mean(n, "actual_forwards"),
        }
        for n in blend_arms
    }
    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    worst = max(
        abs(d)
        for v in agg.values()
        for d in (v["own_delta_vs_spec"], v["comp_delta_vs_spec"])
    )
    if any_nan:
        verdict = "HARD_FAIL: NaN/Inf."
    else:
        verdict = (
            "Visual call on compare_seed*.png. NEUTRALITY vs spec (own Δ / comp Δ): "
            + "; ".join(
                f"{n} {v['own_delta_vs_spec']:+.4f}/{v['comp_delta_vs_spec']:+.4f} "
                f"({v['fwd']:.1f} fwd)"
                for n, v in agg.items()
            )
            + f". Worst |Δ|={worst:.4f}. Gate: all |Δ| ≲ 0.003 → blend "
            "neutrality is mask-shape-independent (Phase-3 region freely choosable); "
            "scatter-only failure → compact-blob constraint carries to partial "
            "recompute. Fits still anchored to full actuals (1a caveat stands)."
        )

    metrics = {
        "spectrum": {
            "window_size": args.window_size,
            "flex_window": args.flex_window,
            "warmup_steps": args.warmup_steps,
        },
        "sigma_c_blend": args.sigma_c,
        "fovea_frac": args.fovea_frac,
        "num_rects": args.num_rects,
        "sea_delta_global": sea_delta,
        "target_refresh_ratio": target_ratio,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "subject_box_px": [int(v) for v in subject_box],
        "aggregate": agg,
        "per_seed": per_seed,
        "any_nan_inf": any_nan,
        "verdict": verdict,
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=[q.name for q in sorted(run_dir.glob("*.png"))],
        device=device,
    )
    log.info("\n" + "=" * 70)
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
