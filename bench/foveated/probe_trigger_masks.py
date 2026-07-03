"""Phase-2t probe — WHERE the SEA trigger looks: attribution check for the 1a win.

1a's ``fovsea`` beat plain aggressive Spectrum at matched compute and the win was
attributed to *fovea-driven* refresh timing. But the earlier SEA-schedule bench
found the **global** SEA trigger's reallocation neutral at matched compute — so
the attribution is unproven: maybe any SEA trigger (content-adaptive timing per
se) beats the fixed window here, mask be damned. Diverse trigger regions decide
it. No merging, no blending — every arm is full-res; only the *refresh timing
metric* changes, at matched forward count (per-region auto-δ):

  * ``sea_rect``    — oracle fovea crop (1a's fovsea; built-in regression check
                      — same seeds/procedure, must reproduce run 1744-p1a).
  * ``sea_global``  — whole-latent distance (the production ``schedule="sea"``).
  * ``sea_scatter`` — random 35% of cells ≈ unbiased spatial sample of global.
  * ``sea_miss``    — anti-oracle rect: background-driven timing (placement null).
  * ``sea_multi3``  — 3 random rects (fragmented trigger region).

Trigger masks reuse ``probe_mask_shapes`` builders with the same seed scheme, so
regions are identical to the P2 merge run. Readouts vs the full-compute baseline:
subject-box RMSE (the oracle rect, in every arm — the region we actually care
about), periphery RMSE, forwards, refresh steps, per-region δ.

Interpretation grid (subject RMSE at matched fwd):
  * ``sea_rect`` < spec, ``sea_global``/``sea_scatter`` ≈ spec
      → fovea crop is load-bearing: 1a's framing CONFIRMED; mask sources (2b)
        matter for the trigger too, and ``sea_miss`` shows the placement cost.
  * ``sea_global``/``sea_scatter`` ≈ ``sea_rect`` < spec
      → win is generic SEA-vs-window timing: 1a attribution WRONG, ship the
        simpler global trigger, no mask needed for timing (masks stay a merge/
        partial-recompute concern only).

Usage:
  uv run python -m bench.foveated.probe_trigger_masks
  uv run python -m bench.foveated.probe_trigger_masks --seeds 40 41 42
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

log = logging.getLogger("bench.foveated.trigger")
logging.basicConfig(level=logging.INFO, format="%(message)s")


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
    # Spectrum knobs — same aggressive point as 0b/1a.
    ap.add_argument("--window_size", type=float, default=3.0)
    ap.add_argument("--flex_window", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=int, default=4)
    ap.add_argument("--fovea_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--num_rects", type=int, default=3)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--label", default="p2t")
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

    oracle = _rect_cells(hp, wp, args.fovea_frac, *args.fovea_center)
    n_oracle = int(oracle.sum())

    def build_cells(name: str, seed: int) -> torch.Tensor:
        """Same names + seed scheme as probe_mask_shapes → identical regions."""
        gen = torch.Generator().manual_seed(
            seed * 1000 + zlib.crc32(name.encode()) % 997
        )
        if name == "rect":
            return oracle
        if name == "global":
            return torch.ones(hp, wp, dtype=torch.bool)
        if name == "rect_miss":
            return _rect_miss_cells(hp, wp, args.fovea_frac, oracle, gen)
        if name == "multi3":
            return _multi_rect_cells(hp, wp, args.fovea_frac, args.num_rects, gen)
        if name == "scatter":
            return _scatter_cells(hp, wp, n_oracle, gen)
        raise SystemExit(f"unknown trigger region {name}")

    def make_dist_fn(cells: torch.Tensor):
        m_lat = (
            cells.repeat_interleave(args.pool, 0)
            .repeat_interleave(args.pool, 1)
            .to(device)
        )
        return lambda now, prev: l1rel(now[..., m_lat], prev[..., m_lat])

    # Fixed evaluation regions: the oracle-rect subject box, in every arm.
    subject_px = np.kron(oracle.numpy(), np.ones((args.pool * 8,) * 2, dtype=bool))
    periph_px = ~subject_px
    box = np.nonzero(subject_px)
    subject_box = (box[1].min(), box[0].min(), box[1].max() + 1, box[0].max() + 1)

    REGIONS = ["rect", "global", "rect_miss", "multi3", "scatter"]
    ARM_OF = {r: f"sea_{r.replace('rect_', '')}" for r in REGIONS}

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

    deltas: dict[str, float] | None = None  # calibrated from seed-0's spec traces
    per_seed = []
    for seed in args.seeds:
        log.info(f"\n=== seed {seed} ===")
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        dist_fns = {r: make_dist_fn(build_cells(r, seed)) for r in REGIONS}
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
        spec_info = run("spec", schedule="window", trace_fns=dist_fns)
        if deltas is None:
            deltas = {
                r: solve_delta_for_refresh_ratio(spec_info["traces"][r], target_ratio)
                for r in REGIONS
            }
            log.info(
                "  per-region delta: "
                + ", ".join(f"{r}={d:.4g}" for r, d in deltas.items())
                + f" (target refresh_ratio {target_ratio:.2f})"
            )
        for r in REGIONS:
            run(ARM_OF[r], schedule="sea", sea_delta=deltas[r], sea_dist_fn=dist_fns[r])

        base = images["full"]
        errmaps = []
        arm_names = ["spec", *[ARM_OF[r] for r in REGIONS]]
        for name in arm_names:
            img, m = images[name], row["arms"][name]
            m["subject_rmse_vs_full"] = _rmse_region(img, base, subject_px)
            m["periph_rmse_vs_full"] = _rmse_region(img, base, periph_px)
            errmaps.append((_errmap(img, base), f"|{name} - full|"))
            log.info(
                f"    {name}: fwd={m['actual_forwards']}  "
                f"subject_rmse={m['subject_rmse_vs_full']:.4f}  "
                f"periph_rmse={m['periph_rmse_vs_full']:.4f}  "
                f"refresh@{m['refresh_steps']}"
            )
        per_seed.append(row)

        names = ["full", *arm_names]
        full_row = _grid(
            [
                (
                    _outline(images[n], subject_box),
                    f"{n} ({row['arms'][n]['actual_forwards']} fwd)",
                )
                for n in names
            ]
        )
        crop_row = _grid([(images[n].crop(subject_box), f"subject {n}") for n in names])
        _stack([full_row, crop_row]).save(run_dir / f"compare_seed{seed}.png")
        _grid(errmaps).save(run_dir / f"errmap_seed{seed}.png")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate + attribution readout ──
    def _mean(name, key):
        return float(np.mean([r["arms"][name][key] for r in per_seed]))

    arm_names = ["spec", *[ARM_OF[r] for r in REGIONS]]
    agg = {
        n: {
            "subject_rmse": _mean(n, "subject_rmse_vs_full"),
            "periph_rmse": _mean(n, "periph_rmse_vs_full"),
            "fwd": _mean(n, "actual_forwards"),
        }
        for n in arm_names
    }
    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    spec_s = agg["spec"]["subject_rmse"]
    if any_nan:
        verdict = "HARD_FAIL: NaN/Inf."
    else:
        verdict = (
            "Visual call on compare_seed*/errmap_seed*.png. ATTRIBUTION — subject "
            f"RMSE vs full at matched fwd: spec {spec_s:.4f} "
            f"({agg['spec']['fwd']:.1f} fwd); "
            + ", ".join(
                f"{n} {agg[n]['subject_rmse']:.4f} ({agg[n]['subject_rmse'] - spec_s:+.4f}, "
                f"{agg[n]['fwd']:.1f} fwd)"
                for n in arm_names[1:]
            )
            + ". sea_rect must reproduce 1a's fovsea win (regression). If "
            "sea_global/sea_scatter ≈ spec, the fovea crop is load-bearing "
            "(1a framing confirmed); if they match sea_rect, the win is generic "
            "SEA-vs-window timing and no trigger mask is needed."
        )

    metrics = {
        "spectrum": {
            "window_size": args.window_size,
            "flex_window": args.flex_window,
            "warmup_steps": args.warmup_steps,
        },
        "fovea_frac": args.fovea_frac,
        "num_rects": args.num_rects,
        "deltas": deltas,
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
