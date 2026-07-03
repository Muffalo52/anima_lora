"""Phase-3s probe — fovea-FRACTION ladder on the real merge path (aggressive masks).

The aggression knob is the fovea *area*: the 2×2 periphery pooling stays fixed
(owner clarification), and the ladder shrinks how much of the image runs
full-res below σ_c. All arms use the real 1b merged stack with the 2b ``combo``
mask (cfgdelta + x0var, built endogenously at the crossing) at σ_c=0.75 (the
standalone merge knee); ``rect``@0.35 is the status-quo reference.

Effective tokens: f + (1−f)/4 → 0.35→51%, 0.25→44%, 0.20→40%, 0.15→36%,
0.10→33%. The stretch gate (plan §3s): ``combo`` at a low fraction matches
``rect``@0.35 subject quality — mask intelligence converting to speed. The
breaking point is the deliverable either way: on multi-subject channel6 three
faces compete for the shrinking budget, so subject cover is expected to be the
first casualty — watch ``subject_cover`` alongside subject RMSE.

Usage:
  uv run python -m bench.foveated.probe_fraction_stretch
  uv run python -m bench.foveated.probe_fraction_stretch --fracs 0.35 0.2 --seeds 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result
from bench.foveated.probe_mask_shapes import (
    _cells_px,
    _rect_cells,
    _ring_px,
    _rmse_sel,
)
from bench.foveated.probe_mask_sources import (
    _draw_boxes,
    _draw_mask_and_boxes,
    _subject_strip,
    denoise_adaptive,
    denoise_baseline,
)
from bench.foveated.probe_velocity_foveation import _grid, _lap_energy, _stack, _to_pil

log = logging.getLogger("bench.foveated.fraction")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(ap)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEG)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40, 41])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sigma_c", type=float, default=0.75)
    ap.add_argument(
        "--fracs", type=float, nargs="+", default=[0.35, 0.25, 0.20, 0.15, 0.10]
    )
    ap.add_argument("--rect_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument(
        "--prompts",
        nargs="+",
        default=["default", "channel6", "ootomo4424330"],
        help="'default' plus keys of hard_prompts.json",
    )
    ap.add_argument("--label", default="p3s")
    args = ap.parse_args()

    device = torch.device(args.device)
    repo = Path(__file__).resolve().parents[2]
    hard = json.loads((Path(__file__).parent / "hard_prompts.json").read_text())

    import inference as inference_mod
    from anima_lora import load_vae
    from library.inference import sampling as inference_utils
    from library.inference.models import load_dit_model
    from library.inference.output import decode_latent
    from library.inference.text import (
        MAX_CROSSATTN_TOKENS,
        ensure_text_strategies,
        prepare_text_inputs,
    )
    from diffusers.utils.torch_utils import randn_tensor

    infer_argv = [
        "--dit", args.dit,
        "--text_encoder", args.text_encoder,
        "--vae", args.vae,
        "--vae_chunk_size", "64",
        "--vae_disable_cache",
        "--attn_mode", "flash",
        "--lora_multiplier", "1.0",
        "--prompt", DEFAULT_PROMPT,
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

    h_lat, w_lat = args.height // 8, args.width // 8
    hp, wp = h_lat // args.pool, w_lat // args.pool
    px_per_cell = args.pool * 8
    oracle = _rect_cells(hp, wp, args.rect_frac, *args.fovea_center)

    def oracle_box():
        m = _cells_px(oracle, px_per_cell)
        nz = np.nonzero(m)
        return [
            int(nz[1].min()),
            int(nz[0].min()),
            int(nz[1].max()) + 1,
            int(nz[0].max()) + 1,
        ]

    prompt_spec = {}
    for pname in args.prompts:
        if pname == "default":
            prompt_spec[pname] = {
                "text": DEFAULT_PROMPT,
                "boxes": {s: [oracle_box()] for s in args.seeds},
            }
        else:
            entry = hard[pname]
            prompt_spec[pname] = {
                "text": (repo / entry["caption"]).read_text().strip(),
                "boxes": {s: entry["boxes"][str(s)] for s in args.seeds},
            }

    embeds = {}
    neg_embed = None
    for pname, spec in prompt_spec.items():
        iargs.prompt = spec["text"]
        ctx, ctx_null = prepare_text_inputs(iargs, device, anima, shared_models=None)
        embeds[pname] = ctx["embed"][0].to(device, torch.bfloat16)
        neg_embed = ctx_null["embed"][0].to(device, torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info("Loading VAE ...")
    vae = load_vae(args.vae, device="cpu", spatial_chunk_size=64)
    vae.to(torch.bfloat16)
    vae.eval()

    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.to(device)
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    run_dir = make_run_dir("foveated", label=args.label)

    arms = [(f"rect{int(args.rect_frac * 100)}", "rect", args.rect_frac)] + [
        (f"combo{int(f * 100)}", "combo", f) for f in args.fracs
    ]

    results = []
    for pname, spec in prompt_spec.items():
        for seed in args.seeds:
            log.info(f"\n=== {pname} / seed {seed} ===")
            boxes = spec["boxes"][seed]
            subject_sel = np.zeros((args.height, args.width), dtype=bool)
            for b in boxes:
                subject_sel[b[1] : b[3], b[0] : b[2]] = True
            init = randn_tensor(
                (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                device=device,
                dtype=torch.bfloat16,
            )
            embed = embeds[pname]

            log.info("  baseline ...")
            lat, base_e2e = denoise_baseline(
                anima, init.clone(), embed, neg_embed, sigmas,
                args.guidance_scale, device, pad,
            )  # fmt: skip
            base_img = _to_pil(decode_latent(vae, lat, device))
            base_img.save(run_dir / f"{pname}_seed{seed}_baseline.png")

            row = {"prompt": pname, "seed": seed, "arms": {}}
            images: dict[str, Image.Image] = {}
            cells_by: dict[str, torch.Tensor] = {}
            for name, source, frac in arms:
                log.info(f"  {name} ...")
                lat, info, cells, _score = denoise_adaptive(
                    anima, init.clone(), embed, neg_embed, sigmas,
                    args.guidance_scale, device, pad,
                    source=source, static_cells=oracle if source == "rect" else None,
                    sigma_c=args.sigma_c, fovea_frac=frac, pool=args.pool,
                )  # fmt: skip
                img = _to_pil(decode_latent(vae, lat, device))
                img.save(run_dir / f"{pname}_seed{seed}_{name}.png")
                images[name], cells_by[name] = img, cells
                own_px = _cells_px(cells, px_per_cell)
                ring = _ring_px(cells, px_per_cell)
                m = dict(info)
                m["nan_inf"] = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
                m["subject_rmse"] = _rmse_sel(img, base_img, subject_sel)
                m["own_mask_rmse"] = _rmse_sel(img, base_img, own_px)
                m["image_rmse"] = _rmse_sel(img, base_img, np.ones_like(subject_sel))
                m["subject_cover"] = float(
                    (own_px & subject_sel).sum() / max(subject_sel.sum(), 1)
                )
                m["periph_lap_ratio"] = _lap_energy(img, ~own_px) / max(
                    _lap_energy(base_img, ~own_px), 1e-9
                )
                m["ring_lap_ratio"] = _lap_energy(img, ring) / max(
                    _lap_energy(base_img, ring), 1e-9
                )
                m["e2e_speedup"] = base_e2e / m["e2e_s"]
                row["arms"][name] = m
                log.info(
                    f"    {name}: subject={m['subject_rmse']:.4f} "
                    f"(cover {m['subject_cover']:.0%})  own={m['own_mask_rmse']:.4f}  "
                    f"frac={m['fovea_frac_actual']:.3f}  "
                    f"periph×{m['periph_lap_ratio']:.2f} ring×{m['ring_lap_ratio']:.2f}  "
                    f"e2e ×{m['e2e_speedup']:.2f}"
                )
            results.append(row)

            full_row = _grid(
                [(_draw_boxes(base_img, boxes), "baseline")]
                + [
                    (
                        _draw_mask_and_boxes(
                            images[n], cells_by[n], px_per_cell, boxes
                        ),
                        f"{n} (cov {row['arms'][n]['subject_cover']:.0%}, "
                        f"×{row['arms'][n]['e2e_speedup']:.2f})",
                    )
                    for n, _, _ in arms
                ]
            )
            strip_row = _grid(
                [(_subject_strip(base_img, boxes), "subjects baseline")]
                + [
                    (_subject_strip(images[n], boxes), f"subjects {n}")
                    for n, _, _ in arms
                ]
            )
            _stack([full_row, strip_row]).save(
                run_dir / f"compare_{pname}_seed{seed}.png"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Aggregate: per prompt × arm ──
    keys = (
        "subject_rmse",
        "own_mask_rmse",
        "subject_cover",
        "fovea_frac_actual",
        "periph_lap_ratio",
        "ring_lap_ratio",
        "e2e_speedup",
    )
    agg = {}
    for pname in args.prompts:
        rows = [r for r in results if r["prompt"] == pname]
        agg[pname] = {
            n: {k: float(np.mean([r["arms"][n][k] for r in rows])) for k in keys}
            for n, _, _ in arms
        }
    any_nan = any(m["nan_inf"] for r in results for m in r["arms"].values())

    rect_name = arms[0][0]
    if any_nan:
        verdict = "HARD_FAIL: NaN/Inf."
    else:
        parts = []
        for pname in args.prompts:
            a = agg[pname]
            parts.append(
                f"[{pname}] "
                + ", ".join(
                    f"{n} {a[n]['subject_rmse']:.4f}(cov {a[n]['subject_cover']:.0%}, "
                    f"×{a[n]['e2e_speedup']:.2f})"
                    for n, _, _ in arms
                )
            )
        verdict = (
            "Visual call on compare_*.png. Subject RMSE (cover, e2e) by prompt: "
            + "; ".join(parts)
            + f". Stretch gate: lowest combo fraction whose subject RMSE ≈ {rect_name} "
            "on every prompt = the shippable aggressive setting; the fraction where "
            "cover/subject break is the deliverable either way."
        )

    metrics = {
        "sigma_c": args.sigma_c,
        "fracs": args.fracs,
        "rect_frac": args.rect_frac,
        "pool_latent_px": args.pool,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "prompts": {p: prompt_spec[p]["boxes"] for p in args.prompts},
        "aggregate": agg,
        "per_run": results,
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
