"""Phase-2b probe — endogenous merge-mask sources vs the static rect, on hard prompts.

P2 fixed the shape rules (compact + subject-following) and the brackets (oracle
rect / rect_miss null). This probe tests whether the mask can be DERIVED at
runtime, per (prompt, seed), from signals the sampler already computes during
the full-res steps before the σ_c crossing — zero extra models, zero extra
forwards:

  * ``cfgdelta`` — accumulated per-cell |v_cond − v_uncond| over pre-crossing
                   steps. Marks where the prompt is steering (text drive is
                   front-loaded, so the signal is complete before crossing).
  * ``x0var``    — per-cell Laplacian energy of x̂₀ estimated at the crossing.
                   Marks detail-critical / high-contrast (moiré-risk) cells.
                   Known risk: over-selects busy texture — measured here.
  * ``combo``    — sum of the two, each normalized.

Score → mask: bisected threshold → morphological open (drop specks) → close
(fill gaps) → dilate 1 cell (boundary margin) → FINAL fraction lands on target.
Static arms ``rect`` (center default — the status quo) and ``rect_miss`` (null)
run the same adaptive code path with fixed cells, so timing is apples-to-apples.

Prompts: the centered default (easy mode — derived must not regress vs rect)
plus hard dataset captions from ``hard_prompts.json`` (multi-subject spread,
pattern-heavy periphery) with hand-annotated per-seed face boxes as the
subject metric. Gate: on hard prompts a derived source clearly beats static
``rect`` on subject RMSE at matched fraction (and trivially beats rect_miss);
on the default prompt it stays ≈ rect. Overlays must look subject-following.

Usage:
  uv run python -m bench.foveated.probe_mask_sources
  uv run python -m bench.foveated.probe_mask_sources --prompts default channel6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result
from bench.foveated.probe_mask_shapes import (
    _cells_px,
    _cells_to_mask4,
    _dilate,
    _erode,
    _rect_cells,
    _rect_miss_cells,
    _ring_px,
    _rmse_sel,
)
from bench.foveated.probe_token_merge import FoveatedTokenMerge, dit_forward
from bench.foveated.probe_velocity_foveation import (
    _grid,
    _lap_energy,
    _stack,
    _to_pil,
)

log = logging.getLogger("bench.foveated.sources")
logging.basicConfig(level=logging.INFO, format="%(message)s")

SOURCES = ["rect", "rect_miss", "x0var", "cfgdelta", "combo"]


# Score → compact mask: promoted to networks/foveated.py (2026-07-03) — the
# bench imports the shipped home so this probe regression-tests the production
# mask pipeline verbatim.
from networks.foveated import score_to_cells  # noqa: E402, F401


# ── Adaptive denoise: accumulate signals full-res, build mask at crossing ───────


@torch.no_grad()
def denoise_adaptive(
    anima,
    x5_init,
    embed,
    neg_embed,
    sigmas,
    guidance,
    device,
    pad,
    *,
    source: str,
    static_cells: torch.Tensor | None,
    sigma_c: float,
    fovea_frac: float,
    pool: int,
):
    """Euler CFG loop. Above σ_c: full-res forwards, accumulating the per-cell
    source signals. At the crossing: build the mask (derived or static), build
    the merger, continue merged. Returns latent, per-arm info, the cells, and
    the raw score map (None for static arms)."""
    x5 = x5_init
    h_lat, w_lat = x5.shape[-2:]
    hp, wp = h_lat // pool, w_lat // pool
    times = {"full_ms": [], "merged_ms": []}
    cfg_acc = torch.zeros(h_lat, w_lat, device=device)
    x0_lap: torch.Tensor | None = None
    merger: FoveatedTokenMerge | None = None
    cells: torch.Tensor | None = None
    mask4 = None
    score_used: np.ndarray | None = None
    frac_actual = None

    def cell_mean(px_map: torch.Tensor) -> np.ndarray:
        return F.avg_pool2d(px_map[None, None], pool)[0, 0].float().cpu().numpy()

    t0 = time.perf_counter()
    n = len(sigmas) - 1
    for i in range(n):
        sigma = float(sigmas[i])
        if merger is None and sigma <= sigma_c:
            # ── crossing: build the mask ──
            if source == "rect":
                cells = static_cells
            elif source == "rect_miss":
                cells = static_cells
            else:
                s_cfg = cell_mean(cfg_acc)
                s_x0 = (
                    cell_mean(x0_lap)
                    if x0_lap is not None
                    else np.zeros((hp, wp), np.float32)
                )
                if source == "cfgdelta":
                    score = s_cfg
                elif source == "x0var":
                    score = s_x0
                else:  # combo
                    score = s_cfg / max(s_cfg.sum(), 1e-9) + s_x0 / max(
                        s_x0.sum(), 1e-9
                    )
                cells, _ = score_to_cells(score, fovea_frac)
                score_used = score
            frac_actual = float(cells.float().mean())
            mask4 = _cells_to_mask4(cells, pool, device)
            merger = FoveatedTokenMerge(anima, mask4, device)

        use_merge = merger is not None
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        m = merger if use_merge else None
        v_c = dit_forward(anima, x5, sigma, embed, pad, m)
        v_u = dit_forward(anima, x5, sigma, neg_embed, pad, m)
        end.record()
        end.synchronize()
        times["merged_ms" if use_merge else "full_ms"].append(start.elapsed_time(end))
        v = (v_u + guidance * (v_c - v_u)).float()

        if not use_merge:
            # accumulate source signals from the full-res forwards
            cfg_acc += (v_c.float() - v_u.float()).abs().mean(dim=(0, 1, 2))
            x0 = (x5.float() - sigma * v).squeeze(2)[0]  # (C, H, W)
            lap = (
                -4 * x0[:, 1:-1, 1:-1]
                + x0[:, :-2, 1:-1]
                + x0[:, 2:, 1:-1]
                + x0[:, 1:-1, :-2]
                + x0[:, 1:-1, 2:]
            )
            e = torch.zeros(h_lat, w_lat, device=device)
            e[1:-1, 1:-1] = (lap**2).mean(dim=0)
            x0_lap = e  # keep the LAST pre-crossing estimate

        x5 = (x5.float() + v * (float(sigmas[i + 1]) - sigma)).to(torch.bfloat16)

    if device.type == "cuda":
        torch.cuda.synchronize()
    e2e = time.perf_counter() - t0

    # P0 final readout: periphery read from the merged representation.
    x4 = x5.float().squeeze(2)
    low = F.avg_pool2d(x4, pool)
    up = F.interpolate(low, size=x4.shape[-2:], mode="bicubic", align_corners=False)
    x4 = mask4 * x4 + (1.0 - mask4) * up
    x5 = x4.unsqueeze(2).to(torch.bfloat16)

    info = {
        "e2e_s": e2e,
        "full_fwd_ms": float(np.mean(times["full_ms"])),
        "merged_fwd_ms": float(np.mean(times["merged_ms"]))
        if times["merged_ms"]
        else None,
        "n_merged_steps": len(times["merged_ms"]),
        "fovea_frac_actual": frac_actual,
    }
    return x5, info, cells, score_used


@torch.no_grad()
def denoise_baseline(anima, x5_init, embed, neg_embed, sigmas, guidance, device, pad):
    x5 = x5_init
    t0 = time.perf_counter()
    for i in range(len(sigmas) - 1):
        sigma = float(sigmas[i])
        v_c = dit_forward(anima, x5, sigma, embed, pad, None)
        v_u = dit_forward(anima, x5, sigma, neg_embed, pad, None)
        v = (v_u + guidance * (v_c - v_u)).float()
        x5 = (x5.float() + v * (float(sigmas[i + 1]) - sigma)).to(torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return x5, time.perf_counter() - t0


# ── Visual helpers ──────────────────────────────────────────────────────────────


def _draw_boxes(img: Image.Image, boxes, color="red", width=5) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    for b in boxes:
        d.rectangle(list(b), outline=color, width=width)
    return out


def _draw_mask_and_boxes(img, cells, px_per_cell, boxes):
    m = _cells_px(cells, px_per_cell)
    edge = _dilate(_dilate(m & ~_erode(m)))
    arr = np.array(img)
    arr[edge] = (255, 255, 255)
    return _draw_boxes(Image.fromarray(arr), boxes)


def _subject_strip(img: Image.Image, boxes, h: int = 256) -> Image.Image:
    crops = []
    for b in boxes:
        c = img.crop(tuple(b))
        crops.append(c.resize((int(c.width * h / c.height), h), Image.LANCZOS))
    strip = Image.new("RGB", (sum(c.width for c in crops), h), "white")
    x = 0
    for c in crops:
        strip.paste(c, (x, 0))
        x += c.width
    return strip


def _score_png(score: np.ndarray, path: Path):
    hi = max(float(np.percentile(score, 99)), 1e-9)
    g = np.clip(score / hi, 0, 1)
    Image.fromarray((g * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST).save(
        path
    )


# ── Main ────────────────────────────────────────────────────────────────────────


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
    ap.add_argument("--fovea_frac", type=float, default=0.35)
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
    ap.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    ap.add_argument("--label", default="p2b")
    args = ap.parse_args()

    device = torch.device(args.device)
    repo = Path(__file__).resolve().parents[2]
    hard = json.loads((Path(__file__).parent / "hard_prompts.json").read_text())

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
    oracle = _rect_cells(hp, wp, args.fovea_frac, *args.fovea_center)

    # Per-prompt: caption text + subject boxes per seed.
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

            miss_gen = torch.Generator().manual_seed(seed * 1000 + 7)
            static = {
                "rect": oracle,
                "rect_miss": _rect_miss_cells(
                    hp, wp, args.fovea_frac, oracle, miss_gen
                ),
            }

            row = {"prompt": pname, "seed": seed, "arms": {}}
            images, cells_by = {}, {}
            for src in args.sources:
                log.info(f"  {src} ...")
                lat, info, cells, score = denoise_adaptive(
                    anima, init.clone(), embed, neg_embed, sigmas,
                    args.guidance_scale, device, pad,
                    source=src, static_cells=static.get(src),
                    sigma_c=args.sigma_c, fovea_frac=args.fovea_frac,
                    pool=args.pool,
                )  # fmt: skip
                img = _to_pil(decode_latent(vae, lat, device))
                img.save(run_dir / f"{pname}_seed{seed}_{src}.png")
                images[src], cells_by[src] = img, cells
                if score is not None:
                    _score_png(score, run_dir / f"{pname}_seed{seed}_{src}_score.png")
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
                row["arms"][src] = m
                log.info(
                    f"    {src}: subject={m['subject_rmse']:.4f} "
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
                            images[s], cells_by[s], px_per_cell, boxes
                        ),
                        f"{s} (cover {row['arms'][s]['subject_cover']:.0%})",
                    )
                    for s in args.sources
                ]
            )
            strip_row = _grid(
                [(_subject_strip(base_img, boxes), "subjects baseline")]
                + [
                    (_subject_strip(images[s], boxes), f"subjects {s}")
                    for s in args.sources
                ]
            )
            _stack([full_row, strip_row]).save(
                run_dir / f"compare_{pname}_seed{seed}.png"
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Aggregate: per prompt × source ──
    agg = {}
    for pname in args.prompts:
        rows = [r for r in results if r["prompt"] == pname]
        agg[pname] = {
            s: {
                k: float(np.mean([r["arms"][s][k] for r in rows]))
                for k in (
                    "subject_rmse",
                    "own_mask_rmse",
                    "subject_cover",
                    "fovea_frac_actual",
                    "periph_lap_ratio",
                    "ring_lap_ratio",
                    "e2e_speedup",
                )
            }
            for s in args.sources
        }
    any_nan = any(m["nan_inf"] for r in results for m in r["arms"].values())
    if any_nan:
        verdict = "HARD_FAIL: NaN/Inf."
    else:
        parts = []
        for pname in args.prompts:
            a = agg[pname]
            parts.append(
                f"[{pname}] "
                + ", ".join(
                    f"{s} {a[s]['subject_rmse']:.4f}(cov {a[s]['subject_cover']:.0%})"
                    for s in args.sources
                )
            )
        verdict = (
            "Visual call on compare_*.png. Subject RMSE by prompt: "
            + "; ".join(parts)
            + ". Gate: on hard prompts a derived source beats static rect clearly "
            "and rect_miss trivially; on default it stays ≈ rect; overlays "
            "subject-following."
        )

    metrics = {
        "sigma_c": args.sigma_c,
        "fovea_frac": args.fovea_frac,
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
