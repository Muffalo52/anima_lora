"""Phase-2 probe — mask-shape and mask-placement controls for deferred foveation.

P0/1b used a single oracle-placed center rect. Before building real mask
*sources* (localizer / x̂₀-variance), establish what a mask's shape and
placement are each worth on the real 1b merge path, at matched fovea fraction
(cost is fraction-only: attention is global, so layout is free):

  * ``rect``      — oracle single rect (P0 default, centered on the subject).
  * ``rect_miss`` — same rect, placement chosen to MINIMIZE overlap with the
                    oracle rect (anti-oracle). The placement null: quantifies
                    what a mis-aimed mask costs on the subject.
  * ``multi3``    — 3 random non-overlapping rects, frac/3 each. Moderate
                    fragmentation (the "multiple subjects" primitive).
  * ``scatter``   — random 2×2-token cells at the same fraction. Maximum
                    boundary perimeter — the shape stress test.

Comparing ``rect`` vs ``rect_miss`` isolates placement; ``rect_miss`` vs
``multi3`` vs ``scatter`` isolates fragmentation with placement held random.
If fragmentation is free, oracle multi-rect is safe by composition and Phase-2
mask-source work reduces to pure saliency; derived masks must then beat the
placement null to justify their complexity.

Readouts per arm (vs same-seed full-compute baseline):
  * own-mask RMSE   — fidelity inside the arm's own fovea (mechanism holds for
                      this shape?),
  * subject RMSE    — fixed oracle-rect box in EVERY arm (what the mask buys
                      on the actual subject), plus subject coverage,
  * own-periphery Laplacian ratio (blur), boundary-ring Laplacian ratio
    (seam/halo flag: ±1 cell around the mask edge),
  * fwd/e2e speed (matched-cost sanity: all shapes ≈ same token count).

Verdict is the usual eyeball on ``compare_seed*.png`` (row 1: full images with
mask edges overlaid; row 2: the fixed subject crop from every arm).

Usage:
  uv run python -m bench.foveated.probe_mask_shapes
  uv run python -m bench.foveated.probe_mask_shapes --sigma_c 0.5 --seeds 40 41 42
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
from bench.foveated.probe_token_merge import FoveatedTokenMerge, denoise, dit_forward
from bench.foveated.probe_velocity_foveation import (
    _fovea_box_px,
    _grid,
    _lap_energy,
    _rmse,
    _stack,
    _to_pil,
)

log = logging.getLogger("bench.foveated.masks")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── Mask builders (on the 2×2-token cell grid; 1 cell = pool latent px) ─────────


# Promoted to networks/foveated.py (2026-07-03) — the bench imports the shipped
# home (bespoke-loop mirroring lesson) and keeps the historical leading-underscore
# aliases so the sibling probes' imports stay stable.
from networks.foveated import (  # noqa: E402
    cells_to_mask4 as _cells_to_mask4,
    dilate as _dilate,
    erode as _erode,
    rect_cells as _rect_cells,
)


def _rect_miss_cells(hp, wp, frac, oracle: torch.Tensor, gen) -> torch.Tensor:
    """Anti-oracle: among random placements, keep the one overlapping the oracle
    rect least (usually zero at frac 0.35)."""
    best, best_ov = None, None
    for _ in range(256):
        cy, cx = torch.rand(2, generator=gen).tolist()
        m = _rect_cells(hp, wp, frac, cy, cx)
        ov = int((m & oracle).sum())
        if best is None or ov < best_ov:
            best, best_ov = m, ov
        if best_ov == 0:
            break
    return best


def _multi_rect_cells(hp, wp, frac, k: int, gen) -> torch.Tensor:
    """k random rects of frac/k each; rejection-sampled non-overlapping (the
    last try is accepted overlapping if sampling fails — actual frac reported)."""
    m = torch.zeros(hp, wp, dtype=torch.bool)
    for _ in range(k):
        for _try in range(500):
            cy, cx = torch.rand(2, generator=gen).tolist()
            r = _rect_cells(hp, wp, frac / k, cy, cx)
            if not (r & m).any():
                break
        m |= r
    return m


def _scatter_cells(hp, wp, n_cells: int, gen) -> torch.Tensor:
    idx = torch.randperm(hp * wp, generator=gen)[:n_cells]
    m = torch.zeros(hp * wp, dtype=torch.bool)
    m[idx] = True
    return m.reshape(hp, wp)


# ── Pixel-space mask metrics/overlays ────────────────────────────────────────────


def _cells_px(cells: torch.Tensor, px_per_cell: int) -> np.ndarray:
    return np.kron(cells.numpy(), np.ones((px_per_cell, px_per_cell), dtype=bool))


def _ring_px(cells: torch.Tensor, px_per_cell: int) -> np.ndarray:
    """±1-cell band around the mask boundary, in image pixels."""
    m = cells.numpy()
    return _cells_px(torch.from_numpy(_dilate(m) & ~_erode(m)), px_per_cell)


def _rmse_sel(a: Image.Image, b: Image.Image, sel: np.ndarray) -> float:
    fa = np.asarray(a, dtype=np.float32) / 255.0
    fb = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.sqrt(np.mean(((fa - fb) ** 2)[sel])))


def _draw_mask_edges(img: Image.Image, cells: torch.Tensor, px_per_cell: int):
    m = _cells_px(cells, px_per_cell)
    edge = m & ~_erode(m)
    edge = _dilate(_dilate(edge))  # ~3px line
    arr = np.array(img)
    arr[edge] = (255, 255, 255)
    return Image.fromarray(arr)


# ── Main ────────────────────────────────────────────────────────────────────────


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
    ap.add_argument("--sigma_c", type=float, default=0.75)
    ap.add_argument("--fovea_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--num_rects", type=int, default=3)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["rect", "rect_miss", "multi3", "scatter"],
        choices=["rect", "rect_miss", "multi3", "scatter"],
    )
    ap.add_argument("--label", default="p2")
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

    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.to(device)

    h_lat, w_lat = args.height // 8, args.width // 8
    if h_lat % args.pool or w_lat % args.pool:
        raise SystemExit(f"latent {h_lat}x{w_lat} not divisible by pool={args.pool}")
    hp, wp = h_lat // args.pool, w_lat // args.pool
    px_per_cell = args.pool * 8

    oracle = _rect_cells(hp, wp, args.fovea_frac, *args.fovea_center)
    n_oracle = int(oracle.sum())
    subject_box = _fovea_box_px(_cells_to_mask4(oracle, args.pool, device))
    subject_sel = np.zeros((args.height, args.width), dtype=bool)
    subject_sel[subject_box[1] : subject_box[3], subject_box[0] : subject_box[2]] = True

    def build_cells(name: str, seed: int) -> torch.Tensor:
        # zlib.crc32, not hash() — string hash is salted per process.
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
        raise SystemExit(f"unknown arm {name}")

    # ── Startup invariants (as in 1b: custom runner ≡ anima(), all-fovea ≡ id) ──
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        probe_x = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=torch.Generator(device="cpu").manual_seed(0),
            device=device,
            dtype=torch.bfloat16,
        )
        t_vec = probe_x.new_full((1,), 0.7)
        ref = anima(probe_x, t_vec, embed, padding_mask=pad)
        mine = dit_forward(anima, probe_x, 0.7, embed, pad, None)
        d1 = float((ref - mine).abs().max())
        ones = torch.ones(1, 1, h_lat, w_lat, device=device)
        all_fovea = FoveatedTokenMerge(anima, ones, device)
        merged_id = dit_forward(anima, probe_x, 0.7, embed, pad, all_fovea)
        d2 = float((mine - merged_id).abs().max())
    log.info(f"invariants: custom-vs-anima max|Δ|={d1:.2e}, all-fovea-merge={d2:.2e}")
    if d1 > 5e-2 or d2 > 1e-6:
        raise SystemExit("startup invariant failed — merge plumbing is wrong")

    run_dir = make_run_dir("foveated", label=args.label)
    per_seed = []
    for seed in args.seeds:
        log.info(f"\n=== seed {seed} ===")
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=torch.Generator(device="cpu").manual_seed(seed),
            device=device,
            dtype=torch.bfloat16,
        )
        images: dict[str, Image.Image] = {}
        cells_by_arm: dict[str, torch.Tensor] = {}
        row = {"seed": seed, "arms": {}}

        for name in ["baseline", *args.arms]:
            log.info(f"  {name} ...")
            merged = name != "baseline"
            cells = build_cells(name, seed) if merged else None
            mask4 = _cells_to_mask4(cells, args.pool, device) if merged else None
            merger = FoveatedTokenMerge(anima, mask4, device) if merged else None
            lat, times, e2e = denoise(
                anima,
                init.clone(),
                embed,
                neg_embed,
                sigmas,
                args.guidance_scale,
                device,
                pad,
                merger,
                mask4,
                args.pool,
                args.sigma_c if merged else None,
            )
            img = _to_pil(decode_latent(vae, lat, device))
            img.save(run_dir / f"seed{seed}_{name}.png")
            images[name] = img
            m = {
                "e2e_s": e2e,
                "full_fwd_ms": float(np.mean(times["full_ms"])),
                "merged_fwd_ms": float(np.mean(times["merged_ms"]))
                if times["merged_ms"]
                else None,
                "nan_inf": bool(torch.isnan(lat).any() or torch.isinf(lat).any()),
                "latent_std": float(lat.float().std()),
            }
            if merged:
                cells_by_arm[name] = cells
                m["n_cells"] = int(cells.sum())
                m["fovea_frac_actual"] = float(cells.float().mean())
                m["subject_cover"] = float((cells & oracle).sum() / oracle.sum())
                m["tokens_merged"] = merger.L_red
                m["token_ratio"] = merger.L_red / merger.L_full
            row["arms"][name] = m

        base = images["baseline"]
        base_e2e = row["arms"]["baseline"]["e2e_s"]
        for name in args.arms:
            img, m, cells = images[name], row["arms"][name], cells_by_arm[name]
            own_px = _cells_px(cells, px_per_cell)
            ring = _ring_px(cells, px_per_cell)
            m["own_mask_rmse"] = _rmse_sel(img, base, own_px)
            m["subject_rmse"] = _rmse(img, base, subject_box)
            m["image_rmse"] = _rmse(img, base)
            m["periph_lap_ratio"] = _lap_energy(img, ~own_px) / max(
                _lap_energy(base, ~own_px), 1e-9
            )
            m["ring_lap_ratio"] = _lap_energy(img, ring) / max(
                _lap_energy(base, ring), 1e-9
            )
            m["e2e_speedup"] = base_e2e / m["e2e_s"]
            log.info(
                f"    {name}: own={m['own_mask_rmse']:.4f}  "
                f"subject={m['subject_rmse']:.4f} (cover {m['subject_cover']:.0%})  "
                f"periph_sharp×{m['periph_lap_ratio']:.2f}  "
                f"ring×{m['ring_lap_ratio']:.2f}  e2e ×{m['e2e_speedup']:.2f}"
            )
        per_seed.append(row)

        full_row = _grid(
            [(base, "baseline")]
            + [
                (_draw_mask_edges(images[n], cells_by_arm[n], px_per_cell), n)
                for n in args.arms
            ]
        )
        crop_row = _grid(
            [
                (images[n].crop(subject_box), f"subject {n}")
                for n in ["baseline", *args.arms]
            ]
        )
        _stack([full_row, crop_row]).save(run_dir / f"compare_seed{seed}.png")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate ──
    def _mean(name, key):
        vals = [r["arms"][name].get(key) for r in per_seed if r["arms"][name].get(key)]
        return float(np.mean(vals)) if vals else None

    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    agg = {
        n: {
            k: _mean(n, k)
            for k in (
                "own_mask_rmse",
                "subject_rmse",
                "subject_cover",
                "periph_lap_ratio",
                "ring_lap_ratio",
                "fovea_frac_actual",
                "e2e_speedup",
            )
        }
        for n in args.arms
    }
    verdict = (
        "HARD_FAIL: NaN/Inf in merged arms."
        if any_nan
        else (
            "NO_HARD_DIVERGENCE — visual call on compare_seed*.png. "
            "Shape axis (own-mask RMSE, matched frac): "
            + ", ".join(f"{n} {v['own_mask_rmse']:.4f}" for n, v in agg.items())
            + ". Placement axis (subject RMSE): "
            + ", ".join(f"{n} {v['subject_rmse']:.4f}" for n, v in agg.items())
            + ". Gate: shapes hold own-mask fidelity ≈ oracle rect (fragmentation "
            "free) AND rect_miss subject damage >> rect (placement worth buying)."
        )
    )

    metrics = {
        "sigma_c": args.sigma_c,
        "fovea_frac": args.fovea_frac,
        "num_rects": args.num_rects,
        "pool_latent_px": args.pool,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "subject_box_px": list(subject_box),
        "cell_grid": [hp, wp],
        "startup_invariants": {"custom_vs_anima": d1, "all_fovea_identity": d2},
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
        artifacts=[p.name for p in sorted(run_dir.glob("*.png"))],
        device=device,
    )
    log.info("\n" + "=" * 70)
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
