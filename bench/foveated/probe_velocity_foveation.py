"""Phase-0 quality-contract probe for foveated denoising (see ``bench/foveated/plan.md``).

Emulates masked block-level token merging (the "build B" mechanism) at the sampler
boundary, with zero attention surgery: run the standard full-res Euler loop, and once
σ ≤ σ_c

  * **every step:** the DiT evaluates on the merged-token view — periphery replaced by
    ``pool(z)`` (``tome_eval``, what real merged attention would see) or on the true
    latent (``frozen``) — and the periphery *velocity* is replaced by its per-group
    pooled broadcast, so all tokens in a 2×2-token (default ``--pool 4`` latent px)
    group share one update, exactly what merged computation would hand them. The latent
    itself is NEVER rewritten — its white noise stays on-manifold.
  * **at decode (once):** the periphery is read from the merged representation
    (``z ← pool(z)``), stripping the HF noise/detail that pooled velocities can never
    remove (``--no_final_pool`` demonstrates that frozen-noise failure).

History: the first version instead *rebuilt* the periphery latent at the σ_c crossing
from pooled x̂₀ + variance-renormalized pooled ε. Run 20260703-1347 falsified both
rebuild ops — group-constant noise is off-manifold (f²× per-mode low-frequency power);
the DiT reads the blocks as content and the periphery never converges (rainbow block
noise). They remain available as ``split_renorm`` / ``pool_plain`` controls.

The fovea (a center rectangle covering ``--fovea_frac`` of the area, snapped to the pool
grid) is untouched, so with the same seed the fovea should stay near-identical to the
baseline; how much periphery coarsening *leaks into* the fovea via attention is one of
the two things this probe measures. The other is whether the periphery degrades to
acceptable "soft half-res detail" or to garbage (grain / seams / tone shift).

No speedup here — the full grid is computed every step. This is purely the quality gate
for Phase 1 (real ToMe-style merging inside blocks).

Verdict is a *visual* call (open ``compare_seed*.png`` at full res). Auto-metrics only
certify change / hard divergence — fovea-region RMSE vs baseline, periphery Laplacian
energy ratio (softness), NaN / latent-std blow-up.

Usage:
  uv run python -m bench.foveated.probe_velocity_foveation                 # tome_eval+frozen × σ_c 0.9/0.75/0.5
  uv run python -m bench.foveated.probe_velocity_foveation --sigma_c 0.75 --seeds 40 41 42
  uv run python -m bench.foveated.probe_velocity_foveation --transition_ops tome_eval --no_final_pool
  uv run python -m bench.foveated.probe_velocity_foveation --include_inverse   # coarsen the FOVEA (damage control)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result

log = logging.getLogger("bench.foveated.probe")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ── Foveation primitives ────────────────────────────────────────────────────────


def pool_broadcast(x4: torch.Tensor, f: int) -> torch.Tensor:
    """Average-pool a (B,C,H,W) tensor by f and broadcast back to full size —
    the "every latent pixel in a merged group shares one value" operator."""
    low = F.avg_pool2d(x4, f)
    return low.repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1)


def build_fovea_mask(
    h_lat: int, w_lat: int, f: int, frac: float, device, center=(0.5, 0.5)
) -> torch.Tensor:
    """(1,1,h_lat,w_lat) float mask, 1=fovea. A rectangle with the latent's aspect
    ratio covering ``frac`` of the area at ``center`` (cy, cx fractions, clamped
    inside), built on the pooled grid so no f×f group straddles the boundary."""
    if h_lat % f or w_lat % f:
        raise SystemExit(
            f"latent {h_lat}x{w_lat} not divisible by pool={f}; pick a resolution "
            f"whose H/8 and W/8 are multiples of {f}"
        )
    hp, wp = h_lat // f, w_lat // f
    s = math.sqrt(frac)
    rh = min(hp, max(1, round(hp * s)))
    rw = min(wp, max(1, round(wp * s)))
    cy, cx = center
    top = min(max(0, round(cy * hp - rh / 2)), hp - rh)
    left = min(max(0, round(cx * wp - rw / 2)), wp - rw)
    m = torch.zeros(1, 1, hp, wp, device=device)
    m[:, :, top : top + rh, left : left + rw] = 1.0
    return m.repeat_interleave(f, dim=-2).repeat_interleave(f, dim=-1)


def foveate_transition(
    x4: torch.Tensor,
    v4: torch.Tensor,
    sigma: float,
    mask4: torch.Tensor,
    f: int,
    op: str,
) -> torch.Tensor:
    """One-time periphery latent rebuild at the σ_c crossing (all float32 4D).

    FALSIFIED (run 20260703-1347): both rebuild ops produce off-manifold periphery.
    ``split_renorm`` matches per-pixel noise *variance* but not *distribution* —
    group-constant ε is rank-deficient with f² per-mode power concentrated in low
    frequencies; the DiT reads the blocks as content and its velocity on that input
    is garbage (rainbow block-noise in every arm). Kept as controls; the faithful
    build-B emulation is ``tome_eval``/``frozen`` (latent never rewritten)."""
    if op == "split_renorm":
        x0 = x4 - sigma * v4  # x̂₀ (velocity form: denoised = z − σ·v)
        eps = x0 + v4  # v = ε − x₀  ⇒  ε = x₀ + v
        x0_p = pool_broadcast(x0, f)
        eps_p = pool_broadcast(eps, f) * f  # unit per-pixel var — but block noise
        xp = (1.0 - sigma) * x0_p + sigma * eps_p
    elif op == "pool_plain":
        xp = pool_broadcast(x4, f)  # σ-stats trap control: noise std drops to 1/f
    else:  # "tome_eval" / "frozen" — latent untouched (the on-manifold modes)
        return x4
    return mask4 * x4 + (1.0 - mask4) * xp


# ── Denoise loop (Euler, velocity form, CFG; optional foveation) ────────────────


@torch.no_grad()
def denoise(
    anima,
    x5_init: torch.Tensor,
    embed: torch.Tensor,
    neg_embed: torch.Tensor,
    sigmas: torch.Tensor,
    guidance: float,
    device,
    mask4: torch.Tensor | None,
    pool: int,
    sigma_c: float | None,
    transition_op: str,
    final_pool: bool = True,
) -> tuple[torch.Tensor, dict]:
    """``sigma_c=None`` (or ``mask4=None``) is the full-res baseline. Below σ_c the
    periphery velocity is group-pooled every step. Modes:

    * ``tome_eval`` — the faithful build-B emulation: the latent is never rewritten
      (its white noise stays on-manifold); the DiT *evaluates* on a composite where
      the periphery is ``pool(z)`` — exactly the merged-token view real ToMe attention
      would see (noise std damped ×1/f, mildly blocky content).
    * ``frozen`` — DiT evaluates on the true latent (periphery HF noise of scale σ_c
      stays visible to the model all the way down); only the velocity is pooled.
    * ``split_renorm`` / ``pool_plain`` — falsified latent-rebuild controls
      (see ``foveate_transition``).

    With ``final_pool`` (default) the periphery is read from the merged representation
    at the end — ``z ← pool(z)`` once before decode — stripping the never-denoised HF
    noise/detail that pooled velocities cannot remove."""
    x5 = x5_init
    info: dict = {"transition_step": None, "extra_nfe": 0}

    def velocity(x: torch.Tensor, sigma_scalar: float) -> torch.Tensor:
        t = x.new_full((x.shape[0],), float(sigma_scalar))  # timestep == σ in [0,1]
        pad = torch.zeros(
            x.shape[0], 1, x.shape[-2], x.shape[-1], dtype=x.dtype, device=device
        )
        v_c = anima(x, t, embed, padding_mask=pad)
        if guidance != 1.0:
            v_u = anima(x, t, neg_embed, padding_mask=pad)
            return v_u + guidance * (v_c - v_u)
        return v_c

    n = len(sigmas) - 1
    for i in range(n):
        sigma = float(sigmas[i])
        foveated = sigma_c is not None and mask4 is not None and sigma <= sigma_c
        if foveated and transition_op == "tome_eval":
            # Evaluate on the merged-token view; step on the true latent.
            x4 = x5.float().squeeze(2)
            x_eval4 = mask4 * x4 + (1.0 - mask4) * pool_broadcast(x4, pool)
            v = velocity(x_eval4.unsqueeze(2).to(x5_init.dtype), sigma).float()
        else:
            v = velocity(x5, sigma).float()
        if foveated:
            x4, v4 = x5.float().squeeze(2), v.squeeze(2)
            if info["transition_step"] is None:
                info["transition_step"] = i
                info["transition_sigma"] = sigma
                x4_new = foveate_transition(x4, v4, sigma, mask4, pool, transition_op)
                if x4_new is not x4:  # latent changed — recompute v to match
                    x5 = x4_new.unsqueeze(2).to(x5_init.dtype)
                    v4 = velocity(x5, sigma).float().squeeze(2)
                    info["extra_nfe"] += 2 if guidance != 1.0 else 1
            v4 = mask4 * v4 + (1.0 - mask4) * pool_broadcast(v4, pool)
            v = v4.unsqueeze(2)
        dt = float(sigmas[i + 1]) - sigma
        x5 = (x5.float() + v * dt).to(torch.bfloat16)

    if sigma_c is not None and mask4 is not None and final_pool:
        # Read the periphery from the merged representation (strip frozen HF).
        # Bicubic upsample of the pooled grid — the paper's smooth Up(·) — rather
        # than nearest broadcast, so low-res periphery reads as soft blur, not mosaic.
        x4 = x5.float().squeeze(2)
        low = F.avg_pool2d(x4, pool)
        up = F.interpolate(low, size=x4.shape[-2:], mode="bicubic", align_corners=False)
        x4 = mask4 * x4 + (1.0 - mask4) * up
        x5 = x4.unsqueeze(2).to(torch.bfloat16)
    return x5, info


# ── Image helpers / cheap metrics (flag hard divergence only) ───────────────────


def _to_pil(pixels: torch.Tensor) -> Image.Image:
    arr = pixels.clamp(-1, 1).add(1).mul(127.5).round().byte()  # (C,H,W) [-1,1]
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())


def _fovea_box_px(mask4: torch.Tensor) -> tuple[int, int, int, int]:
    """Pixel-space (left, top, right, bottom) of the fovea rect (latent → ×8)."""
    m = mask4[0, 0]
    rows = torch.nonzero(m.any(dim=1)).flatten()
    cols = torch.nonzero(m.any(dim=0)).flatten()
    return (
        int(cols[0]) * 8,
        int(rows[0]) * 8,
        (int(cols[-1]) + 1) * 8,
        (int(rows[-1]) + 1) * 8,
    )


def _rmse(a: Image.Image, b: Image.Image, box=None) -> float:
    if box is not None:
        a, b = a.crop(box), b.crop(box)
    fa = np.asarray(a, dtype=np.float32) / 255.0
    fb = np.asarray(b, dtype=np.float32) / 255.0
    return float(np.sqrt(np.mean((fa - fb) ** 2)))


def _lap_energy(img: Image.Image, periphery: np.ndarray | None = None) -> float:
    """Mean squared 4-neighbour Laplacian (sharpness proxy), optionally restricted
    to a boolean periphery mask."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (
        -4 * g
        + np.roll(g, 1, 0)
        + np.roll(g, -1, 0)
        + np.roll(g, 1, 1)
        + np.roll(g, -1, 1)
    )
    lap = lap[1:-1, 1:-1]
    if periphery is not None:
        sel = periphery[1:-1, 1:-1]
        return float(np.mean(lap[sel] ** 2)) if sel.any() else float("nan")
    return float(np.mean(lap**2))


def _outline(img: Image.Image, box, color="white", width=4) -> Image.Image:
    out = img.copy()
    ImageDraw.Draw(out).rectangle(box, outline=color, width=width)
    return out


def _grid(cells: list[tuple[Image.Image, str]], cell_w: int = 512) -> Image.Image:
    """One-row contact sheet with a label strip under each (thumbnailed) cell."""
    strip = 28
    thumbs = []
    for im, label in cells:
        scale = cell_w / im.width
        thumbs.append(
            (im.resize((cell_w, int(im.height * scale)), Image.LANCZOS), label)
        )
    ch = max(t.height for t, _ in thumbs)
    sheet = Image.new("RGB", (cell_w * len(thumbs), ch + strip), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (t, label) in enumerate(thumbs):
        sheet.paste(t, (k * cell_w, 0))
        draw.text((k * cell_w + 6, ch + 6), label, fill="black")
    return sheet


def _stack(rows: list[Image.Image]) -> Image.Image:
    w = max(r.width for r in rows)
    sheet = Image.new("RGB", (w, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    return sheet


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
    # Foveation knobs
    ap.add_argument(
        "--sigma_c",
        type=float,
        nargs="+",
        default=[0.9, 0.75, 0.5],
        help="Cut σ values to sweep — periphery effort-sharing starts once σ ≤ σ_c.",
    )
    ap.add_argument(
        "--transition_ops",
        nargs="+",
        default=["tome_eval", "frozen"],
        choices=["tome_eval", "frozen", "split_renorm", "pool_plain"],
        help="Foveation mode below σ_c (tome_eval/frozen = on-manifold build-B "
        "emulations; split_renorm/pool_plain = falsified latent-rebuild controls).",
    )
    ap.add_argument(
        "--no_final_pool",
        action="store_true",
        help="Skip reading the periphery from the merged representation at decode "
        "(leaves the never-denoised HF noise in — frozen-noise demo).",
    )
    ap.add_argument(
        "--fovea_frac",
        type=float,
        default=0.35,
        help="Area fraction of the fovea rectangle.",
    )
    ap.add_argument(
        "--fovea_center",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        metavar=("CY", "CX"),
        help="Fovea rect center as height/width fractions (stand-in until the "
        "localizer mask — e.g. 0.38 0.5 to cover a portrait face+torso).",
    )
    ap.add_argument(
        "--pool",
        type=int,
        default=4,
        help="Merge-group edge in latent px (4 = 2×2 DiT tokens).",
    )
    ap.add_argument(
        "--include_inverse",
        action="store_true",
        help="Add a control arm that coarsens the FOVEA and keeps the periphery — "
        "shows what the probe looks like when it bites something important.",
    )
    ap.add_argument("--label", default="p0")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Build a fully-populated inference args namespace, then load the bare DiT ──
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
    iargs.lora_weight = None  # BARE DiT — Phase 0 tests the unadapted model
    iargs.sampler = "euler"

    ensure_text_strategies(args.text_encoder, MAX_CROSSATTN_TOKENS)

    log.info("Loading bare DiT (no LoRA, eager) ...")
    anima = load_dit_model(iargs, device, torch.bfloat16)

    log.info("Encoding prompt + negative prompt ...")
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
    mask4 = build_fovea_mask(
        h_lat, w_lat, args.pool, args.fovea_frac, device, center=args.fovea_center
    )
    fovea_box = _fovea_box_px(mask4)
    fovea_tokens = float(mask4.mean())
    # Effective token ratio a real Phase-1 merge would run at post-σ_c: each merged
    # group of (pool/patch_spatial)² DiT tokens would occupy one slot.
    groups = (args.pool // 2) ** 2  # DiT tokens per merged group (patch_spatial=2)
    eff_ratio = fovea_tokens + (1.0 - fovea_tokens) / groups
    log.info(
        f"fovea rect {fovea_box} px — {fovea_tokens:.0%} of tokens full effort; "
        f"post-σ_c effective token ratio if Phase-1 merged for real: {eff_ratio:.0%}"
    )

    # ── Arms: baseline + σ_c × transition_op (+ optional inverse-mask control) ──
    arms: list[dict] = [{"name": "baseline", "sigma_c": None, "op": "-", "mask": None}]
    for op in args.transition_ops:
        for sc in args.sigma_c:
            suffix = f"_{op}" if len(args.transition_ops) > 1 else ""
            arms.append(
                {"name": f"sc{sc:g}{suffix}", "sigma_c": sc, "op": op, "mask": mask4}
            )
    if args.include_inverse:
        arms.append(
            {
                "name": "inverse_sc0.75",
                "sigma_c": 0.75,
                "op": args.transition_ops[0],
                "mask": 1.0 - mask4,
            }
        )

    run_dir = make_run_dir("foveated", label=args.label)
    per_seed = []

    for seed in args.seeds:
        log.info(f"\n=== seed {seed} ===")
        seed_g = torch.Generator(device="cpu").manual_seed(seed)
        init = randn_tensor(
            (1, anima.LATENT_CHANNELS, 1, h_lat, w_lat),
            generator=seed_g,
            device=device,
            dtype=torch.bfloat16,
        )

        images: dict[str, Image.Image] = {}
        row = {"seed": seed, "arms": {}}
        for arm in arms:
            log.info(f"  {arm['name']} ...")
            lat, info = denoise(
                anima,
                init,
                embed,
                neg_embed,
                sigmas,
                args.guidance_scale,
                device,
                arm["mask"],
                args.pool,
                arm["sigma_c"],
                arm["op"],
                final_pool=not args.no_final_pool,
            )
            nan = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
            std = float(lat.float().std())
            px = decode_latent(vae, lat, device)
            img = _to_pil(px)
            img.save(run_dir / f"seed{seed}_{arm['name']}.png")
            images[arm["name"]] = img
            row["arms"][arm["name"]] = {
                "nan_inf": nan,
                "latent_std": std,
                **{k: v for k, v in info.items() if v is not None},
            }

        base = images["baseline"]
        periph = np.ones((args.height, args.width), dtype=bool)
        periph[fovea_box[1] : fovea_box[3], fovea_box[0] : fovea_box[2]] = False
        base_lap = _lap_energy(base, periph)
        for arm in arms[1:]:
            img = images[arm["name"]]
            m = row["arms"][arm["name"]]
            m["fovea_rmse_vs_base"] = _rmse(img, base, fovea_box)
            m["image_rmse_vs_base"] = _rmse(img, base)
            m["periph_lap_ratio"] = _lap_energy(img, periph) / max(base_lap, 1e-9)
            log.info(
                f"    {arm['name']}: fovea_rmse={m['fovea_rmse_vs_base']:.4f}  "
                f"periph_sharp×{m['periph_lap_ratio']:.2f}  "
                f"t_step={m.get('transition_step')}"
            )
        per_seed.append(row)

        full_row = _grid(
            [(_outline(images[a["name"]], fovea_box), a["name"]) for a in arms]
        )
        crop_row = _grid(
            [(images[a["name"]].crop(fovea_box), f"fovea {a['name']}") for a in arms]
        )
        _stack([full_row, crop_row]).save(run_dir / f"compare_seed{seed}.png")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate auto-verdict (hard divergence only; the verdict is visual) ──
    def _collect(key):
        return [
            r["arms"][a["name"]][key]
            for r in per_seed
            for a in arms[1:]
            if key in r["arms"][a["name"]] and not a["name"].startswith("inverse")
        ]

    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    stds = [m["latent_std"] for r in per_seed for m in r["arms"].values()]
    fovea_rmses = _collect("fovea_rmse_vs_base")
    hard_fail = any_nan or (max(stds) / max(min(stds), 1e-9) > 3.0)
    if hard_fail:
        verdict = (
            "HARD_FAIL: NaN/Inf or latent-std blow-up — foveated velocity sharing "
            "destabilizes the trajectory. Check per-arm metrics, then plan.md."
        )
    else:
        verdict = (
            "NO_HARD_DIVERGENCE: quality is a VISUAL call — open compare_seed*.png "
            "at full res. PASS needs (a) fovea crops ≈ baseline (composition intact; "
            f"fovea_rmse range {min(fovea_rmses):.4f}–{max(fovea_rmses):.4f} here), "
            "(b) periphery soft-but-clean (no grain/seams/tone shift). See plan.md "
            "kill criteria."
        )

    metrics = {
        "sigma_c_sweep": args.sigma_c,
        "transition_ops": args.transition_ops,
        "fovea_frac": args.fovea_frac,
        "fovea_token_share": fovea_tokens,
        "phase1_effective_token_ratio": eff_ratio,
        "pool_latent_px": args.pool,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "fovea_box_px": list(fovea_box),
        "per_seed": per_seed,
        "any_nan_inf": any_nan,
        "verdict": verdict,
    }
    artifacts = [p.name for p in sorted(run_dir.glob("*.png"))]
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=device,
    )

    log.info("\n" + "=" * 70)
    log.info(f"  arms: {[a['name'] for a in arms]}")
    if fovea_rmses:
        log.info(
            f"  fovea RMSE vs baseline: {min(fovea_rmses):.4f}–{max(fovea_rmses):.4f}"
        )
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}  (open compare_seed*.png)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
