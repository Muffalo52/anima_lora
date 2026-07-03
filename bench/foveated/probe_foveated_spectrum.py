"""Phase-1a probe — foveated Spectrum: spend the refresh budget on the fovea.

Gate from ``docs/proposal/foveated_denoise.md`` §Phase 1a, premise measured by 0b
(aggressive Spectrum's forecast error concentrates ×1.6–1.7 in the fovea while
the periphery — smooth, slowly-evolving — is exactly where Chebyshev
extrapolation is strongest). Two zero-surgery reallocation mechanisms, tested
separately and composed, all at (near-)matched actual-forward count:

* ``fovblend`` — the proposal's cheapest form: schedule unchanged; on actual
  forwards below σ_c the *emitted* features are mask-blended (fovea = actual,
  periphery = forecast; one extra ``final_layer`` pass, no block cost), while
  the forecaster history is still updated with the full actuals. Periphery
  stays full-res real detail (nothing is merged) — it just rides its own
  Chebyshev fit. Quality-side validation for the partial-recompute endgame.

* ``fovsea`` — refresh *timing* reallocation: the SEA trigger's distance metric
  is computed on the **fovea crop only**, δ auto-calibrated (existing
  ``solve_delta_for_refresh_ratio`` machinery) to the window schedule's own
  refresh fraction — same compute, refreshes fire when the *fovea* is changing
  rather than on a content-blind global rhythm.

* ``fov1a`` — both composed.

Hypothesis to falsify: at equal (±1) forward count, ``fovsea``/``fov1a``'s fovea
is closer to the full-compute baseline than plain aggressive Spectrum's. KILL
the 1a line (partial recompute inherits the merge line instead) if neither arm
beats ``spec``'s fovea, or if ``fovblend`` visibly damages the periphery
(forecast-emitted periphery drifting = the escalations are built on sand).

The loop is bench-local (reuses ``SpectrumPredictor`` / ``_spectrum_fast_forward``
/ the SEA helpers verbatim); production wiring into ``spectrum_denoise`` comes
after the gate passes, per the bespoke-loop-mirroring lesson.

Usage:
  uv run python -m bench.foveated.probe_foveated_spectrum
  uv run python -m bench.foveated.probe_foveated_spectrum --seeds 40 41 42 --sigma_c 0.5
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import numpy as np
import torch
from PIL import Image

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result
from bench.foveated.probe_spectrum_compose import _errmap, _rmse_region
from bench.foveated.probe_velocity_foveation import (
    _fovea_box_px,
    _grid,
    _outline,
    _stack,
    _to_pil,
    build_fovea_mask,
)

log = logging.getLogger("bench.foveated.fovspectrum")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@torch.no_grad()
def spectrum_arm(
    anima,
    init: torch.Tensor,
    embed: torch.Tensor,
    neg_embed: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    guidance: float,
    pad: torch.Tensor,
    device,
    *,
    schedule: str,  # "all" | "window" | "sea"
    warmup: int,
    window_size: float,
    flex_window: float,
    blend: bool = False,
    mask_tok5: torch.Tensor | None = None,  # (1,1,H_t,W_t,1), 1=fovea
    sigma_c: float | None = None,
    sea_delta: float | None = None,
    sea_dist_fn=None,  # callable(sea_now, sea_prev) -> float (trigger metric)
    record_trace: bool = False,
    trace_fns: dict | None = None,  # name -> dist fn, recorded alongside (P2t)
):
    """One denoise run. Mirrors ``spectrum_denoise``'s decision/forecast logic
    (window rule, SEA accumulate-reset, warmup/tail forcing) minus the
    side-channels this bare-DiT probe doesn't use."""
    from library.inference import sampling as inference_utils
    from networks.spectrum import _spectrum_fast_forward
    from networks.spectrum_forecast import SpectrumPredictor
    from networks.spectrum_sea import sea_filter

    m_basis, lam, w_blend = 3, 0.1, 0.3  # spectrum_denoise defaults
    num_steps = len(timesteps)
    stop_at = num_steps - 3
    do_cfg = guidance != 1.0
    latents = init.clone()

    captured = {}
    hook = anima.final_layer.register_forward_pre_hook(
        lambda module, args: captured.__setitem__("feat", args[0].detach().clone())
    )

    cond_fc = uncond_fc = None
    curr_ws, consec = window_size, 0
    sea_prev, sea_accum = None, 0.0
    trace: list[float] = []
    traces: dict[str, list[float]] = {k: [] for k in (trace_fns or {})}
    refresh_steps: list[int] = []

    try:
        for i, t in enumerate(timesteps):
            sigma = float(sigmas[i])

            if schedule == "sea" or record_trace or trace_fns:
                sea_now = sea_filter(latents[:, :, 0], sigma)
                if sea_prev is not None:
                    if schedule == "sea" or record_trace:
                        d = sea_dist_fn(sea_now, sea_prev)
                        sea_accum += d
                        if record_trace and warmup <= i < stop_at:
                            trace.append(d)
                    if trace_fns and warmup <= i < stop_at:
                        for k, fn in trace_fns.items():
                            traces[k].append(fn(sea_now, sea_prev))
                sea_prev = sea_now

            if schedule == "all" or i < warmup or i >= stop_at:
                actual = True
            elif schedule == "sea":
                actual = sea_accum >= sea_delta
            else:  # window
                actual = (consec + 1) % max(1, math.floor(curr_ws)) == 0

            t_exp = t.expand(latents.shape[0])
            if actual:
                noise_pred = anima(latents, t_exp, embed, padding_mask=pad)
                feat = captured["feat"]
                if cond_fc is None:
                    cond_fc = SpectrumPredictor(
                        m_basis, lam, w_blend, device, feat.shape[1:], num_steps
                    )
                do_blend = (
                    blend
                    and sigma_c is not None
                    and sigma <= sigma_c
                    and cond_fc.cheb.t_buf.numel() >= 2
                )
                if do_blend:
                    blended = (
                        mask_tok5 * feat + (1.0 - mask_tok5) * cond_fc.predict(float(i))
                    ).to(feat.dtype)
                    noise_pred = _spectrum_fast_forward(anima, t_exp, blended)
                cond_fc.update(float(i), feat)  # history stays anchored to actuals

                if do_cfg:
                    uncond_noise_pred = anima(
                        latents, t_exp, neg_embed, padding_mask=pad
                    )
                    ufeat = captured["feat"]
                    if uncond_fc is None:
                        uncond_fc = SpectrumPredictor(
                            m_basis, lam, w_blend, device, ufeat.shape[1:], num_steps
                        )
                    if do_blend and uncond_fc.cheb.t_buf.numel() >= 2:
                        ublended = (
                            mask_tok5 * ufeat
                            + (1.0 - mask_tok5) * uncond_fc.predict(float(i))
                        ).to(ufeat.dtype)
                        uncond_noise_pred = _spectrum_fast_forward(
                            anima, t_exp, ublended
                        )
                    uncond_fc.update(float(i), ufeat)
                    noise_pred = uncond_noise_pred + guidance * (
                        noise_pred - uncond_noise_pred
                    )

                if schedule == "window" and i >= warmup:
                    curr_ws = round(curr_ws + flex_window, 3)
                consec = 0
                sea_accum = 0.0
                refresh_steps.append(i)
            else:
                pred = cond_fc.predict(float(i))
                noise_pred = _spectrum_fast_forward(anima, t_exp, pred)
                if do_cfg:
                    upred = uncond_fc.predict(float(i))
                    uncond_noise_pred = _spectrum_fast_forward(anima, t_exp, upred)
                    noise_pred = uncond_noise_pred + guidance * (
                        noise_pred - uncond_noise_pred
                    )
                consec += 1

            latents = inference_utils.step(latents, noise_pred, sigmas, i).to(
                latents.dtype
            )
    finally:
        hook.remove()

    return latents, {
        "actual_forwards": len(refresh_steps),
        "refresh_steps": refresh_steps,
        "trace": trace,
        "traces": traces,
    }


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
    # Spectrum knobs — same aggressive point as the 0b probe.
    ap.add_argument("--window_size", type=float, default=3.0)
    ap.add_argument("--flex_window", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=int, default=4)
    # Foveation knobs — σ_c=0.5 is the Spectrum-composed knee (0b), blend arm only.
    ap.add_argument("--sigma_c", type=float, default=0.5)
    ap.add_argument("--fovea_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--label", default="p1a")
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
    mask4 = build_fovea_mask(
        h_lat, w_lat, args.pool, args.fovea_frac, device, center=args.fovea_center
    )
    fovea_box = _fovea_box_px(mask4)
    # Latent-px slice of the fovea rect (for the fovea-cropped SEA metric) and the
    # token-level blend mask (eager feature layout (B,1,H_t,W_t,D)).
    p = anima.patch_spatial
    lat_rows = torch.nonzero(mask4[0, 0].any(dim=1)).flatten()
    lat_cols = torch.nonzero(mask4[0, 0].any(dim=0)).flatten()
    fovea_lat_box = (
        int(lat_rows[0]),
        int(lat_rows[-1]) + 1,
        int(lat_cols[0]),
        int(lat_cols[-1]) + 1,
    )
    mask_tok5 = (
        (mask4[0, 0, ::p, ::p] > 0.5).float().view(1, 1, h_lat // p, w_lat // p, 1)
    )

    # δ target = the window schedule's own refresh fraction (like-for-like swap)
    stop_at = num_steps - 3
    target_ratio = window_decision_fraction(
        num_steps, args.warmup_steps, stop_at, args.window_size, args.flex_window
    )

    from networks.spectrum_sea import l1rel

    def fovea_sea_dist(now, prev):
        t_, b_, l_, r_ = fovea_lat_box
        return l1rel(now[..., t_:b_, l_:r_], prev[..., t_:b_, l_:r_])

    common = dict(
        warmup=args.warmup_steps,
        window_size=args.window_size,
        flex_window=args.flex_window,
        sea_dist_fn=fovea_sea_dist,  # spec's trace recording needs it too
    )
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    run_dir = make_run_dir("foveated", label=args.label)
    periph = np.ones((args.height, args.width), dtype=bool)
    periph[fovea_box[1] : fovea_box[3], fovea_box[0] : fovea_box[2]] = False
    fovea_px = ~periph

    sea_delta = None  # calibrated from the first seed's spec trace
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
        spec_info = run("spec", schedule="window", record_trace=True)
        if sea_delta is None:
            sea_delta = solve_delta_for_refresh_ratio(spec_info["trace"], target_ratio)
            log.info(
                f"  fovea-SEA delta={sea_delta:.4g} (target refresh_ratio "
                f"{target_ratio:.2f} over {len(spec_info['trace'])} decision steps)"
            )
        run(
            "fovblend",
            schedule="window",
            blend=True,
            mask_tok5=mask_tok5,
            sigma_c=args.sigma_c,
        )
        run("fovsea", schedule="sea", sea_delta=sea_delta)
        run(
            "fov1a",
            schedule="sea",
            sea_delta=sea_delta,
            blend=True,
            mask_tok5=mask_tok5,
            sigma_c=args.sigma_c,
        )

        base = images["full"]
        errmaps = []
        for name in ("spec", "fovblend", "fovsea", "fov1a"):
            img, m = images[name], row["arms"][name]
            m["fovea_rmse_vs_full"] = _rmse_region(img, base, fovea_px)
            m["periph_rmse_vs_full"] = _rmse_region(img, base, periph)
            errmaps.append((_errmap(img, base), f"|{name} - full|"))
            log.info(
                f"    {name}: fwd={m['actual_forwards']}  "
                f"fovea_rmse={m['fovea_rmse_vs_full']:.4f}  "
                f"periph_rmse={m['periph_rmse_vs_full']:.4f}  "
                f"refresh@{m['refresh_steps']}"
            )
        per_seed.append(row)

        names = ["full", "spec", "fovblend", "fovsea", "fov1a"]
        full_row = _grid(
            [
                (
                    _outline(images[n], fovea_box),
                    f"{n} ({row['arms'][n]['actual_forwards']} fwd)",
                )
                for n in names
            ]
        )
        crop_row = _grid([(images[n].crop(fovea_box), f"fovea {n}") for n in names])
        _stack([full_row, crop_row]).save(run_dir / f"compare_seed{seed}.png")
        _grid(errmaps).save(run_dir / f"errmap_seed{seed}.png")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate + gate ──
    def _mean(name, key):
        return float(np.mean([r["arms"][name][key] for r in per_seed]))

    agg = {
        n: {
            "fovea_rmse": _mean(n, "fovea_rmse_vs_full"),
            "periph_rmse": _mean(n, "periph_rmse_vs_full"),
            "fwd": _mean(n, "actual_forwards"),
        }
        for n in ("spec", "fovblend", "fovsea", "fov1a")
    }
    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    spec_f = agg["spec"]["fovea_rmse"]
    if any_nan:
        verdict = "HARD_FAIL: NaN/Inf."
    else:
        verdict = (
            "Visual call on compare_seed*/errmap_seed*.png. GATE — fovea RMSE vs "
            f"full at matched fwd: spec {spec_f:.4f} ({agg['spec']['fwd']:.1f} fwd); "
            + ", ".join(
                f"{n} {agg[n]['fovea_rmse']:.4f} ({agg[n]['fovea_rmse'] - spec_f:+.4f}, "
                f"{agg[n]['fwd']:.1f} fwd)"
                for n in ("fovblend", "fovsea", "fov1a")
            )
            + ". PASS needs fovsea or fov1a < spec at fwd ≤ spec+1; fovblend must "
            "not damage the periphery (it's the partial-recompute foundation)."
        )

    metrics = {
        "spectrum": {
            "window_size": args.window_size,
            "flex_window": args.flex_window,
            "warmup_steps": args.warmup_steps,
        },
        "sigma_c_blend": args.sigma_c,
        "fovea_frac": args.fovea_frac,
        "sea_delta": sea_delta,
        "target_refresh_ratio": target_ratio,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "fovea_box_px": list(fovea_box),
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
