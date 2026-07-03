"""Phase-0b Spectrum-compose smoke for foveated denoising.

Gate defined in ``docs/proposal/foveated_denoise.md`` §Phase 0b. Two questions,
one run:

1. **Headroom** (measured first, nearly free): where does aggressive Spectrum's
   forecast error actually land spatially? The foveated-Spectrum reframe assumes
   it concentrates in the fovea (detailed, fast-evolving regions) — if the error
   map of plain aggressive Spectrum vs the all-actual baseline is spatially
   uniform, spatial refresh allocation has little to buy. Read out as the
   fovea/periphery RMSE ratio of the ``spec`` arm plus an error-heatmap PNG.

2. **Compose smoke**: P0's velocity foveation wired into ``spectrum_denoise``
   (``foveation=`` hook points: composite eval-view on actual forwards below
   σ_c, output-side periphery pooling of every emitted v — forecast steps
   included, one forced actual forward at the σ_c crossing, bicubic merged
   readout at the end). KILL the composition if foveation adds visible fovea
   damage on top of Spectrum's, or if the σ_c feature discontinuity
   destabilizes cached steps (NaN / latent-std blow-up / garbage renders).

Arms (same seed each): ``full`` = all-actual forwards (warmup pinned to
num_steps — identical code path, zero forecasting), ``spec`` = aggressive
Spectrum, ``spec_fov_sc*`` = same Spectrum settings + foveation per σ_c.

Verdict is a *visual* call (open ``compare_seed*.png`` / ``errmap_seed*.png``
at full res). Auto-metrics certify change / hard divergence only — no metric
we have tracks Anima sample quality.

Usage:
  uv run python -m bench.foveated.probe_spectrum_compose
  uv run python -m bench.foveated.probe_spectrum_compose --sigma_c 0.75 --seeds 40 41 42
  uv run python -m bench.foveated.probe_spectrum_compose --window_size 2 --flex_window 0.25   # shipped-default Spectrum
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result
from bench.foveated.probe_velocity_foveation import (
    _fovea_box_px,
    _grid,
    _lap_energy,
    _outline,
    _rmse,
    _stack,
    _to_pil,
    build_fovea_mask,
    pool_broadcast,
)

log = logging.getLogger("bench.foveated.compose")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class VelocityFoveation:
    """P0 velocity foveation as the duck-typed adapter consumed by
    ``networks.spectrum.spectrum_denoise(foveation=...)``.

    All ops are 4D ``(B,C,H,W)`` on the latent grid — explicit ``squeeze(2)`` /
    ``unsqueeze(2)`` at the 5D boundary (CLAUDE.md dim-2 invariant). The latent
    is never rewritten; only what the compute sees / emits is pooled.
    """

    def __init__(self, mask4: torch.Tensor, pool: int, sigma_c: float):
        self.mask4 = mask4  # (1,1,h_lat,w_lat) float, 1 = fovea
        self.pool = pool
        self.sigma_c = sigma_c
        self.forced_step: int | None = None

    def _active(self, sigma: float) -> bool:
        return sigma <= self.sigma_c

    def force_actual(self, i: int, sigma: float) -> bool:
        """True exactly once — at the first step with σ ≤ σ_c."""
        if self._active(sigma) and self.forced_step is None:
            self.forced_step = i
            return True
        return False

    def eval_view(self, latents5: torch.Tensor, sigma: float) -> torch.Tensor:
        """Composite the DiT evaluates on below σ_c: periphery = pool(z)."""
        if not self._active(sigma):
            return latents5
        x4 = latents5.float().squeeze(2)
        comp = self.mask4 * x4 + (1.0 - self.mask4) * pool_broadcast(x4, self.pool)
        return comp.unsqueeze(2).to(latents5.dtype)

    def pool_velocity(self, v5: torch.Tensor, sigma: float) -> torch.Tensor:
        """Group-pool the periphery of an emitted v (actual or forecast)."""
        if not self._active(sigma):
            return v5
        v4 = v5.float().squeeze(2)
        v4 = self.mask4 * v4 + (1.0 - self.mask4) * pool_broadcast(v4, self.pool)
        return v4.unsqueeze(2).to(v5.dtype)

    def final_readout(self, latents5: torch.Tensor) -> torch.Tensor:
        """Read the periphery from the merged representation (bicubic up)."""
        if self.forced_step is None:  # σ_c never crossed
            return latents5
        x4 = latents5.float().squeeze(2)
        low = F.avg_pool2d(x4, self.pool)
        up = F.interpolate(low, size=x4.shape[-2:], mode="bicubic", align_corners=False)
        x4 = self.mask4 * x4 + (1.0 - self.mask4) * up
        return x4.unsqueeze(2).to(latents5.dtype)


def _rmse_region(a: Image.Image, b: Image.Image, region: np.ndarray) -> float:
    """RMSE over a boolean pixel region."""
    fa = np.asarray(a, dtype=np.float32) / 255.0
    fb = np.asarray(b, dtype=np.float32) / 255.0
    d2 = ((fa - fb) ** 2).mean(axis=2)
    return float(np.sqrt(d2[region].mean()))


def _errmap(a: Image.Image, b: Image.Image) -> Image.Image:
    """|a−b| heatmap (mean over RGB), auto-gained to its 99th percentile."""
    fa = np.asarray(a, dtype=np.float32) / 255.0
    fb = np.asarray(b, dtype=np.float32) / 255.0
    err = np.abs(fa - fb).mean(axis=2)
    hi = max(float(np.percentile(err, 99)), 1e-6)
    g = np.clip(err / hi, 0.0, 1.0)
    return Image.fromarray((g * 255).astype(np.uint8)).convert("RGB")


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
    # Spectrum knobs — aggressive by default (the hypothesis's operating point).
    ap.add_argument("--window_size", type=float, default=3.0)
    ap.add_argument("--flex_window", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=int, default=4)
    # Foveation knobs (P0v3 defaults)
    ap.add_argument("--sigma_c", type=float, nargs="+", default=[0.75, 0.5])
    ap.add_argument("--fovea_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--label", default="p0b")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── Model loading — identical to the P0 probe ──
    import inference as inference_mod
    from anima_lora import load_vae
    from diffusers.utils.torch_utils import randn_tensor
    from library.inference import sampling as inference_utils
    from library.inference.models import load_dit_model
    from library.inference.output import decode_latent
    from library.inference.sampler_context import SamplerSideChannels
    from library.inference.text import (
        MAX_CROSSATTN_TOKENS,
        ensure_text_strategies,
        prepare_text_inputs,
    )
    from networks.spectrum import spectrum_denoise

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
    iargs.lora_weight = None  # BARE DiT — same substrate as P0
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

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps = timesteps.to(device, torch.bfloat16)  # σ∈[0,1] — DiT time arg
    sigmas = sigmas.to(device)
    num_steps = len(timesteps)

    h_lat, w_lat = args.height // 8, args.width // 8
    mask4 = build_fovea_mask(
        h_lat, w_lat, args.pool, args.fovea_frac, device, center=args.fovea_center
    )
    fovea_box = _fovea_box_px(mask4)

    # Actual-forward counter: blocks only run on actual forwards (the cached
    # fast path is t_embedder → final_layer → unpatchify), so one hook on
    # blocks[0] counts them exactly. CFG doubles the per-step calls.
    fwd_counter = {"n": 0}
    hook = anima.blocks[0].register_forward_hook(
        lambda *a, **k: fwd_counter.__setitem__("n", fwd_counter["n"] + 1)
    )
    per_step_calls = 2 if args.guidance_scale != 1.0 else 1

    ctx = SamplerSideChannels()
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    def run_arm(init: torch.Tensor, spectrum: bool, fov: VelocityFoveation | None):
        fwd_counter["n"] = 0
        lat = spectrum_denoise(
            anima,
            init.clone(),
            timesteps,
            sigmas,
            embed,
            neg_embed,
            pad,
            args.guidance_scale,
            None,  # euler step
            device,
            ctx,
            window_size=args.window_size,
            flex_window=args.flex_window,
            warmup_steps=num_steps if not spectrum else args.warmup_steps,
            foveation=fov,
        )
        return lat, fwd_counter["n"] // per_step_calls

    arms = [{"name": "full", "spectrum": False, "sigma_c": None}]
    arms.append({"name": "spec", "spectrum": True, "sigma_c": None})
    for sc in args.sigma_c:
        arms.append({"name": f"spec_fov_sc{sc:g}", "spectrum": True, "sigma_c": sc})

    run_dir = make_run_dir("foveated", label=args.label)
    periph = np.ones((args.height, args.width), dtype=bool)
    periph[fovea_box[1] : fovea_box[3], fovea_box[0] : fovea_box[2]] = False
    fovea_px = ~periph

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
            fov = (
                VelocityFoveation(mask4, args.pool, arm["sigma_c"])
                if arm["sigma_c"] is not None
                else None
            )
            lat, n_fwd = run_arm(init, arm["spectrum"], fov)
            nan = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
            img = _to_pil(decode_latent(vae, lat, device))
            img.save(run_dir / f"seed{seed}_{arm['name']}.png")
            images[arm["name"]] = img
            row["arms"][arm["name"]] = {
                "actual_forwards": n_fwd,
                "nan_inf": nan,
                "latent_std": float(lat.float().std()),
                **({"fov_forced_step": fov.forced_step} if fov is not None else {}),
            }

        # Metrics vs the all-actual baseline
        base = images["full"]
        base_lap = _lap_energy(base, periph)
        errmaps = []
        for arm in arms[1:]:
            name = arm["name"]
            img, m = images[name], row["arms"][name]
            m["fovea_rmse_vs_full"] = _rmse_region(img, base, fovea_px)
            m["periph_rmse_vs_full"] = _rmse_region(img, base, periph)
            m["fovea_over_periph_err"] = m["fovea_rmse_vs_full"] / max(
                m["periph_rmse_vs_full"], 1e-9
            )
            m["image_rmse_vs_full"] = _rmse(img, base)
            m["periph_lap_ratio"] = _lap_energy(img, periph) / max(base_lap, 1e-9)
            errmaps.append((_errmap(img, base), f"|{name} - full|"))
            log.info(
                f"    {name}: fwd={m['actual_forwards']}  "
                f"fovea_rmse={m['fovea_rmse_vs_full']:.4f}  "
                f"periph_rmse={m['periph_rmse_vs_full']:.4f}  "
                f"(fov/periph {m['fovea_over_periph_err']:.2f})  "
                f"periph_sharp×{m['periph_lap_ratio']:.2f}"
            )
        per_seed.append(row)

        full_row = _grid(
            [
                (
                    _outline(images[a["name"]], fovea_box),
                    f"{a['name']} ({row['arms'][a['name']]['actual_forwards']} fwd)",
                )
                for a in arms
            ]
        )
        crop_row = _grid(
            [(images[a["name"]].crop(fovea_box), f"fovea {a['name']}") for a in arms]
        )
        _stack([full_row, crop_row]).save(run_dir / f"compare_seed{seed}.png")
        _grid(errmaps).save(run_dir / f"errmap_seed{seed}.png")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    hook.remove()

    # ── Aggregates: headroom stat + compose-smoke certifiers ──
    def _mean(name, key):
        return float(np.mean([r["arms"][name][key] for r in per_seed]))

    headroom = _mean("spec", "fovea_over_periph_err")
    spec_fovea = _mean("spec", "fovea_rmse_vs_full")
    fov_stats = {
        a["name"]: {
            "fovea_rmse_vs_full": _mean(a["name"], "fovea_rmse_vs_full"),
            "vs_spec_fovea_rmse": _mean(a["name"], "fovea_rmse_vs_full") - spec_fovea,
            "actual_forwards": _mean(a["name"], "actual_forwards"),
        }
        for a in arms[2:]
    }
    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    stds = [m["latent_std"] for r in per_seed for m in r["arms"].values()]
    if any_nan or (max(stds) / max(min(stds), 1e-9) > 3.0):
        verdict = (
            "HARD_FAIL: NaN/Inf or latent-std blow-up — the σ_c discontinuity "
            "destabilizes cached steps. KILL the composition (see proposal 0b)."
        )
    else:
        verdict = (
            f"NO_HARD_DIVERGENCE — visual call on compare_seed*/errmap_seed*.png. "
            f"HEADROOM: spec fovea/periph error ratio {headroom:.2f} "
            f"(>1 = forecast error concentrates in the fovea → spatial refresh "
            f"allocation has something to buy; ~1 or <1 = weak headroom). "
            f"COMPOSE: spec fovea RMSE {spec_fovea:.4f}; foveated arms "
            + ", ".join(
                f"{k} {v['fovea_rmse_vs_full']:.4f} ({v['vs_spec_fovea_rmse']:+.4f})"
                for k, v in fov_stats.items()
            )
            + ". KILL if foveation visibly damages the fovea beyond spec's own drift."
        )

    metrics = {
        "spectrum": {
            "window_size": args.window_size,
            "flex_window": args.flex_window,
            "warmup_steps": args.warmup_steps,
        },
        "sigma_c_sweep": args.sigma_c,
        "fovea_frac": args.fovea_frac,
        "pool_latent_px": args.pool,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "fovea_box_px": list(fovea_box),
        "headroom_fovea_over_periph_err": headroom,
        "spec_fovea_rmse": spec_fovea,
        "fov_arms": fov_stats,
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
    log.info(f"  {verdict}")
    log.info(f"  → {run_dir}  (open compare_seed*.png + errmap_seed*.png)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
