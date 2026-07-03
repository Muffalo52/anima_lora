"""Phase-3 probe — partial-recompute pre-gate + mergefresh, on the composed stack.

Phase 3 of ``bench/foveated/plan.md``: the foveated-Spectrum endgame. All arms
run the bench-local aggressive Spectrum loop (window 3 / flex 3 / warmup 4, the
0b/1a/2t operating point) with the Phase-2b ``combo`` mask (cfgdelta + x0var,
built endogenously at the σ_c crossing). Legs covered here:

* **3-pre** (``pr_all`` / ``pr_catchup2``) — the anchoring pre-gate 1a/P2f owed.
  Partial recompute never gives the periphery an actual update below σ_c; every
  blend arm so far had fits anchored to FULL actuals. Emulated at full compute
  and at partial-recompute's REAL cadence (every step below σ_c is an actual —
  that's what the ~⅓-cost refresh budget buys): emit = fovea actual + periphery
  from a forecaster FROZEN at the crossing; live fits get fovea-only anchors.
  ``pr_catchup2`` re-anchors fully every 2nd below-step (the 1c rescue arm).
  ``pr_all``'s subject RMSE doubles as the quality CEILING estimate for a real
  Phase-3b implementation (fovea sees real features every step below σ_c).

* **3a** (``mergefresh`` / ``mergeall``) — the zero-new-surgery competitor that
  sets 3b's bar. ``mergefresh``: spec's schedule, but post-crossing actuals run
  the 1b merged stack (×2.15/fwd — same refreshes, cheaper). ``mergeall``:
  Spectrum above σ_c, EVERY step below is a merged actual (no forecasting below
  the crossing at all — the Spectrum→deferred-merge handoff; ``mergeall75``
  moves the handoff to the standalone merge knee σ_c=0.75, costs more FFE).

Cost is reported in **full-forward equivalents (FFE)**: merged forwards at
their measured ms ratio, emulated fovea-only refreshes at the proposal's ~⅓
estimate (``PR_COST``). Gates:
  * 3-pre: ``pr_all`` periphery (own-mask complement) degradation vs ``spec``
    within the 1a neutrality standard (≲0.003) or visually clean → 3b's
    foundation holds; else ``pr_catchup2`` must rescue it (re-anchor tax).
  * 3a: ``mergefresh`` subject RMSE ≈ ``spec`` at lower FFE; ``mergeall``
    subject RMSE < ``spec`` at ≈ matched FFE.
  * 3b is only worth building if ``pr_all``'s subject ceiling clearly beats
    ``mergeall``'s realized number.

Usage:
  uv run python -m bench.foveated.probe_phase3
  uv run python -m bench.foveated.probe_phase3 --prompts default --seeds 40
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT, add_model_args
from bench._common import make_run_dir, write_result
from bench.foveated.probe_mask_shapes import (
    _cells_px,
    _cells_to_mask4,
    _rect_cells,
    _ring_px,
    _rmse_sel,
)
from bench.foveated.probe_mask_sources import (
    _draw_boxes,
    _draw_mask_and_boxes,
    _subject_strip,
    score_to_cells,
)
from bench.foveated.probe_spectrum_compose import _errmap
from bench.foveated.probe_token_merge import FoveatedTokenMerge, dit_forward
from bench.foveated.probe_velocity_foveation import _grid, _lap_energy, _stack, _to_pil

log = logging.getLogger("bench.foveated.phase3")
logging.basicConfig(level=logging.INFO, format="%(message)s")

PR_COST = 1.0 / 3.0  # fovea-query refresh ≈ ⅓ forward (proposal §Phase 3 arithmetic)


@torch.no_grad()
def phase3_arm(
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
    mode: str,  # "full" | "spec" | "pr" | "mergefresh" | "mergeall"
    sigma_c: float | None = None,
    catchup_every: int = 0,  # pr only: full re-anchor every k-th below-step
    fovea_frac: float = 0.35,
    pool: int = 4,
    warmup: int = 4,
    window_size: float = 3.0,
    flex_window: float = 3.0,
):
    """One denoise run. Mirrors ``spectrum_arm``'s window/forecast logic; below
    σ_c the mode decides how actuals are computed and what anchors the fits.
    The combo mask (2b winner) is built endogenously at the crossing from
    signals accumulated over every emitted velocity (actuals AND forecasts —
    unlike 2b's all-actual loop; the forecast steps just smooth the signal)."""
    from library.inference import sampling as inference_utils
    from networks.spectrum import _spectrum_fast_forward
    from networks.spectrum_forecast import SpectrumPredictor

    m_basis, lam, w_blend = 3, 0.1, 0.3  # spectrum_denoise defaults
    num_steps = len(timesteps)
    stop_at = num_steps - 3
    do_cfg = guidance != 1.0
    latents = init.clone()
    h_lat, w_lat = latents.shape[-2:]
    masked_mode = mode in ("pr", "mergefresh", "mergeall")
    merge_mode = mode in ("mergefresh", "mergeall")
    frozen_mode = mode == "pr"

    captured = {}
    hook = anima.final_layer.register_forward_pre_hook(
        lambda module, args: captured.__setitem__("feat", args[0].detach().clone())
    )

    cond_fc = uncond_fc = None
    frozen: dict = {}
    curr_ws, consec = window_size, 0
    refresh_steps: list[int] = []
    times = {"full_ms": [], "merged_ms": []}
    n_fovea_only = 0
    n_post = 0  # below-σ_c actuals after the crossing handoff

    cfg_acc = torch.zeros(h_lat, w_lat, device=device)
    x0_lap: torch.Tensor | None = None
    cells = mask4 = mask_tok5 = merger = None
    frac_actual = None

    def cell_mean(px_map: torch.Tensor) -> np.ndarray:
        return F.avg_pool2d(px_map[None, None], pool)[0, 0].float().cpu().numpy()

    t0 = time.perf_counter()
    try:
        for i, t in enumerate(timesteps):
            sigma = float(sigmas[i])
            below = masked_mode and sigma_c is not None and sigma <= sigma_c
            force_actual = False
            if below and cells is None:
                # ── crossing: build the combo mask (2b pipeline) ──
                s_cfg = cell_mean(cfg_acc)
                s_x0 = cell_mean(x0_lap) if x0_lap is not None else np.zeros_like(s_cfg)
                score = s_cfg / max(s_cfg.sum(), 1e-9) + s_x0 / max(s_x0.sum(), 1e-9)
                cells, _ = score_to_cells(score, fovea_frac)
                frac_actual = float(cells.float().mean())
                mask4 = _cells_to_mask4(cells, pool, device)
                p = anima.patch_spatial
                mask_tok5 = (
                    (mask4[0, 0, ::p, ::p] > 0.5)
                    .float()
                    .view(1, 1, h_lat // p, w_lat // p, 1)
                )
                if merge_mode:
                    merger = FoveatedTokenMerge(anima, mask4, device)
                force_actual = True  # forced actual at the crossing (0b machinery)

            if mode == "full" or i < warmup or i >= stop_at or force_actual:
                actual = True
            elif mode in ("pr", "mergeall") and below:
                actual = True  # partial-recompute cadence: every below-step
            else:
                actual = (consec + 1) % max(1, math.floor(curr_ws)) == 0

            # pr bookkeeping: the crossing handoff is always a FULL actual;
            # afterwards catchup_every=k re-anchors fully on every k-th step.
            reanchor = False
            if below and frozen_mode and actual:
                if force_actual:
                    reanchor = True
                elif catchup_every > 0 and n_post % catchup_every == catchup_every - 1:
                    reanchor = True

            t_exp = t.expand(latents.shape[0])
            # mergefresh keeps 0b's stability machinery: the crossing handoff is
            # a FULL actual (its below-σ_c cached steps forecast from it);
            # mergeall has no forecasts below, so its crossing runs merged (1b).
            use_merge = (
                actual
                and below
                and merge_mode
                and not (mode == "mergefresh" and force_actual)
            )
            if actual:
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                if use_merge:
                    v_c = dit_forward(anima, latents, float(t), embed, pad, merger)
                else:
                    v_c = anima(latents, t_exp, embed, padding_mask=pad)
                feat_c = captured["feat"]
                v_u = feat_u = None
                if do_cfg:
                    if use_merge:
                        v_u = dit_forward(
                            anima, latents, float(t), neg_embed, pad, merger
                        )
                    else:
                        v_u = anima(latents, t_exp, neg_embed, padding_mask=pad)
                    feat_u = captured["feat"]
                end.record()
                end.synchronize()
                times["merged_ms" if use_merge else "full_ms"].append(
                    start.elapsed_time(end)
                )

                if cond_fc is None:
                    cond_fc = SpectrumPredictor(
                        m_basis, lam, w_blend, device, feat_c.shape[1:], num_steps
                    )
                    if do_cfg:
                        uncond_fc = SpectrumPredictor(
                            m_basis, lam, w_blend, device, feat_u.shape[1:], num_steps
                        )

                fovea_only = below and frozen_mode and not reanchor
                if fovea_only:
                    # partial-recompute emulation: the periphery of both the
                    # emitted feature and the fit anchor comes from the frozen
                    # crossing-time forecaster; only the fovea sees the actual.
                    bl_c = (
                        mask_tok5 * feat_c
                        + (1.0 - mask_tok5) * frozen["cond"].predict(float(i))
                    ).to(feat_c.dtype)
                    v_c = _spectrum_fast_forward(anima, t_exp, bl_c)
                    cond_fc.update(float(i), bl_c)
                    if do_cfg:
                        bl_u = (
                            mask_tok5 * feat_u
                            + (1.0 - mask_tok5) * frozen["uncond"].predict(float(i))
                        ).to(feat_u.dtype)
                        v_u = _spectrum_fast_forward(anima, t_exp, bl_u)
                        uncond_fc.update(float(i), bl_u)
                    n_fovea_only += 1
                else:
                    cond_fc.update(float(i), feat_c)
                    if do_cfg:
                        uncond_fc.update(float(i), feat_u)
                    if reanchor:
                        frozen["cond"] = copy.deepcopy(cond_fc)
                        if do_cfg:
                            frozen["uncond"] = copy.deepcopy(uncond_fc)

                if below and not force_actual:
                    n_post += 1
                if i >= warmup and mode != "full":
                    curr_ws = round(curr_ws + flex_window, 3)
                consec = 0
                refresh_steps.append(i)
            else:
                pred_c = cond_fc.predict(float(i))
                pred_u = uncond_fc.predict(float(i)) if do_cfg else None
                if below and frozen_mode:
                    pred_c = (
                        mask_tok5 * pred_c
                        + (1.0 - mask_tok5) * frozen["cond"].predict(float(i))
                    ).to(pred_c.dtype)
                    if do_cfg:
                        pred_u = (
                            mask_tok5 * pred_u
                            + (1.0 - mask_tok5) * frozen["uncond"].predict(float(i))
                        ).to(pred_u.dtype)
                v_c = _spectrum_fast_forward(anima, t_exp, pred_c)
                v_u = _spectrum_fast_forward(anima, t_exp, pred_u) if do_cfg else None
                consec += 1

            noise_pred = v_u + guidance * (v_c - v_u) if do_cfg else v_c

            if masked_mode and cells is None:
                # accumulate the endogenous mask signals (pre-crossing)
                cfg_acc += (v_c.float() - v_u.float()).abs().mean(dim=(0, 1, 2))
                x0 = (latents.float() - sigma * noise_pred.float()).squeeze(2)[0]
                lap = (
                    -4 * x0[:, 1:-1, 1:-1]
                    + x0[:, :-2, 1:-1]
                    + x0[:, 2:, 1:-1]
                    + x0[:, 1:-1, :-2]
                    + x0[:, 1:-1, 2:]
                )
                e = torch.zeros(h_lat, w_lat, device=device)
                e[1:-1, 1:-1] = (lap**2).mean(dim=0)
                x0_lap = e

            latents = inference_utils.step(latents, noise_pred, sigmas, i).to(
                latents.dtype
            )
    finally:
        hook.remove()

    if device.type == "cuda":
        torch.cuda.synchronize()
    e2e = time.perf_counter() - t0

    if merge_mode and mask4 is not None:
        # P0 final readout: periphery read from the merged representation.
        x4 = latents.float().squeeze(2)
        low = F.avg_pool2d(x4, pool)
        up = F.interpolate(low, size=x4.shape[-2:], mode="bicubic", align_corners=False)
        latents = (mask4 * x4 + (1.0 - mask4) * up).unsqueeze(2).to(torch.bfloat16)

    n_full = len(times["full_ms"]) - (n_fovea_only if frozen_mode else 0)
    full_ms = float(np.mean(times["full_ms"])) if times["full_ms"] else None
    merged_ms = float(np.mean(times["merged_ms"])) if times["merged_ms"] else None
    ratio = merged_ms / full_ms if merged_ms and full_ms else None
    ffe = (
        n_full
        + n_fovea_only * PR_COST
        + (len(times["merged_ms"]) * ratio if ratio else 0.0)
    )
    return latents, {
        "actual_forwards": len(refresh_steps),
        "refresh_steps": refresh_steps,
        "n_full": n_full,
        "n_fovea_only": n_fovea_only,
        "n_merged": len(times["merged_ms"]),
        "full_fwd_ms": full_ms,
        "merged_fwd_ms": merged_ms,
        "ffe": float(ffe),
        "ffe_emulated": frozen_mode,  # pr arms: fovea-only cost is the ⅓ estimate
        "e2e_s": e2e,
        "fovea_frac_actual": frac_actual,
        "cells": cells,
    }


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
    # Spectrum knobs — the 0b/1a aggressive point.
    ap.add_argument("--window_size", type=float, default=3.0)
    ap.add_argument("--flex_window", type=float, default=3.0)
    ap.add_argument("--warmup_steps", type=int, default=4)
    # Foveation knobs — σ_c=0.5 is the composed knee (0b); mergeall75 escalates.
    ap.add_argument("--sigma_c", type=float, default=0.5)
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
    ap.add_argument("--label", default="p3")
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

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps = timesteps.to(device, torch.bfloat16)
    sigmas = sigmas.to(device)
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    run_dir = make_run_dir("foveated", label=args.label)

    ARMS = [
        ("full", dict(mode="full")),
        ("spec", dict(mode="spec")),
        ("pr_all", dict(mode="pr", sigma_c=args.sigma_c)),
        ("pr_catchup2", dict(mode="pr", sigma_c=args.sigma_c, catchup_every=2)),
        ("mergefresh", dict(mode="mergefresh", sigma_c=args.sigma_c)),
        ("mergeall", dict(mode="mergeall", sigma_c=args.sigma_c)),
        ("mergeall75", dict(mode="mergeall", sigma_c=0.75)),
    ]
    common = dict(
        fovea_frac=args.fovea_frac,
        pool=args.pool,
        warmup=args.warmup_steps,
        window_size=args.window_size,
        flex_window=args.flex_window,
    )

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

            row = {"prompt": pname, "seed": seed, "arms": {}}
            images: dict[str, Image.Image] = {}
            cells_by: dict[str, torch.Tensor | None] = {}
            for name, kw in ARMS:
                log.info(f"  {name} ...")
                lat, info = phase3_arm(
                    anima, init, embeds[pname], neg_embed, timesteps, sigmas,
                    args.guidance_scale, pad, device, **common, **kw,
                )  # fmt: skip
                img = _to_pil(decode_latent(vae, lat, device))
                img.save(run_dir / f"{pname}_seed{seed}_{name}.png")
                images[name] = img
                cells_by[name] = info.pop("cells")
                info["nan_inf"] = bool(torch.isnan(lat).any() or torch.isinf(lat).any())
                info["latent_std"] = float(lat.float().std())
                row["arms"][name] = info

            base = images["full"]
            errmaps = []
            for name, _ in ARMS[1:]:
                img, m = images[name], row["arms"][name]
                cells = cells_by[name]
                if cells is None and name == "spec":
                    cells = cells_by["pr_all"]  # sc0.5 arms share the trajectory
                m["subject_rmse"] = _rmse_sel(img, base, subject_sel)
                m["image_rmse"] = _rmse_sel(img, base, np.ones_like(subject_sel))
                if cells is not None:
                    own_px = _cells_px(cells, px_per_cell)
                    ring = _ring_px(cells, px_per_cell)
                    m["own_mask_rmse"] = _rmse_sel(img, base, own_px)
                    m["comp_rmse"] = _rmse_sel(img, base, ~own_px)
                    m["subject_cover"] = (
                        float(
                            (_cells_px(cells_by[name], px_per_cell) & subject_sel).sum()
                            / max(subject_sel.sum(), 1)
                        )
                        if cells_by[name] is not None
                        else None
                    )
                    m["ring_lap_ratio"] = _lap_energy(img, ring) / max(
                        _lap_energy(base, ring), 1e-9
                    )
                errmaps.append((_errmap(img, base), f"|{name} - full|"))
                log.info(
                    f"    {name}: subj={m['subject_rmse']:.4f}  "
                    f"own={m.get('own_mask_rmse', float('nan')):.4f}  "
                    f"comp={m.get('comp_rmse', float('nan')):.4f}  "
                    f"ffe={m['ffe']:.1f}{'~' if m['ffe_emulated'] else ''}  "
                    f"e2e={m['e2e_s']:.1f}s  refresh@{m['refresh_steps']}"
                )
            results.append(row)

            full_row = _grid(
                [(_draw_boxes(base, boxes), "full")]
                + [
                    (
                        _draw_mask_and_boxes(images[n], cells_by[n], px_per_cell, boxes)
                        if cells_by[n] is not None
                        else _draw_boxes(images[n], boxes),
                        f"{n} (ffe {row['arms'][n]['ffe']:.1f}"
                        f"{'~' if row['arms'][n]['ffe_emulated'] else ''})",
                    )
                    for n, _ in ARMS[1:]
                ]
            )
            strip_row = _grid(
                [(_subject_strip(base, boxes), "subjects full")]
                + [
                    (_subject_strip(images[n], boxes), f"subjects {n}")
                    for n, _ in ARMS[1:]
                ]
            )
            _stack([full_row, strip_row]).save(
                run_dir / f"compare_{pname}_seed{seed}.png"
            )
            _grid(errmaps).save(run_dir / f"errmap_{pname}_seed{seed}.png")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Aggregate: per prompt × arm ──
    arm_names = [n for n, _ in ARMS[1:]]
    keys = ("subject_rmse", "own_mask_rmse", "comp_rmse", "ffe", "e2e_s")
    agg = {}
    for pname in args.prompts:
        rows = [r for r in results if r["prompt"] == pname]
        agg[pname] = {
            n: {
                k: float(np.mean([r["arms"][n][k] for r in rows]))
                for k in keys
                if all(k in r["arms"][n] for r in rows)
            }
            for n in arm_names
        }
    overall = {
        n: {
            k: float(np.mean([r["arms"][n][k] for r in results]))
            for k in keys
            if all(k in r["arms"][n] for r in results)
        }
        for n in arm_names
    }
    any_nan = any(m["nan_inf"] for r in results for m in r["arms"].values())

    if any_nan:
        verdict = "HARD_FAIL: NaN/Inf."
    else:
        s = overall
        verdict = (
            "Visual call on compare_*/errmap_*.png. "
            f"[3-pre] pr_all comp {s['pr_all']['comp_rmse']:.4f} vs spec "
            f"{s['spec']['comp_rmse']:.4f} (Δ{s['pr_all']['comp_rmse'] - s['spec']['comp_rmse']:+.4f}; "
            f"neutrality ≲0.003 → 3b foundation holds), catchup2 Δ"
            f"{s['pr_catchup2']['comp_rmse'] - s['spec']['comp_rmse']:+.4f}. "
            f"[3a] subject RMSE @ FFE — spec {s['spec']['subject_rmse']:.4f} @ "
            f"{s['spec']['ffe']:.1f}; mergefresh {s['mergefresh']['subject_rmse']:.4f} @ "
            f"{s['mergefresh']['ffe']:.1f} (gate: ≈spec, cheaper); mergeall "
            f"{s['mergeall']['subject_rmse']:.4f} @ {s['mergeall']['ffe']:.1f} "
            f"(gate: <spec at ≈matched); mergeall75 {s['mergeall75']['subject_rmse']:.4f} @ "
            f"{s['mergeall75']['ffe']:.1f}. "
            f"[3b ceiling] pr_all subject {s['pr_all']['subject_rmse']:.4f} @ "
            f"~{s['pr_all']['ffe']:.1f} emulated — build 3b only if this clearly "
            "beats mergeall's realized number."
        )

    metrics = {
        "spectrum": {
            "window_size": args.window_size,
            "flex_window": args.flex_window,
            "warmup_steps": args.warmup_steps,
        },
        "sigma_c": args.sigma_c,
        "fovea_frac": args.fovea_frac,
        "pool_latent_px": args.pool,
        "pr_cost_assumed": PR_COST,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "prompts": {p: prompt_spec[p]["boxes"] for p in args.prompts},
        "aggregate_by_prompt": agg,
        "aggregate_overall": overall,
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
