"""Phase-1b probe — REAL masked token merging (fixed-mask foveated merge).

The production mechanism behind the P0/P0b emulations: below σ_c the DiT's block
stack actually runs on a reduced token sequence — fovea tokens kept 1:1, each
2×2-token periphery group merged to one token — so this measures the real
wall-clock outcome, not an emulated quality contract.

Mechanism (whole-stack merge; bench-side, zero model edits):
  * ``prepare_embedded_sequence`` → full patch-grid tokens + rope, as always.
  * Merge once after patch embed: fovea tokens pass through, periphery 2×2
    groups are averaged → ``(B, L_red, D)`` with L_red ≈ 51% of L at
    ``fovea_frac 0.35``. Handed to the model's own ``_run_blocks`` in the
    fake-5D ``(B,1,L_red,1,D)`` native-flatten layout blocks already accept.
  * Merged rope: a 2×2 group sits at its members' mean position — for a
    symmetric group, the renormalized elementwise mean of the members'
    (cos, sin) rows IS the exact mean-position rope (per axis, the two angles
    are symmetric about the center: mean vector keeps the center angle, only
    its norm shrinks — renormalizing restores it).
  * Broadcast back to the full grid before ``final_layer`` → full-res velocity
    with group-shared periphery updates (the P0 contract), latent full-res
    end-to-end, final bicubic merged readout as in P0.

Whole-stack (vs per-block merge/broadcast) is the faithful choice here: P0's
final pooled readout already discards within-group detail, which is the only
thing per-block round-trips would preserve — at strictly more gather/scatter.

Startup invariants (run every launch, ~3 forwards):
  1. custom full-res runner ≡ ``anima()`` (bit-close),
  2. merge with an all-fovea mask ≡ full-res forward (index/rope plumbing is
     a pure permutation there → bit-exact).

Arms: ``baseline`` (full compute), ``merge_sc{0.75,0.5}``. Readouts: per-forward
CUDA-event ms (full vs merged), end-to-end loop seconds, fovea RMSE vs baseline
(compare against P0v3's emulation: 0.031–0.089 at these σ_c), periphery
Laplacian ratio. Verdict is the usual full-res eyeball on ``compare_seed*.png``.

Usage:
  uv run python -m bench.foveated.probe_token_merge
  uv run python -m bench.foveated.probe_token_merge --sigma_c 0.5 --seeds 40 41 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

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
)

log = logging.getLogger("bench.foveated.merge")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class FoveatedTokenMerge:
    """Fixed-mask foveated token merge: index maps + merged rope for one
    (grid, mask) pair, applied at the block-stack boundary.

    ``mask4`` is the P0 latent-pixel mask (1,1,h_lat,w_lat), built on the pool
    grid so every 2×2-token cell is uniformly fovea or periphery.
    """

    def __init__(self, anima, mask4: torch.Tensor, device: torch.device):
        p = anima.patch_spatial  # latent px per token edge (2)
        h_lat, w_lat = mask4.shape[-2:]
        self.H_t, self.W_t = h_lat // p, w_lat // p
        if self.H_t % 2 or self.W_t % 2:
            raise SystemExit("token grid must be even for 2x2 merge cells")
        self.H_c, self.W_c = self.H_t // 2, self.W_t // 2

        tok_mask = mask4[0, 0, ::p, ::p] > 0.5  # (H_t, W_t) — uniform per token
        cell_fovea = tok_mask[::2, ::2]  # uniform per 2x2-token cell
        tokf = tok_mask.flatten()
        cellp = (~cell_fovea).flatten()

        self.fovea_tok_idx = torch.nonzero(tokf).flatten().to(device)
        self.periph_cell_idx = torch.nonzero(cellp).flatten().to(device)
        self.n_f = int(self.fovea_tok_idx.numel())
        self.L_red = self.n_f + int(self.periph_cell_idx.numel())
        self.L_full = self.H_t * self.W_t

        # full token index → reduced index (fovea rank, or n_f + its cell's rank)
        fov_rank = torch.cumsum(tokf.long(), 0) - 1
        cell_rank = torch.cumsum(cellp.long(), 0) - 1
        ti = torch.arange(self.L_full)
        cell_of_tok = (ti // self.W_t // 2) * self.W_c + (ti % self.W_t) // 2
        self.gather_idx = torch.where(
            tokf, fov_rank, self.n_f + cell_rank[cell_of_tok]
        ).to(device)
        self._rope_cache: tuple | None = None

    def _cell_mean(self, flat_tok: torch.Tensor) -> torch.Tensor:
        """(..., L_full, D*) row-major token rows → (..., H_c*W_c, D*) cell means."""
        lead = flat_tok.shape[:-2]  # () for 2D rope rows, (B,) for token batches
        rest = flat_tok.shape[-1:]
        x = flat_tok.reshape(-1, self.H_t, self.W_t, *rest)
        x = x.reshape(-1, self.H_c, 2, self.W_c, 2, *rest).mean(dim=(2, 4))
        return x.reshape(*lead, self.H_c * self.W_c, *rest)

    def merge(self, x_BTHWD: torch.Tensor) -> torch.Tensor:
        """(B,1,H_t,W_t,D) grid → fake-5D (B,1,L_red,1,D) merged sequence."""
        B, T, H, W, D = x_BTHWD.shape
        flat = x_BTHWD.reshape(B, H * W, D)
        fov = flat[:, self.fovea_tok_idx]
        per = self._cell_mean(flat)[:, self.periph_cell_idx]
        return torch.cat([fov, per], dim=1).unsqueeze(1).unsqueeze(3)

    def unmerge(self, x_red: torch.Tensor, B: int, D: int) -> torch.Tensor:
        """fake-5D (B,1,L_red,1,D) → (B,1,H_t,W_t,D) with group-broadcast periphery."""
        flat = x_red.reshape(B, self.L_red, D)
        return flat[:, self.gather_idx].reshape(B, 1, self.H_t, self.W_t, D)

    def merge_rope(self, rope_cos_sin: tuple) -> tuple:
        """Merged (cos, sin): fovea rows pass through; periphery cell rows are the
        renormalized elementwise mean of their 4 members — exact mean-position
        rope for a symmetric 2×2 group. Cached (rope is constant across steps)."""
        if self._rope_cache is not None:
            return self._rope_cache
        cos_, sin_ = rope_cos_sin  # (L_full, 1, 1, D_head), seq on dim 0
        c32, s32 = cos_.float(), sin_.float()
        cm = self._cell_mean(c32.reshape(self.L_full, -1)).reshape(
            self.H_c * self.W_c, *cos_.shape[1:]
        )[self.periph_cell_idx]
        sm = self._cell_mean(s32.reshape(self.L_full, -1)).reshape(
            self.H_c * self.W_c, *sin_.shape[1:]
        )[self.periph_cell_idx]
        norm = torch.sqrt(cm * cm + sm * sm).clamp_min(1e-6)
        cos_red = torch.cat([c32[self.fovea_tok_idx], cm / norm], dim=0).to(cos_.dtype)
        sin_red = torch.cat([s32[self.fovea_tok_idx], sm / norm], dim=0).to(sin_.dtype)
        self._rope_cache = (cos_red, sin_red)
        return self._rope_cache


def dit_forward(anima, x5, t_scalar, context, pad, merger: FoveatedTokenMerge | None):
    """The model's own forward path with an optional merge at the block-stack
    boundary. ``merger=None`` reproduces ``anima()`` op-for-op (verified at
    startup) so per-arm timings are apples-to-apples."""
    from networks import attention_dispatch

    B = x5.shape[0]
    t = x5.new_full((B, 1), float(t_scalar))
    x, rope = anima.prepare_embedded_sequence(x5, padding_mask=pad)
    rope = (rope[0].to(x.dtype), rope[1].to(x.dtype))
    t_emb, adaln = anima.t_embedder(t)
    t_emb = anima.t_embedding_norm(t_emb)
    if anima.enable_pooled_text_modulation:
        t_emb = t_emb + anima._pooled_text_delta(
            context.max(dim=1).values, t_emb
        ).unsqueeze(1)
    attn_params = attention_dispatch.AttentionParams.create_attention_params(
        anima.attn_mode, anima.attn_softmax_scale
    )
    B_, T_, H_, W_, D_ = x.shape
    if merger is not None:
        x = anima._run_blocks(
            merger.merge(x),
            t_emb,
            context,
            attn_params,
            rope_cos_sin=merger.merge_rope(rope),
            adaln_lora_B_T_3D=adaln,
        )
        x = merger.unmerge(x, B_, D_)
    else:
        x = anima._run_blocks(
            x, t_emb, context, attn_params, rope_cos_sin=rope, adaln_lora_B_T_3D=adaln
        )
    t_emb_final = t_emb + (
        anima._mod_guidance_final_w * anima._mod_guidance_delta
    ).unsqueeze(1)
    return anima.unpatchify(anima.final_layer(x, t_emb_final, adaln_lora_B_T_3D=adaln))


@torch.no_grad()
def denoise(
    anima,
    x5_init,
    embed,
    neg_embed,
    sigmas,
    guidance,
    device,
    pad,
    merger: FoveatedTokenMerge | None,
    mask4,
    pool: int,
    sigma_c: float | None,
):
    """Euler CFG loop; below σ_c both CFG branches run the merged stack. Timing
    via CUDA events per DiT call, bucketed full vs merged."""
    x5 = x5_init
    times = {"full_ms": [], "merged_ms": []}

    def velocity(x, sigma, use_merge):
        m = merger if use_merge else None
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        v_c = dit_forward(anima, x, sigma, embed, pad, m)
        v_u = (
            dit_forward(anima, x, sigma, neg_embed, pad, m) if guidance != 1.0 else None
        )
        end.record()
        end.synchronize()
        times["merged_ms" if use_merge else "full_ms"].append(start.elapsed_time(end))
        return v_u + guidance * (v_c - v_u) if v_u is not None else v_c

    t0 = time.perf_counter()
    n = len(sigmas) - 1
    for i in range(n):
        sigma = float(sigmas[i])
        use_merge = merger is not None and sigma_c is not None and sigma <= sigma_c
        v = velocity(x5, sigma, use_merge).float()
        x5 = (x5.float() + v * (float(sigmas[i + 1]) - sigma)).to(torch.bfloat16)
    if device.type == "cuda":
        torch.cuda.synchronize()
    e2e = time.perf_counter() - t0

    if merger is not None and sigma_c is not None:
        # P0 final readout: periphery read from the merged representation.
        x4 = x5.float().squeeze(2)
        low = F.avg_pool2d(x4, pool)
        up = F.interpolate(low, size=x4.shape[-2:], mode="bicubic", align_corners=False)
        x4 = mask4 * x4 + (1.0 - mask4) * up
        x5 = x4.unsqueeze(2).to(torch.bfloat16)
    return x5, times, e2e


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
    ap.add_argument("--sigma_c", type=float, nargs="+", default=[0.75, 0.5])
    ap.add_argument("--fovea_frac", type=float, default=0.35)
    ap.add_argument(
        "--fovea_center", type=float, nargs=2, default=[0.38, 0.5], metavar=("CY", "CX")
    )
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--label", default="p1b")
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
    mask4 = build_fovea_mask(
        h_lat, w_lat, args.pool, args.fovea_frac, device, center=args.fovea_center
    )
    fovea_box = _fovea_box_px(mask4)
    merger = FoveatedTokenMerge(anima, mask4, device)
    log.info(
        f"tokens: {merger.L_full} full → {merger.L_red} merged "
        f"({merger.L_red / merger.L_full:.0%}; fovea {merger.n_f})"
    )

    # ── Startup invariants ──
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
        all_fovea = FoveatedTokenMerge(anima, torch.ones_like(mask4), device)
        merged_id = dit_forward(anima, probe_x, 0.7, embed, pad, all_fovea)
        d2 = float((mine - merged_id).abs().max())
    log.info(f"invariants: custom-vs-anima max|Δ|={d1:.2e}, all-fovea-merge={d2:.2e}")
    if d1 > 5e-2 or d2 > 1e-6:
        raise SystemExit("startup invariant failed — merge plumbing is wrong")

    arms = [{"name": "baseline", "sigma_c": None}]
    arms += [{"name": f"merge_sc{sc:g}", "sigma_c": sc} for sc in args.sigma_c]

    run_dir = make_run_dir("foveated", label=args.label)
    periph = np.ones((args.height, args.width), dtype=bool)
    periph[fovea_box[1] : fovea_box[3], fovea_box[0] : fovea_box[2]] = False

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
        for arm in arms:
            log.info(f"  {arm['name']} ...")
            lat, times, e2e = denoise(
                anima,
                init.clone(),
                embed,
                neg_embed,
                sigmas,
                args.guidance_scale,
                device,
                pad,
                merger if arm["sigma_c"] is not None else None,
                mask4,
                args.pool,
                arm["sigma_c"],
            )
            img = _to_pil(decode_latent(vae, lat, device))
            img.save(run_dir / f"seed{seed}_{arm['name']}.png")
            images[arm["name"]] = img
            row["arms"][arm["name"]] = {
                "e2e_s": e2e,
                "full_fwd_ms": float(np.mean(times["full_ms"])),
                "merged_fwd_ms": float(np.mean(times["merged_ms"]))
                if times["merged_ms"]
                else None,
                "n_merged_steps": len(times["merged_ms"]),
                "nan_inf": bool(torch.isnan(lat).any() or torch.isinf(lat).any()),
                "latent_std": float(lat.float().std()),
            }

        base = images["baseline"]
        base_e2e = row["arms"]["baseline"]["e2e_s"]
        base_lap = _lap_energy(base, periph)
        for arm in arms[1:]:
            img, m = images[arm["name"]], row["arms"][arm["name"]]
            m["fovea_rmse_vs_base"] = _rmse(img, base, fovea_box)
            m["image_rmse_vs_base"] = _rmse(img, base)
            m["periph_lap_ratio"] = _lap_energy(img, periph) / max(base_lap, 1e-9)
            m["e2e_speedup"] = base_e2e / m["e2e_s"]
            m["fwd_speedup"] = (
                m["full_fwd_ms"] / m["merged_fwd_ms"] if m["merged_fwd_ms"] else None
            )
            log.info(
                f"    {arm['name']}: fovea_rmse={m['fovea_rmse_vs_base']:.4f}  "
                f"periph_sharp×{m['periph_lap_ratio']:.2f}  "
                f"fwd {m['full_fwd_ms']:.0f}→{m['merged_fwd_ms']:.0f}ms "
                f"(×{m['fwd_speedup']:.2f} on {m['n_merged_steps']} steps)  "
                f"e2e ×{m['e2e_speedup']:.2f}"
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

    # ── Aggregate ──
    def _mean(name, key):
        vals = [r["arms"][name][key] for r in per_seed if r["arms"][name][key]]
        return float(np.mean(vals)) if vals else None

    any_nan = any(m["nan_inf"] for r in per_seed for m in r["arms"].values())
    agg = {
        a["name"]: {
            k: _mean(a["name"], k)
            for k in (
                "fovea_rmse_vs_base",
                "periph_lap_ratio",
                "fwd_speedup",
                "e2e_speedup",
            )
        }
        for a in arms[1:]
    }
    verdict = (
        "HARD_FAIL: NaN/Inf in merged arms."
        if any_nan
        else (
            "NO_HARD_DIVERGENCE — visual call on compare_seed*.png. Real-merge vs "
            "P0v3 emulation fovea RMSE (emul: 0.089@sc0.75 / 0.031@sc0.5): "
            + ", ".join(
                f"{k} {v['fovea_rmse_vs_base']:.4f} (fwd ×{v['fwd_speedup']:.2f}, "
                f"e2e ×{v['e2e_speedup']:.2f})"
                for k, v in agg.items()
            )
            + ". Gate: quality ≈ emulation AND wall-clock saving worth shipping."
        )
    )

    metrics = {
        "sigma_c_sweep": args.sigma_c,
        "fovea_frac": args.fovea_frac,
        "pool_latent_px": args.pool,
        "tokens_full": merger.L_full,
        "tokens_merged": merger.L_red,
        "token_ratio": merger.L_red / merger.L_full,
        "infer_steps": args.infer_steps,
        "flow_shift": args.flow_shift,
        "guidance_scale": args.guidance_scale,
        "resolution_hw": [args.height, args.width],
        "fovea_box_px": list(fovea_box),
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
