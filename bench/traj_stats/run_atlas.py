#!/usr/bin/env python
"""traj_stats Phase 1 — the anime-domain trajectory atlas.

Turns single-render traces into domain statistics and answers the three
Phase 1 questions of ``_archive/proposals/traj_latent_stats.md``:

  (a) how front-loaded is token commitment in this domain (commit-CDF, E(σ)),
  (b) which channels carry it (cbits(c, σ)),
  (c) does generation match inversion — do corpus statistics transfer to the
      generation process at all.

Three arms, composable via ``--arms``:

* **gen** — seed × prompt grid with ``--traj_stats`` on, in-process via
  ``build_inference_bundle`` (one model load for the whole grid). Prompts are
  drawn from the tag_dropout prompt-set families (detailed / sparse /
  no_trigger) for spread.
* **inv** — DirectEdit inversion replaying *real* cached-corpus images as
  trajectories: cached VAE latents (``*_anima.npz``) + cached TE embeds
  (``*_anima_te.safetensors`` v0) feed ``directedit.invert`` with the
  recorder threaded through (no VAE, no text encoder forward needed).
* **static** — zero-cost corpus statistics straight off the cached latents
  (same quantizer, same normalization, no trajectory): per-channel code
  entropy + within-image token redundancy. The column redundancy intuitions
  were always based on, shown next to the trajectory-resolved ones.

All trajectory traces are canonicalized to σ-descending knot order before
aggregation (inversion records σ-ascending), and commit is recomputed from
the stored per-step codes in that shared order so the commit-CDF means the
same thing in both arms. E(σ)'s τ is calibrated per trace from the min-σ
activity floor and swept across quantile bands (0.90 / 0.95 / 0.99) as the
proposal requires.

Usage::

    uv run python bench/traj_stats/run_atlas.py --label phase1
    uv run python bench/traj_stats/run_atlas.py --smoke --label smoke
    uv run python bench/traj_stats/run_atlas.py --arms static --label static-only
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER = "models/text_encoders/qwen_3_06b_base.safetensors"

DEFAULT_PROMPTS_JSON = "bench/tag_dropout/results/prompt_sets_mignon.json"
CACHE_ROOT = "post_image_dataset/lora"

# 1024-tier free-fit token band (frozen (4032, 4200) plus slack for the two
# families) — the inversion arm stays inside one tier so σ-knot aggregation
# never mixes resolutions.
INV_TOKEN_BAND = (3800, 4400)

TAU_QUANTILES = (0.90, 0.95, 0.99)
SEED_POOL = [42, 1337, 7, 2024, 555, 31337, 9001, 123]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--arms",
        default="gen,inv,static",
        help="Comma list of arms to run: gen,inv,static.",
    )
    p.add_argument("--prompts_json", default=DEFAULT_PROMPTS_JSON)
    p.add_argument("--num_seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--sampler", default="er_sde")
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--k", type=int, default=4, help="quantization bits/channel")
    p.add_argument(
        "--num_inv",
        type=int,
        default=8,
        help="real corpus images for the inversion arm",
    )
    p.add_argument(
        "--static_sample",
        type=int,
        default=512,
        help="corpus images sampled for the static column",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny grid: 512², 8 steps, 2 prompts × 1 seed, 2 inv",
    )
    p.add_argument("--label", default=None)
    a = p.parse_args()
    if a.smoke:
        a.size, a.steps, a.num_seeds, a.num_inv, a.static_sample = (
            [512, 512],
            8,
            1,
            2,
            32,
        )
    return a


# --------------------------------------------------------------------------
# prompt / corpus selection
# --------------------------------------------------------------------------


def pick_prompts(prompts_json: str, smoke: bool) -> list[dict]:
    """detailed×4 + sparse×2 + no_trigger×2 from a tag_dropout prompt-set."""
    with open(prompts_json) as f:
        ps = json.load(f)["prompts"]
    plan = (
        [("detailed", 1), ("sparse", 1)]
        if smoke
        else [("detailed", 4), ("sparse", 2), ("no_trigger", 2)]
    )
    out = []
    for fam, n in plan:
        for entry in ps[fam][:n]:
            out.append({"family": fam, "id": entry["id"], "text": entry["text"]})
    return out


def pick_inversion_images(num: int) -> list[dict]:
    """Deterministic spread: ≤1 image per artist, 1024-tier band, TE cached."""
    root = REPO_ROOT / CACHE_ROOT
    by_artist: dict[str, list[Path]] = {}
    for p in sorted(root.glob("*/*_anima.npz")):
        te = p.with_name(p.name.replace("_anima.npz", "_anima_te.safetensors"))
        # stem_WxH_anima.npz -> te sidecar drops the _WxH size token
        stem = p.name[: -len("_anima.npz")]
        if "_" in stem:
            base, size_tok = stem.rsplit("_", 1)
            if "x" in size_tok:
                te = p.with_name(base + "_anima_te.safetensors")
                try:
                    w_px, h_px = (int(v) for v in size_tok.split("x"))
                except ValueError:
                    continue
                # filename token is pixel WxH; DiT tokens = px / (8·8·4)
                if not (INV_TOKEN_BAND[0] <= (w_px * h_px) // 256 <= INV_TOKEN_BAND[1]):
                    continue
        if te.exists():
            by_artist.setdefault(p.parent.name, []).append(p)
    rng = random.Random(42)
    artists = sorted(by_artist)
    rng.shuffle(artists)
    picks = []
    for artist in artists[:num]:
        picks.append({"artist": artist, "npz": str(rng.choice(by_artist[artist]))})
    if len(picks) < num:
        log.warning("inversion arm: only %d/%d eligible artists", len(picks), num)
    return picks


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------


def run_gen_arm(bundle, base_args, run_dir: Path, prompts, seeds, k) -> list[dict]:
    traces_dir = run_dir / "traces_gen"
    manifest = []
    seen: set[str] = set()
    total = len(prompts) * len(seeds)
    for pi, prompt in enumerate(prompts):
        for seed in seeds:
            a = copy.copy(base_args)
            a.prompt = prompt["text"]
            a.seed = seed
            a.traj_stats = True
            a.traj_stats_dir = str(traces_dir)
            a.traj_stats_k = k
            t0 = time.perf_counter()
            bundle.generate(a)
            dt = time.perf_counter() - t0
            new = sorted(set(str(p) for p in traces_dir.glob("*.npz")) - seen)
            seen.update(new)
            if len(new) != 1:
                raise RuntimeError(f"expected 1 new trace, got {new}")
            manifest.append(
                {
                    "trace": new[0],
                    "arm": "gen",
                    "seed": seed,
                    **{key: prompt[key] for key in ("family", "id")},
                }
            )
            log.info(
                "[gen %d/%d] %s seed=%d  %.1fs",
                len(manifest),
                total,
                prompt["id"],
                seed,
                dt,
            )
    return manifest


def run_inv_arm(bundle, base_args, run_dir: Path, picks, steps, k) -> list[dict]:
    from safetensors.torch import load_file

    from library.inference import get_timesteps_sigmas
    from library.inference.editing.directedit import invert
    from library.inference.traj_stats import TrajStatsRecorder

    device = bundle.device
    traces_dir = run_dir / "traces_inv"
    _, sigmas = get_timesteps_sigmas(steps, base_args.flow_shift, device)
    manifest = []
    for n, pick in enumerate(picks, start=1):
        npz = Path(pick["npz"])
        d = np.load(npz)
        lat_key = next(key for key in d.files if key.startswith("latents_"))
        z = torch.from_numpy(d[lat_key])  # (16, H, W) fp32
        z_clean = z.unsqueeze(0).unsqueeze(2).to(device, torch.bfloat16)
        te_path = npz.with_name(
            npz.name[: -len("_anima.npz")].rsplit("_", 1)[0] + "_anima_te.safetensors"
        )
        embed = load_file(str(te_path))["crossattn_emb_v0"]
        embed = embed.unsqueeze(0).to(device, torch.bfloat16)  # (1, 512, 1024)

        rec = TrajStatsRecorder(
            str(traces_dir),
            k=k,
            meta={
                "arm": "inv",
                "artist": pick["artist"],
                "stem": npz.stem,
                "infer_steps": steps,
                "guidance_scale": 1.0,
                "latent_shape": list(z.shape),
            },
        )
        t0 = time.perf_counter()
        invert(
            bundle.model,
            z_clean,
            embed,
            None,
            sigmas,
            guidance_scale=1.0,
            traj_recorder=rec,
        )
        path = rec.flush()
        manifest.append({"trace": path, "arm": "inv", **pick})
        log.info(
            "[inv %d/%d] %s %s  %.1fs",
            n,
            len(picks),
            pick["artist"],
            tuple(z.shape),
            time.perf_counter() - t0,
        )
    return manifest


def run_static_arm(num: int, k: int) -> dict:
    from library.inference.traj_stats import QWEN_LATENTS_MEAN, QWEN_LATENTS_STD

    mean = np.asarray(QWEN_LATENTS_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(QWEN_LATENTS_STD, dtype=np.float32)[:, None, None]
    levels = 1 << k
    paths = sorted((REPO_ROOT / CACHE_ROOT).glob("*/*_anima.npz"))
    rng = random.Random(42)
    rng.shuffle(paths)
    paths = paths[:num]

    cbits_all, modal_all, uniq_all = [], [], []
    for p in paths:
        d = np.load(p)
        lat_key = next(key for key in d.files if key.startswith("latents_"))
        lat = d[lat_key].astype(np.float32)  # (16, H, W)
        c, h, w = lat.shape
        h -= h % 2
        w -= w % 2
        xn = (lat[:, :h, :w] - mean) / std
        tok = xn.reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4))
        codes = np.clip(
            np.floor((np.clip(tok, -4.0, 4.0) + 4.0) / 8.0 * levels), 0, levels - 1
        ).astype(np.uint8)
        # per-channel code entropy (bits) — the static twin of cbits(c, ·)
        ent = np.empty(c, dtype=np.float64)
        for ci in range(c):
            counts = np.bincount(codes[ci].ravel(), minlength=levels)
            prob = counts / counts.sum()
            nz = prob[prob > 0]
            ent[ci] = float(-(nz * np.log2(nz)).sum())
        cbits_all.append(ent)
        # within-image token redundancy on the joint 16-channel code
        flat = codes.reshape(c, -1).T
        _, counts = np.unique(flat, axis=0, return_counts=True)
        modal_all.append(float(counts.max() / flat.shape[0]))
        uniq_all.append(float(len(counts) / flat.shape[0]))

    cb = np.stack(cbits_all)  # (N, 16)
    return {
        "n_images": len(paths),
        "cbits_mean": cb.mean(axis=0).round(4).tolist(),
        "cbits_p25": np.percentile(cb, 25, axis=0).round(4).tolist(),
        "cbits_p75": np.percentile(cb, 75, axis=0).round(4).tolist(),
        "modal_token_share_mean": round(float(np.mean(modal_all)), 4),
        "modal_token_share_p75": round(float(np.percentile(modal_all, 75)), 4),
        "unique_code_frac_mean": round(float(np.mean(uniq_all)), 4),
    }


# --------------------------------------------------------------------------
# trace analysis (σ-descending canonical order for both arms)
# --------------------------------------------------------------------------


def trace_profile(npz_path: str) -> dict:
    d = np.load(npz_path)
    sig = d["sigmas"]  # natural record order
    order = np.argsort(-sig, kind="stable")  # σ-descending
    sig_d = sig[order]
    act = d["activity"].astype(np.float32)[order]  # (S, B, Ht, Wt)
    hf = d["hf"].astype(np.float32)[order]
    guide = d["guide"].astype(np.float32)[order]
    cbits = d["cbits"].astype(np.float32)[order]  # (S, C)
    codes = d["codes"][order]  # (S, B, C, Ht, Wt)

    s = codes.shape[0]
    # commit recomputed from codes in the shared σ-descending order: last knot
    # index at which any channel's code changed (0 = never changed after σ_max)
    changed = (codes[1:] != codes[:-1]).any(axis=2)  # (S-1, B, Ht, Wt)
    idx = np.arange(1, s, dtype=np.int32)[:, None, None, None]
    commit = (
        (changed * idx).max(axis=0)
        if s > 1
        else np.zeros(codes.shape[1:2] + codes.shape[3:], dtype=np.int32)
    )
    n_tok = commit.size
    commit_cdf = [float((commit <= i).sum() / n_tok) for i in range(s)]

    # τ floor: activity at the smallest-σ knot that has any real values
    # (inversion's min-σ knot is its step-0 NaN, so walk up)
    floor = None
    for j in range(s - 1, -1, -1):
        if not np.all(np.isnan(act[j])):
            floor = act[j]
            break
    e_curves = {}
    taus = {}
    for q in TAU_QUANTILES:
        tau = float(np.nanquantile(floor, q)) if floor is not None else float("nan")
        taus[f"q{int(q * 100)}"] = tau
        with np.errstate(invalid="ignore"):
            e = np.nanmean((act > tau).astype(np.float32), axis=(1, 2, 3))
        e_curves[f"q{int(q * 100)}"] = e
    with np.errstate(invalid="ignore"):
        hf_mean = np.nanmean(hf, axis=(1, 2, 3))
    if np.all(np.isnan(guide)):  # no-CFG trace (inversion arm)
        guide_mean = np.full(guide.shape[0], np.nan, dtype=np.float32)
    else:
        guide_mean = np.nanmean(guide, axis=(1, 2, 3))
    return {
        "sigmas": sig_d,
        "commit_cdf": np.asarray(commit_cdf),
        "E": e_curves,
        "taus": taus,
        "cbits": cbits,
        "hf_mean": hf_mean,
        "guide_mean": guide_mean,
    }


def _agg(stack: np.ndarray) -> dict:
    return {
        "mean": np.nanmean(stack, axis=0).round(4).tolist(),
        "p25": np.nanpercentile(stack, 25, axis=0).round(4).tolist(),
        "p75": np.nanpercentile(stack, 75, axis=0).round(4).tolist(),
    }


def aggregate_arm(profiles: list[dict]) -> dict:
    sig0 = profiles[0]["sigmas"]
    for pr in profiles[1:]:
        assert np.allclose(pr["sigmas"], sig0, atol=1e-6), "σ-knot mismatch"
    out = {
        "n_traces": len(profiles),
        "sigmas": sig0.round(6).tolist(),
        "commit_cdf": _agg(np.stack([pr["commit_cdf"] for pr in profiles])),
        "E": {
            q: _agg(np.stack([pr["E"][q] for pr in profiles])) for q in profiles[0]["E"]
        },
        "tau": {
            q: round(float(np.mean([pr["taus"][q] for pr in profiles])), 5)
            for q in profiles[0]["taus"]
        },
        "cbits_mean": np.nanmean(np.stack([pr["cbits"] for pr in profiles]), axis=0)
        .round(4)
        .tolist(),  # (S, C)
        "hf_mean": np.nanmean(np.stack([pr["hf_mean"] for pr in profiles]), axis=0)
        .round(5)
        .tolist(),
    }
    guide = np.stack([pr["guide_mean"] for pr in profiles])
    if not np.all(np.isnan(guide)):
        out["guide_mean"] = np.nanmean(guide, axis=0).round(5).tolist()
    return out


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b) -> float:
    return _pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def cross_arm_summary(gen: dict, inv: dict, static: dict | None) -> dict:
    """(c): does generation match inversion (and the static column)?"""
    out: dict = {}
    if gen and inv:
        g_cdf = np.asarray(gen["commit_cdf"]["mean"])
        i_cdf = np.asarray(inv["commit_cdf"]["mean"])
        out["commit_cdf_max_gap"] = round(float(np.abs(g_cdf - i_cdf).max()), 4)
        g_e = np.asarray(gen["E"]["q95"]["mean"], dtype=np.float64)
        i_e = np.asarray(inv["E"]["q95"]["mean"], dtype=np.float64)
        ok = ~(np.isnan(g_e) | np.isnan(i_e))
        out["E_q95_mean_abs_gap"] = round(float(np.abs(g_e - i_e)[ok].mean()), 4)
        g_cb = np.asarray(gen["cbits_mean"])  # (S, C)
        i_cb = np.asarray(inv["cbits_mean"])
        per_knot_r = [_pearson(g_cb[j], i_cb[j]) for j in range(g_cb.shape[0])]
        out["cbits_channel_pearson_per_knot"] = [round(r, 4) for r in per_knot_r]
        out["cbits_channel_pearson_min"] = round(float(np.nanmin(per_knot_r)), 4)
        out["cbits_channel_pearson_median"] = round(float(np.nanmedian(per_knot_r)), 4)
        out["cbits_channel_spearman_sigma_min"] = round(
            _spearman(g_cb[-1], i_cb[-1]), 4
        )
    if static:
        st = np.asarray(static["cbits_mean"])
        for name, arm in (("gen", gen), ("inv", inv)):
            if arm:
                final = np.asarray(arm["cbits_mean"])[-1]
                out[f"static_vs_{name}_final_cbits_pearson"] = round(
                    _pearson(st, final), 4
                )
                out[f"static_vs_{name}_final_cbits_spearman"] = round(
                    _spearman(st, final), 4
                )
    return out


# --------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    run_dir = make_run_dir("traj_stats", label=args.label)
    seeds = SEED_POOL[: args.num_seeds]

    manifest: list[dict] = []
    static_summary = None
    bundle = None
    base_args = None

    if "gen" in arms or "inv" in arms:
        from anima_lora import GenerationRequest
        from library.runtime.harness import build_inference_bundle

        req = GenerationRequest(
            dit=DIT,
            vae=VAE,
            text_encoder=TEXT_ENCODER,
            prompt="placeholder",
            image_size=(args.size[0], args.size[1]),
            infer_steps=args.steps,
            guidance_scale=args.cfg,
            sampler=args.sampler,
            seed=seeds[0],
            save_path=str(run_dir),
        )
        base_args = req.to_args()
        bundle = build_inference_bundle(base_args, with_vae=False)

    if "gen" in arms:
        prompts = pick_prompts(args.prompts_json, args.smoke)
        manifest += run_gen_arm(bundle, base_args, run_dir, prompts, seeds, args.k)

    if "inv" in arms:
        picks = pick_inversion_images(args.num_inv)
        manifest += run_inv_arm(bundle, base_args, run_dir, picks, args.steps, args.k)

    if "static" in arms:
        log.info("[static] sampling %d corpus latents ...", args.static_sample)
        static_summary = run_static_arm(args.static_sample, args.k)

    # ---- aggregate --------------------------------------------------------
    agg: dict = {}
    for arm in ("gen", "inv"):
        entries = [m for m in manifest if m["arm"] == arm]
        if entries:
            profiles = [trace_profile(m["trace"]) for m in entries]
            agg[arm] = aggregate_arm(profiles)
    if static_summary:
        agg["static"] = static_summary
    agg["cross_arm"] = cross_arm_summary(agg.get("gen"), agg.get("inv"), static_summary)

    (run_dir / "atlas_summary.json").write_text(json.dumps(agg, indent=2))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    headline = {
        "arms": arms,
        "n_gen": sum(m["arm"] == "gen" for m in manifest),
        "n_inv": sum(m["arm"] == "inv" for m in manifest),
        "cross_arm": agg["cross_arm"],
    }
    for arm in ("gen", "inv"):
        if arm in agg:
            cdf = agg[arm]["commit_cdf"]["mean"]
            sig = agg[arm]["sigmas"]
            mid = len(cdf) // 2
            headline[f"{arm}_commit_cdf_mid"] = {"sigma": sig[mid], "cdf": cdf[mid]}
            headline[f"{arm}_E_q95_last"] = agg[arm]["E"]["q95"]["mean"][-1]

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=["atlas_summary.json", "manifest.json"],
    )
    log.info("atlas done: %s", run_dir)
    log.info("headline: %s", json.dumps(headline["cross_arm"], indent=1))


if __name__ == "__main__":
    main()
