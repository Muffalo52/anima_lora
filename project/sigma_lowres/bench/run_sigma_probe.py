#!/usr/bin/env python
"""sigma_lowres Measurement B — per-σ-bin gradient-equivalence probe.

The tier_routing Phase 3a instrument (redraw-floor null, re-encode confound
control, demote arms) with one change: gradients are accumulated into
**per-σ-bin buckets** instead of one pooled estimate, so the demotion gap
becomes a curve gap_e(σ) instead of a scalar. Phase 3a marginalized σ out;
SwD's spectral analysis (arXiv:2503.16397 §3) pre-registers the hypothesis
that the gap concentrates below a tier-specific σ* and collapses above it
(``project/sigma_lowres/initial_proposal.md`` H2/H3).

σ bins are uniform on (0, 1) — the mechanism axis. Per-bin means across
images are the verdict quantity (the estimator class that was reliable in
3a); per-image per-bin rows land in ``per_image.jsonl`` for split-half
reliability analysis.

Usage::

    uv run python project/sigma_lowres/bench/run_sigma_probe.py \
        --adapter output/ckpt/anima_soup_sincos.safetensors --label phase0
    uv run python project/sigma_lowres/bench/run_sigma_probe.py \
        --adapter <ckpt> --smoke --label smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)
from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402
    DIT,
    VAE,
    cosine,
    encode_probe_latents,
    spearman,
)
from library.io.cache import (  # noqa: E402
    load_cached_latents,
    load_cached_text_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--adapter", required=True, help="trained plain-LoRA checkpoint")
    p.add_argument("--dit", default=DIT)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--num_images", type=int, default=40)
    p.add_argument("--bins", type=int, default=8, help="uniform σ bins on (0,1)")
    p.add_argument("--draws_per_bin", type=int, default=8)
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--demote_edges", default="896,768")
    p.add_argument("--artists", default=None, help="csv restriction on the corpus")
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument("--score_limit", type=int, default=None)
    p.add_argument("--no_reenc_control", action="store_true")
    p.add_argument(
        "--per_group",
        action="store_true",
        help="additionally report per-parameter-group gaps (Q2 J-decomposition): "
        "module types (incl. lora_up row-splits of the fused qkv/kv projs — "
        "the RoPE q/k-vs-v discriminator) x 28 blocks. Bookkeeping only — "
        "same forwards/backwards, per-slice cosines of the same flat "
        "gradient vectors.",
    )
    p.add_argument(
        "--endpoint_bin",
        action="store_true",
        help="append an exact sigma=1.0 bin (input = pure noise; any gap there "
        "is the target x graph floor — the S2/S3 term of the two-term account)",
    )
    p.add_argument(
        "--x_zero",
        action="store_true",
        help="zero the image in BOTH input and target on every grid (input = "
        "sigma*eps, target = eps; captions and latent shapes kept). Isolates "
        "pure graph-shape gradient sensitivity — no content anywhere. Implies "
        "--no_reenc_control (re-encode of nothing = the floor arm).",
    )
    p.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="use gradient checkpointing INSTEAD of block compile (fallback)",
    )
    p.add_argument(
        "--activation_memory_budget",
        type=float,
        default=0.99,
        help="partitioner knapsack cap under compile (freefit knee; 1.0 = off)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.num_images = 4
        args.bins = 4
        args.draws_per_bin = 2
        args.demote_edges = "896"
        args.score_limit = args.score_limit or 120
    return args


def start_heartbeat(interval: float = 45.0) -> None:
    """Unconditional keep-alive line every ``interval`` seconds.

    The daemon's command-job stall watchdog kills anything silent for 120 s
    (``ANIMA_DAEMON_CMD_STALL_TIMEOUT``), and the probe's longest quiet
    stretches — the first-call inductor compile and each 64-draw arm — exceed
    that. A plain daemon thread outlives every silent phase.
    """
    t0 = time.time()

    def beat() -> None:
        while True:
            time.sleep(interval)
            print(f"[hb] {time.time() - t0:.0f}s", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def bin_sigmas(bins: int, draws: int) -> torch.Tensor:
    """(bins, draws) σ grid: uniform bins on (0,1), stratified midpoints
    inside each bin. Uniform (not training-density) — per-bin means make the
    marginal density irrelevant, and σ is the axis under test."""
    b = torch.arange(bins, dtype=torch.float64).view(-1, 1)
    j = (torch.arange(draws, dtype=torch.float64) + 0.5).view(1, -1)
    return ((b + j / draws) / bins).to(torch.float32)


def build_sigmas(bins: int, draws: int, endpoint: bool) -> torch.Tensor:
    """Uniform-bin grid, optionally with an exact σ=1.0 bin appended.
    ``--bins 0 --endpoint_bin`` gives an endpoint-only grid."""
    parts = []
    if bins > 0:
        parts.append(bin_sigmas(bins, draws))
    if endpoint:
        parts.append(torch.ones(1, draws, dtype=torch.float32))
    if not parts:
        raise SystemExit("need --bins > 0 and/or --endpoint_bin")
    return torch.cat(parts, dim=0)


GROUP_RE = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")


def build_groups(network) -> dict[str, list[tuple[int, int]]]:
    """Group -> flat-vector ranges (sorted-name order — must match
    ``grad_estimate_binned``'s flatten), at two levels: ``type:<module minus
    block prefix>`` and ``block:<idx>``.

    Fused projections additionally get row-block sub-groups on ``lora_up``
    (rows are contiguous in the row-major flatten): ``self_attn_qkv_proj`` →
    ``type:self_attn_up_{q,k,v}`` and ``cross_attn_kv_proj`` →
    ``type:cross_attn_up_{k,v}``. RoPE touches self-attn q/k only, so the
    q/k-vs-v contrast is the RoPE discriminator; ``lora_down`` is shared
    across the fused heads and stays only in the module-level group."""
    named = [(n, p) for n, p in sorted(network.named_parameters()) if p.requires_grad]
    groups: dict[str, list[tuple[int, int]]] = {}
    pos = 0
    for name, p in named:
        s, e = pos, pos + p.numel()
        pos = e
        m = GROUP_RE.match(name.split(".")[0])
        keys = (
            (f"type:{m.group(2)}", f"block:{int(m.group(1)):02d}")
            if m
            else ("type:other", "block:other")
        )
        for k in keys:
            groups.setdefault(k, []).append((s, e))
        if m and name.endswith(".lora_up.weight"):
            typ = m.group(2)
            if typ == "self_attn_qkv_proj":
                third = p.numel() // 3  # rows are [q; k; v] blocks
                for j, sub in enumerate(("q", "k", "v")):
                    groups.setdefault(f"type:self_attn_up_{sub}", []).append(
                        (s + j * third, s + (j + 1) * third)
                    )
            elif typ == "cross_attn_kv_proj":
                half = p.numel() // 2  # rows are [k; v] blocks
                for j, sub in enumerate(("k", "v")):
                    groups.setdefault(f"type:cross_attn_up_{sub}", []).append(
                        (s + j * half, s + (j + 1) * half)
                    )
    return groups


def grouped_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    groups: dict[str, list[tuple[int, int]]],
) -> dict[str, float]:
    """Per-group cosine of two flat gradient vectors over each group's
    flat-vector ranges."""
    out = {}
    for g, ranges in groups.items():
        d = na = nb = 0.0
        for s, e in ranges:
            va, vb = a[s:e], b[s:e]
            d += float(va.dot(vb))
            na += float(va.dot(va))
            nb += float(vb.dot(vb))
        out[g] = d / (na**0.5 * nb**0.5) if na > 0 and nb > 0 else 0.0
    return out


def grad_estimate_binned(
    bundle,
    latents: torch.Tensor,
    crossattn: torch.Tensor,
    sigmas: torch.Tensor,  # (bins, draws)
    seeds: list[int],  # len == bins * draws
) -> tuple[list[torch.Tensor], list[float]]:
    """Per-σ-bin accumulated-gradient estimates.

    Returns ``(vecs, norms)``: per bin, the flattened LoRA gradient summed
    over that bin's draws (float32, CPU) and its L2 norm. Same forward/
    backward cost as the pooled 3a estimator at equal total draws — only the
    accumulator is split.
    """
    device = bundle.device
    params = [
        p for _, p in sorted(bundle.network.named_parameters()) if p.requires_grad
    ]
    lat = latents.unsqueeze(0).to(device)  # (1, 16, H, W) float32
    pad = torch.zeros(
        1, 1, lat.shape[-2], lat.shape[-1], dtype=torch.bfloat16, device=device
    )
    vecs: list[torch.Tensor] = []
    norms: list[float] = []
    n_bins, n_draws = sigmas.shape
    for b in range(n_bins):
        for p in params:
            p.grad = None
        for j in range(n_draws):
            seed = seeds[b * n_draws + j]
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(
                lat.shape, generator=gen, device=device, dtype=lat.dtype
            )
            sigma_b = sigmas[b, j].to(device).view(1)
            noisy = (1.0 - sigma_b) * lat + sigma_b * noise
            target = noise - lat
            noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = bundle.anima(noisy_5d, sigma_b, crossattn, padding_mask=pad)
            pred = pred.squeeze(2).float()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
        vec = torch.cat([p.grad.detach().float().flatten().cpu() for p in params])
        vecs.append(vec)
        norms.append(float(vec.norm()))
    for p in params:
        p.grad = None
    return vecs, norms


def main() -> None:
    args = parse_args()
    start_heartbeat()
    edges = [int(e) for e in args.demote_edges.split(",") if e]
    if args.x_zero:
        args.no_reenc_control = True
    reenc_control = not args.no_reenc_control
    device = torch.device("cuda")
    sigmas = build_sigmas(args.bins, args.draws_per_bin, args.endpoint_bin)
    total_draws = int(sigmas.numel())

    log.info("scoring corpus + selecting probe set (3a-compatible)…")
    artists = args.artists.split(",") if args.artists else None
    records = score_corpus(artists=artists, k=args.quant_k, limit=args.score_limit)
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    log.info(
        f"probe set: {len(probe)} images, {len({r.artist for r in probe})} artists; "
        f"{args.bins} σ-bins × {args.draws_per_bin} draws × "
        f"{2 + int(reenc_control) + len(edges)} arms"
    )

    log.info(f"encoding demoted arms ({edges}) + reenc={reenc_control}…")
    extra_latents = encode_probe_latents(probe, edges, args.vae, device, reenc_control)
    if args.x_zero:
        # keep the exact demoted grid shapes, drop all content
        extra_latents = {k: torch.zeros_like(v) for k, v in extra_latents.items()}

    from library.runtime.harness import build_anima, compile_blocks_for_training

    args.gradient_checkpointing = args.grad_ckpt
    args.compile = False  # compile is wired below (needs dynamic-seq marks)
    bundle = build_anima(args, dit_path=args.dit, adapter=args.adapter, train_mode=True)

    if not args.grad_ckpt:
        counts = {r.tokens for r in probe}
        counts.update(
            (t.shape[-2] // 2) * (t.shape[-1] // 2) for t in extra_latents.values()
        )
        compile_blocks_for_training(
            bundle.anima,
            bundle.network,
            backend="inductor",
            mode=None,
            n_token_families=len(counts),
            seq_range=(min(counts), max(counts)),
            dynamic_seq=True,
            activation_memory_budget=args.activation_memory_budget,
            partitioner_aggressive_recomputation=True,
            grad_ckpt=False,
        )

    groups: dict[str, list[tuple[int, int]]] | None = None
    if args.per_group:
        groups = build_groups(bundle.network)
        n_type = sum(1 for g in groups if g.startswith("type:"))
        n_block = sum(1 for g in groups if g.startswith("block:"))
        log.info(f"per-group: {n_type} type groups + {n_block} block groups")

    centers = [round(float(s), 4) for s in sigmas.mean(dim=1)]
    run_dir = make_run_dir(
        "sigma_lowres", args.label, root=Path(__file__).resolve().parent / "results"
    )
    rows_path = run_dir / "per_image.jsonl"
    rows: list[dict] = []
    t0 = time.time()

    for i, r in enumerate(probe):
        crossattn, _ = load_cached_text_features(r.te_path, variant=0)
        if crossattn is None:
            log.info(f"  [{i}] {r.artist}/{r.stem}: no crossattn_emb — skipped")
            continue
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        native = load_cached_latents(r.npz_path)[0]
        if args.x_zero:
            native = torch.zeros_like(native)

        def seeds(arm_idx: int) -> list[int]:
            base = args.seed * 1_000_000 + i * 10_000 + arm_idx * 1_000
            return [base + d for d in range(total_draws)]

        g_a, n_a = grad_estimate_binned(bundle, native, crossattn, sigmas, seeds(0))
        g_b, n_b = grad_estimate_binned(bundle, native, crossattn, sigmas, seeds(1))
        floor = [cosine(a, b) for a, b in zip(g_a, g_b)]
        row = {
            "artist": r.artist,
            "stem": r.stem,
            "redundancy": round(r.redundancy, 4),
            "tokens_native": r.tokens,
            "sigma_centers": centers,
            "cos_floor": [round(c, 5) for c in floor],
            "gnorm_native": [round(0.5 * (x + y), 3) for x, y in zip(n_a, n_b)],
        }
        floor_g: list[dict[str, float]] = []
        if args.per_group:
            floor_g = [grouped_cosine(a, b, groups) for a, b in zip(g_a, g_b)]
            row["cosg_floor"] = {g: [round(fb[g], 5) for fb in floor_g] for g in groups}

        def grouped_gaps(g_arm: list[torch.Tensor]) -> dict[str, list[float]]:
            out: dict[str, list[float]] = {g: [] for g in groups}
            for bi, (a, b, d) in enumerate(zip(g_a, g_b, g_arm)):
                ca = grouped_cosine(a, d, groups)
                cb = grouped_cosine(b, d, groups)
                for g in groups:
                    out[g].append(round(floor_g[bi][g] - 0.5 * (ca[g] + cb[g]), 5))
            return out

        arm_idx = 2
        if reenc_control:
            re_lat = extra_latents[(r.stem, "reenc")]
            g_re, _ = grad_estimate_binned(
                bundle, re_lat, crossattn, sigmas, seeds(arm_idx)
            )
            arm_idx += 1
            c = [0.5 * (cosine(a, g) + cosine(b, g)) for a, b, g in zip(g_a, g_b, g_re)]
            row["cos_reenc"] = [round(v, 5) for v in c]
            row["gap_reenc"] = [round(f - v, 5) for f, v in zip(floor, c)]
            if args.per_group:
                row["gapg_reenc"] = grouped_gaps(g_re)
        for e in edges:
            lat = extra_latents[(r.stem, f"demote{e}")]
            g_d, n_d = grad_estimate_binned(
                bundle, lat, crossattn, sigmas, seeds(arm_idx)
            )
            arm_idx += 1
            c = [0.5 * (cosine(a, g) + cosine(b, g)) for a, b, g in zip(g_a, g_b, g_d)]
            row[f"cos_{e}"] = [round(v, 5) for v in c]
            row[f"gap_{e}"] = [round(f - v, 5) for f, v in zip(floor, c)]
            row[f"gnorm_{e}"] = [round(v, 3) for v in n_d]
            if args.per_group:
                row[f"gapg_{e}"] = grouped_gaps(g_d)
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        gap_str = " ".join(
            f"{k}={[f'{v:+.3f}' for v in row[k]]}" for k in row if k.startswith("gap_")
        )
        log.info(f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} {gap_str}")

    if not rows:
        raise SystemExit("no per-image rows produced")

    def bin_stats(key: str) -> dict:
        m = np.array([r[key] for r in rows])  # (n_images, bins)
        mean = m.mean(axis=0)
        sem = m.std(axis=0, ddof=1) / np.sqrt(m.shape[0])
        # split-half reliability of the bin-mean curve (odd/even image split)
        h1, h2 = m[0::2].mean(axis=0), m[1::2].mean(axis=0)
        return {
            "mean": [round(float(v), 5) for v in mean],
            "sem": [round(float(v), 5) for v in sem],
            "spearman_sigma": round(spearman(np.arange(len(mean)), mean), 4)
            if len(mean) > 1
            else None,
            "splithalf_pearson": round(float(np.corrcoef(h1, h2)[0, 1]), 4)
            if len(mean) > 2
            else None,
        }

    headline: dict = {
        "n_images": len(rows),
        "bins": args.bins,
        "draws_per_bin": args.draws_per_bin,
        "sigma_centers": centers,
        "adapter": args.adapter,
        "cos_floor": bin_stats("cos_floor"),
        "gnorm_native": bin_stats("gnorm_native"),
        "wall_time_s": round(time.time() - t0, 1),
    }
    if reenc_control:
        headline["gap_reenc"] = bin_stats("gap_reenc")
    for e in edges:
        headline[f"gap_{e}"] = bin_stats(f"gap_{e}")

    if args.per_group:

        def group_stats(key: str) -> dict:
            out = {}
            for g in rows[0][key]:
                m = np.array([r[key][g] for r in rows])
                mean = m.mean(axis=0)
                sem = m.std(axis=0, ddof=1) / np.sqrt(m.shape[0])
                h1, h2 = m[0::2].mean(axis=0), m[1::2].mean(axis=0)
                out[g] = {
                    "mean": [round(float(v), 5) for v in mean],
                    "sem": [round(float(v), 5) for v in sem],
                    "splithalf_pearson": round(float(np.corrcoef(h1, h2)[0, 1]), 4)
                    if len(mean) > 2
                    else None,
                }
            return out

        for key in [k for k in rows[0] if k.startswith("gapg_")]:
            headline[key] = group_stats(key)
        for e in edges:
            stats = headline.get(f"gapg_{e}")
            if not stats:
                continue
            tg = {
                g[5:]: v["mean"][-1] for g, v in stats.items() if g.startswith("type:")
            }
            ranked = sorted(tg.items(), key=lambda kv: -kv[1])
            log.info(
                f"[per-group] {e} type gaps @ last bin: "
                + ", ".join(f"{n}={v:+.3f}" for n, v in ranked)
            )

    log.info(json.dumps(headline, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path],
        extra={"probe_set": [f"{r['artist']}/{r['stem']}" for r in rows]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
