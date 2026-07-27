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
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
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
    p.add_argument(
        "--sigma_window",
        default="0,1",
        help="LO,HI sub-interval the uniform bins cover (e.g. 0.5,1.0 puts "
        "every bin in the high-σ crossover region); --endpoint_bin unaffected",
    )
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--demote_edges", default="896,768")
    p.add_argument(
        "--data_root",
        default=None,
        help="alternate dataset root holding lora/ + resized/ (e.g. the "
        "probe-local 1280 cache from prep_1280_probe.py); default = "
        "post_image_dataset",
    )
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
        "--pool",
        type=int,
        default=0,
        help="stratified gradient-pooling: sort the probe set by redundancy, "
        "chunk into strata of N images, and ADDITIONALLY report pooled gap "
        "curves — per stratum the per-image bin-gradients are summed across "
        "images (gradient accumulation = the batch-SGD aggregate object) "
        "before cosines, in two variants: unweighted (training-realistic, "
        "large-gnorm images dominate) and per-image-normalized (side-channel), "
        "plus an all-images aggregate. Pooled cosines are dominated by the "
        "shared cross-image gradient component, so pooled floors/gaps are NOT "
        "comparable to per-image gaps or the ±0.04 instrument band — each "
        "stratum carries its own noise-redraw floor and an image-split-half "
        "floor. Per-image rows are still written unchanged.",
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


def bin_sigmas(bins: int, draws: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """(bins, draws) σ grid: uniform bins on (lo, hi), stratified midpoints
    inside each bin. Uniform (not training-density) — per-bin means make the
    marginal density irrelevant, and σ is the axis under test. The window
    concentrates all bins in a sub-interval (crossover localization)."""
    b = torch.arange(bins, dtype=torch.float64).view(-1, 1)
    j = (torch.arange(draws, dtype=torch.float64) + 0.5).view(1, -1)
    u = (b + j / draws) / bins
    return (lo + (hi - lo) * u).to(torch.float32)


def build_sigmas(
    bins: int, draws: int, endpoint: bool, lo: float = 0.0, hi: float = 1.0
) -> torch.Tensor:
    """Uniform-bin grid over the (lo, hi) window, optionally with an exact
    σ=1.0 bin appended. ``--bins 0 --endpoint_bin`` gives an endpoint-only
    grid."""
    parts = []
    if bins > 0:
        parts.append(bin_sigmas(bins, draws, lo, hi))
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


class PoolAccumulator:
    """Cross-image gradient accumulator for one stratum (or the aggregate).

    Holds, per arm and per σ-bin, the running sum of per-image bin-gradient
    vectors (``sums``, unweighted — the batch-SGD object) and of per-image
    L2-normalized vectors (``nsums`` — the equal-weight side-channel), plus a
    parity split of the native (a+b) sum for the image-split-half floor.
    Everything is float32 CPU; memory is O(arms x bins) vectors regardless of
    stratum size — but each vector is a full flat LoRA gradient (~311 MB at
    77M params), so one accumulator is ~19 GB at 5 arms x 5 bins. With
    ``backing_dir`` set, every accumulator vector lives in a disk memmap
    instead of RAM (pages are cache-evictable) — used for the all-images
    aggregate, which is written 10x but read once at the end. Without it the
    aggregate + stratum accumulators together OOM a 46 GB box. ``release()``
    additionally closes the memmap handles between merges: mapped file pages
    count against process RSS while a handle is open even though they're
    reclaimable, so a released aggregate costs ~zero RSS outside merges.
    """

    def __init__(self, backing_dir: Path | None = None) -> None:
        self.backing_dir = backing_dir
        self._numel: int | None = None
        self.sums: dict[str, list[torch.Tensor]] = {}
        self.nsums: dict[str, list[torch.Tensor]] = {}
        self.halves: dict[int, list[torch.Tensor]] = {}
        self.n = 0
        self.redundancy: list[float] = []

    def _stores(self) -> tuple[tuple[str, dict], ...]:
        return ("sums", self.sums), ("nsums", self.nsums), ("halves", self.halves)

    def _open(self, name: str, key, idx: int, mode: str) -> torch.Tensor:
        mm = np.memmap(
            self.backing_dir / f"{name}_{key}_{idx}.f32",
            dtype=np.float32,
            mode=mode,
            shape=(self._numel,),
        )
        return torch.from_numpy(mm)

    def _materialize(self, name: str, key, idx: int, v: torch.Tensor) -> torch.Tensor:
        if self.backing_dir is None:
            return v
        self.backing_dir.mkdir(parents=True, exist_ok=True)
        self._numel = v.numel()
        t = self._open(name, key, idx, "w+")
        t.copy_(v)
        return t

    def _add(
        self,
        name: str,
        store: dict,
        key,
        vecs: list[torch.Tensor],
        scales: list[float] | None = None,
    ) -> None:
        if scales is None:
            scales = [1.0] * len(vecs)
        if key not in store:
            store[key] = [
                self._materialize(name, key, i, v * s)
                for i, (v, s) in enumerate(zip(vecs, scales))
            ]
        else:
            for acc, v, s in zip(store[key], vecs, scales):
                acc += v * s

    def release(self) -> None:
        """Backed mode: replace each vector list with its length, dropping the
        memmap handles (data is on disk). Reopened by ``ensure_open``."""
        if self.backing_dir is None:
            return
        for _, store in self._stores():
            for key, vecs in store.items():
                if not isinstance(vecs, int):
                    store[key] = len(vecs)

    def ensure_open(self) -> None:
        if self.backing_dir is None:
            return
        for name, store in self._stores():
            for key, val in store.items():
                if isinstance(val, int):
                    store[key] = [self._open(name, key, i, "r+") for i in range(val)]

    def add_image(self, arms: dict[str, list[torch.Tensor]], redundancy: float) -> None:
        for key, vecs in arms.items():
            self._add("sums", self.sums, key, vecs)
            self._add(
                "nsums",
                self.nsums,
                key,
                vecs,
                [1.0 / (float(v.norm()) + 1e-12) for v in vecs],
            )
        native = [a + b for a, b in zip(arms["a"], arms["b"])]
        self._add("halves", self.halves, self.n % 2, native)
        self.n += 1
        self.redundancy.append(redundancy)

    def merge(self, other: "PoolAccumulator") -> None:
        self.ensure_open()
        for name, mine in self._stores():
            theirs = getattr(other, name)
            for key, vecs in theirs.items():
                self._add(name, mine, key, vecs)
        self.n += other.n
        self.redundancy.extend(other.redundancy)


def pool_stats(acc: PoolAccumulator, arm_keys: list[str]) -> dict:
    """Pooled-cosine curves for one accumulator: noise-redraw floor
    (pooled-a vs pooled-b over the same images), per-arm cos/gap, the
    normalized variant (``norm_`` prefix), and the image-split-half floor
    (pooled native over even- vs odd-indexed images — includes image-sampling
    variance, which the redraw floor does not)."""
    acc.ensure_open()
    out: dict = {
        "n_images": acc.n,
        "redundancy_mean": round(float(np.mean(acc.redundancy)), 4),
        "redundancy_range": [
            round(min(acc.redundancy), 4),
            round(max(acc.redundancy), 4),
        ],
    }
    for prefix, store in (("", acc.sums), ("norm_", acc.nsums)):
        a, b = store["a"], store["b"]
        floor = [cosine(x, y) for x, y in zip(a, b)]
        out[f"{prefix}cos_floor"] = [round(v, 5) for v in floor]
        for key in arm_keys:
            d = store[key]
            c = [0.5 * (cosine(x, g) + cosine(y, g)) for x, y, g in zip(a, b, d)]
            out[f"{prefix}cos_{key}"] = [round(v, 5) for v in c]
            out[f"{prefix}gap_{key}"] = [round(f - v, 5) for f, v in zip(floor, c)]
    out["gnorm_pooled"] = [
        round(0.5 * (float(x.norm()) + float(y.norm())), 3)
        for x, y in zip(acc.sums["a"], acc.sums["b"])
    ]
    if len(acc.halves) == 2:
        out["imgsplit_floor"] = [
            round(cosine(h0, h1), 5) for h0, h1 in zip(acc.halves[0], acc.halves[1])
        ]
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
    lo, hi = (float(v) for v in args.sigma_window.split(","))
    if not 0.0 <= lo < hi <= 1.0:
        raise SystemExit(
            f"--sigma_window must satisfy 0 <= LO < HI <= 1, got {lo},{hi}"
        )
    sigmas = build_sigmas(args.bins, args.draws_per_bin, args.endpoint_bin, lo, hi)
    total_draws = int(sigmas.numel())

    log.info("scoring corpus + selecting probe set (3a-compatible)…")
    artists = args.artists.split(",") if args.artists else None
    records = score_corpus(
        artists=artists,
        k=args.quant_k,
        limit=args.score_limit,
        data_root=Path(args.data_root).resolve() if args.data_root else None,
    )
    probe = select_probe_set(
        records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
    )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    if args.pool:
        probe = sorted(probe, key=lambda r: r.redundancy)
        log.info(
            f"pool mode: {args.pool} images/stratum, sorted by redundancy "
            f"({probe[0].redundancy:.3f}..{probe[-1].redundancy:.3f})"
        )
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
    arm_keys = (["reenc"] if reenc_control else []) + [str(e) for e in edges]
    pool_strata: list[dict] = []
    pool_spill = run_dir / "pool_agg_spill"
    pool_agg = PoolAccumulator(backing_dir=pool_spill)
    cur_pool = PoolAccumulator()

    def finalize_stratum() -> None:
        if not args.pool or cur_pool.n == 0:
            return
        stats = pool_stats(cur_pool, arm_keys)
        pool_strata.append(stats)
        pool_agg.merge(cur_pool)
        pool_agg.release()  # drop memmap handles — RSS-free between merges
        cur_pool.__init__()
        gaps = " ".join(f"gap_{k}@last={stats[f'gap_{k}'][-1]:+.4f}" for k in arm_keys)
        log.info(
            f"[pool s{len(pool_strata) - 1}] n={stats['n_images']} "
            f"redundancy {stats['redundancy_range'][0]:.2f}"
            f"-{stats['redundancy_range'][1]:.2f} "
            f"floor@last={stats['cos_floor'][-1]:.4f} {gaps}"
        )

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

        arms: dict[str, list[torch.Tensor]] = {"a": g_a, "b": g_b}
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
            arms["reenc"] = g_re
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
            arms[str(e)] = g_d
        if args.pool:
            cur_pool.add_image(arms, r.redundancy)
            if cur_pool.n == args.pool:
                finalize_stratum()
        # free this image's ~8 GB of flat gradient vectors now — otherwise
        # the locals keep them resident through the next image's compute
        for vecs in arms.values():
            vecs.clear()
        g_a = g_b = arms = None  # noqa: F841
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        gap_str = " ".join(
            f"{k}={[f'{v:+.3f}' for v in row[k]]}" for k in row if k.startswith("gap_")
        )
        log.info(f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} {gap_str}")

    if not rows:
        raise SystemExit("no per-image rows produced")
    finalize_stratum()  # remainder stratum (may be smaller than --pool)

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

    if args.pool and pool_strata:
        # stratum-level redundancy trend: spearman of stratum redundancy mean
        # vs pooled gap at the last (highest-σ) bin, per demote edge
        trend = {}
        if len(pool_strata) > 2:
            red = np.array([s["redundancy_mean"] for s in pool_strata])
            for e in edges:
                g = np.array([s[f"gap_{e}"][-1] for s in pool_strata])
                trend[f"spearman_redundancy_gap_{e}"] = round(spearman(red, g), 4)
        headline["pool"] = {
            "size": args.pool,
            "strata": pool_strata,
            "aggregate": pool_stats(pool_agg, arm_keys),
            **trend,
        }
        del pool_agg  # release memmap handles before removing the spill
        shutil.rmtree(pool_spill, ignore_errors=True)

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
