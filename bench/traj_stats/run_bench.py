#!/usr/bin/env python
"""traj_stats Phase 0 bench — recorder bit-exactness + overhead budget.

The acceptance gate for the passive trajectory recorder
(``_archive/proposals/traj_latent_stats.md`` Phase 0):

1. **Determinism control** — two recorder-OFF generations at the same seed
   produce bit-identical latents (otherwise bit-exactness is unmeasurable).
2. **Bit-exactness** — a recorder-ON generation produces latents bit-identical
   to the recorder-OFF baseline (pure observation).
3. **Overhead** — recorder-ON wall time ≤ 2 % over baseline at the benched
   resolution (proposal budget, quoted at 1024²).

Also drops the recorded ``.npz`` and its derived summary (E(σ) curve,
commit-CDF) as artifacts — the first real trajectory traces of Phase 1.

Usage::

    uv run python bench/traj_stats/run_bench.py --label phase0
    uv run python bench/traj_stats/run_bench.py --with_spectrum --label phase0-spec
    uv run python bench/traj_stats/run_bench.py --size 512 512 --steps 12  # smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER = "models/text_encoders/qwen_3_06b_base.safetensors"

OVERHEAD_BUDGET_PCT = 2.0

DEFAULT_PROMPT = (
    "1girl, silver hair, school uniform, cherry blossoms, detailed background, "
    "masterpiece, best quality"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--sampler", default="er_sde")
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=4, help="quantization bits/channel")
    p.add_argument(
        "--with_spectrum",
        action="store_true",
        help="Also run the pair under --spectrum (hook smoke).",
    )
    p.add_argument("--label", default=None)
    return p.parse_args()


def _timed_generate(bundle, args) -> tuple[torch.Tensor, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    latent = bundle.generate(args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return latent.detach().to("cpu"), time.perf_counter() - t0


def run_pair(bundle, base_args, run_dir: Path, tag: str, k: int) -> dict:
    """baseline ×2 (warm + determinism control) then recorder-on; compare."""
    off = copy.copy(base_args)
    # warm run: TE encode -> conds_cache, cudnn autotune, lazy inits
    lat_warm, t_warm = _timed_generate(bundle, off)
    lat_base, t_base = _timed_generate(bundle, off)
    deterministic = torch.equal(lat_warm, lat_base)

    on = copy.copy(base_args)
    on.traj_stats = True
    on.traj_stats_dir = str(run_dir / f"traces_{tag}")
    on.traj_stats_k = k
    # warm the recorder path too (one-time cuDNN conv-algo init for the 3x3
    # Laplacian) so both arms are timed at steady state; bit-exactness is
    # checked on BOTH recorder runs.
    lat_rec_warm, t_rec_warm = _timed_generate(bundle, on)
    lat_rec, t_rec = _timed_generate(bundle, on)
    bit_exact = torch.equal(lat_base, lat_rec) and torch.equal(lat_base, lat_rec_warm)

    traces = sorted(Path(on.traj_stats_dir).glob("*.npz"))
    npz = traces[-1] if traces else None
    overhead_pct = 100.0 * (t_rec - t_base) / t_base
    per_step_ms = 1000.0 * (t_rec - t_base) / base_args.infer_steps

    res = {
        "deterministic_control": bool(deterministic),
        "bit_exact": bool(bit_exact),
        "t_warm_s": round(t_warm, 3),
        "t_baseline_s": round(t_base, 3),
        "t_recorder_warm_s": round(t_rec_warm, 3),
        "t_recorder_s": round(t_rec, 3),
        "overhead_pct": round(overhead_pct, 3),
        "recorder_ms_per_step": round(per_step_ms, 3),
        "npz": str(npz) if npz else None,
        "npz_mb": round(npz.stat().st_size / 2**20, 2) if npz else None,
    }
    log.info(
        "[%s] deterministic=%s bit_exact=%s overhead=%.2f%% (%.3fs -> %.3fs)",
        tag,
        deterministic,
        bit_exact,
        overhead_pct,
        t_base,
        t_rec,
    )
    return res


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("traj_stats", label=args.label)

    from anima_lora import GenerationRequest
    from library.inference.traj_stats import derive_summary
    from library.runtime.harness import build_inference_bundle

    req = GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        prompt=args.prompt,
        image_size=(args.size[0], args.size[1]),
        infer_steps=args.steps,
        guidance_scale=args.cfg,
        sampler=args.sampler,
        seed=args.seed,
        save_path=str(run_dir),
    )
    base_args = req.to_args()
    # Latent-space probe: no decode, no VAE.
    bundle = build_inference_bundle(base_args, with_vae=False)

    results = {"main": run_pair(bundle, base_args, run_dir, "main", args.k)}

    if args.with_spectrum:
        import networks.spectrum  # noqa: F401 — registers the spectrum runner

        spec_args = copy.copy(base_args)
        spec_args.spectrum = True
        results["spectrum"] = run_pair(bundle, spec_args, run_dir, "spectrum", args.k)

    # Derived summary of the main-arm trace — the first Phase 1 data point.
    summary = None
    if results["main"]["npz"]:
        summary = derive_summary(results["main"]["npz"])
        (run_dir / "summary_main.json").write_text(json.dumps(summary, indent=2))

    gate = {
        "deterministic_control": all(
            r["deterministic_control"] for r in results.values()
        ),
        "bit_exact": all(r["bit_exact"] for r in results.values()),
        "overhead_within_budget": results["main"]["overhead_pct"]
        <= OVERHEAD_BUDGET_PCT,
    }
    verdict = "PASS" if all(gate.values()) else "FAIL"

    metrics = {
        "verdict": verdict,
        "gate": gate,
        "arms": results,
        "config": {
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "size": args.size,
            "seed": args.seed,
            "k": args.k,
        },
    }
    if summary is not None:
        metrics["trace_summary"] = {
            "tau": summary["tau"],
            "E_first_last": [summary["E"][1], summary["E"][-1]],
            "commit_cdf_mid": summary["commit_cdf"][len(summary["commit_cdf"]) // 2],
        }

    artifacts = ["summary_main.json"] if summary is not None else []
    for tag, r in results.items():
        if r["npz"]:
            dest = run_dir / f"trace_{tag}.npz"
            if Path(r["npz"]).resolve() != dest.resolve():
                shutil.copy2(r["npz"], dest)
            artifacts.append(dest.name)

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
    )
    log.info("verdict: %s  gate=%s", verdict, gate)
    log.info("envelope: %s", out)
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
