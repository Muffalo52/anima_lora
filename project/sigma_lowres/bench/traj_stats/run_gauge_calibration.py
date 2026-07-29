#!/usr/bin/env python
"""traj_stats Phase 2 — gauge calibration: known-bad vs known-good.

The acceptance test for the trajectory-intactness gauge
(``_archive/proposals/traj_latent_stats.md`` Phase 2): record matched
(prompt, seed) trace grids for a baseline and three candidate arms, run
``gauge.py`` on each, and demand

* **foveation** (``--fovea_sigma_c 0.75`` — the archived known-bad) is flagged
  ``process-broken``, rediscovering P4t from traces alone, *without looking
  at any final image* (measured signature: commit-CDF hole + in-loop hf
  blow-up from the piecewise-constant periphery — the flatline the proposal
  predicted only exists post-readout);
* **SMC-CFG** (shipped, quality-neutral) lands ``intact``;
* **Spectrum** lands ``perturbed`` — see EXPECT below for why not "intact".

All arms run ``--sampler euler``: the foveated runner forces Euler, so the
baseline must integrate identically or D would measure the sampler, not the
intervention. If this gate fails on the known-good side, re-tune
``gauge.BANDS`` / τ / normalization before anyone trusts the gauge; if it
fails on the known-bad side, the trace set is insufficient (proposal
falsifier 3) — stop.

Usage::

    uv run python project/sigma_lowres/bench/traj_stats/run_gauge_calibration.py --label phase2
    uv run python project/sigma_lowres/bench/traj_stats/run_gauge_calibration.py --smoke --label smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from project.sigma_lowres.bench.traj_stats.gauge import gauge  # noqa: E402
from project.sigma_lowres.bench.traj_stats.run_atlas import (  # noqa: E402
    DIT,
    TEXT_ENCODER,
    VAE,
    pick_prompts,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 1337]
FOVEA_SIGMA_C = 0.75

# arm -> (args attr, value) applied on top of the shared baseline namespace
ARMS = {
    "spectrum": [("spectrum", True)],
    "smc": [("smc_cfg", True)],
    "fovea": [("fovea_sigma_c", FOVEA_SIGMA_C)],
}
# Pre-registered expectations, one exemplar per band. Spectrum is expected
# "perturbed", NOT "intact": the first calibration run showed its forecast
# steps are process-visible (dE −0.65 in the forecast band) even though the
# output is quality-neutral by its own bench — quality-neutral ≠
# process-transparent, and the gauge measures the latter.
EXPECT = {"spectrum": "perturbed", "smc": "intact", "fovea": "process-broken"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--k", type=int, default=4)
    p.add_argument(
        "--prompts_json", default="bench/tag_dropout/results/prompt_sets_mignon.json"
    )
    p.add_argument(
        "--smoke", action="store_true", help="512², 8 steps, 2 prompts × 1 seed"
    )
    p.add_argument(
        "--reuse",
        default=None,
        metavar="RUN_DIR",
        help="Skip rendering; gauge the traces_* dirs of a previous run "
        "(offline re-calibration after a BANDS / verdict-logic change).",
    )
    p.add_argument("--label", default=None)
    return p.parse_args()


def record_arm(
    bundle, base_args, traces_dir: Path, prompts, seeds, k, overrides
) -> None:
    for prompt in prompts:
        for seed in seeds:
            a = copy.copy(base_args)
            a.prompt = prompt["text"]
            a.seed = seed
            a.traj_stats = True
            a.traj_stats_dir = str(traces_dir)
            a.traj_stats_k = k
            for attr, val in overrides:
                setattr(a, attr, val)
            t0 = time.perf_counter()
            bundle.generate(a)
            log.info(
                "  %s seed=%d  %.1fs", prompt["id"], seed, time.perf_counter() - t0
            )


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(
        "traj_stats", label=args.label, root=Path(__file__).resolve().parent / "results"
    )
    seeds = SEEDS[:1] if args.smoke else SEEDS
    if args.smoke:
        args.size, args.steps = [512, 512], 8
    prompts = pick_prompts(args.prompts_json, smoke=True)  # detailed×1+sparse×1
    if not args.smoke:
        prompts = pick_prompts(args.prompts_json, smoke=False)[:4]

    if args.reuse:
        src = Path(args.reuse)
        traces_root = src
        log.info("reusing traces from %s (no rendering)", src)
    else:
        traces_root = run_dir

        from anima_lora import GenerationRequest
        from library.runtime.harness import build_inference_bundle

        import networks.foveated  # noqa: F401 — registers the foveated runner
        import networks.spectrum  # noqa: F401 — registers the spectrum runner

        req = GenerationRequest(
            dit=DIT,
            vae=VAE,
            text_encoder=TEXT_ENCODER,
            prompt="placeholder",
            image_size=(args.size[0], args.size[1]),
            infer_steps=args.steps,
            guidance_scale=args.cfg,
            sampler="euler",
            seed=seeds[0],
            save_path=str(run_dir),
        )
        base_args = req.to_args()
        bundle = build_inference_bundle(base_args, with_vae=False)

        log.info("[baseline] %d renders", len(prompts) * len(seeds))
        record_arm(
            bundle, base_args, run_dir / "traces_baseline", prompts, seeds, args.k, []
        )
        for arm, overrides in ARMS.items():
            log.info("[%s] %d renders", arm, len(prompts) * len(seeds))
            record_arm(
                bundle,
                base_args,
                run_dir / f"traces_{arm}",
                prompts,
                seeds,
                args.k,
                overrides,
            )

    results = {}
    gate = {}
    for arm in ARMS:
        res = gauge(
            str(traces_root / "traces_baseline"), str(traces_root / f"traces_{arm}")
        )
        (run_dir / f"gauge_{arm}.json").write_text(json.dumps(res, indent=2))
        results[arm] = {
            "verdict": res["verdict"],
            "reasons": res["reasons"],
            "aggregate": res["aggregate"],
        }
        gate[arm] = res["verdict"] == EXPECT[arm]
        log.info(
            "[gauge:%s] verdict=%s (expect %s) %s",
            arm,
            res["verdict"],
            EXPECT[arm],
            res["reasons"],
        )

    verdict = "PASS" if all(gate.values()) else "FAIL"
    metrics = {
        "verdict": verdict,
        "gate": gate,
        "expected": EXPECT,
        "arms": results,
        "config": {
            "steps": args.steps,
            "cfg": args.cfg,
            "size": args.size,
            "sampler": "euler",
            "seeds": seeds,
            "n_prompts": len(prompts),
            "fovea_sigma_c": FOVEA_SIGMA_C,
        },
    }
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[f"gauge_{arm}.json" for arm in ARMS],
    )
    log.info("verdict: %s  gate=%s", verdict, gate)
    log.info("envelope: %s", out)
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
