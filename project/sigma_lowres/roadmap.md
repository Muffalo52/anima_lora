# sigma_lowres — roadmap

Status: Phase 0 DONE (spectral mechanism refuted, σ-map measured), Phase 1a
DONE (ratio transfer FAILED), pooled-gradient probe DONE 2026-07-25,
**1280→1024 probe DONE 2026-07-26** (Q1 answered: ratio refuted as governor,
capacity predicts the ordering, but the safe threshold is route-dependent —
1280→1024 floors at σ ≥ ~0.75–0.875, not 0.5; report.md "1280→1024 probe").
Safe set = {1024→896 @ σ>0.5, 1280→1024 @ σ>σ\*∈(0.625,0.875)}. The cheap
harness for off-corpus tiers now exists (`prep_1280_probe.py` probe-local
cache + `--data_root` — no corpus re-preprocess, ~47 min/probe).

## Next: σ-window refinement (localize σ\* for 1280→1024)

`run_sigma_probe.py --sigma_window 0.5,1.0 --bins 5` on the same probe-local
cache — all bins in the crossover region (centers 0.55…0.95). **Started
2026-07-26, deprioritized at 5/24 images** (partial rows
`bench/results/20260726-2109/`; same command re-runs) in favor of the
prior-distance discriminator, which landed the same day (groundings.md G6:
no 1280 discontinuity, prior ↮ Floor — the Floor is graph-side). Payoff of resuming is gate-position-sensitive: σ\* ≈ 0.65 →
~9% epoch saving on 1280-tier data, σ\* ≈ 0.75 → ~5%. Then the decision
point:

- The corpus has no 1280 tier today (`target_res = [1024, 896]`), so the new
  route's practical value is conditional on adopting one; the map/paper value
  (3-route boundary σ\*(route)) is already banked.
- Phase 1b remains gated on judging {1024→896 @ σ>0.5} (~13–14%) + any
  adopted-1280 increment worth the dual-cache complexity.

## Phase 1b — trainer wiring **[BUILT 2026-07-26]** + the gate **[OPEN]**

Wiring shipped opt-in (`--sigma_lowres`, route pinned to 1024→896 @ σ>0.5) —
full description in `methods.md` §"Phase 1b trainer wiring". Key deviations
from the sketch below: σ is drawn trainer-side (σ-first via
`draw_flat_sigmas`, single source of density truth) rather than at batch
assembly, and the sibling cache is an **in-npz key** (`demoted_{H}x{W}`, `make
preprocess-demote`) rather than a stem-suffixed file — reconcile and bucket
discovery needed no changes at all.

- **Gate (still owed)**: fixed-steps A/B — CMMD non-inferior (within-run
  usage only, per `project_cmmd_val_signal`) + rendered spot-check + realized
  wall-clock logged. First A/B in flight: `tenth` preset × 4 epochs,
  `--sigma_lowres` arm vs baseline. Pitch is wall-clock at fixed steps,
  never "more steps in the same time" (autoscale lesson). CMMD regression →
  close the line (pre-committed).

## Phase 1c — bespoke loops (EC / turbo) — gated on separate probes

Each needs its own operating-point probe before any wiring (questions.md Q5).
Do not schedule until Phase 1b has shipped and survived its gate.

## Kill criteria

- 1280→1024 probe fails AND ~14% is judged below the dual-cache complexity
  bar → close; keep `rapsd.py` + `run_sigma_probe.py` as reusable
  instruments and the "spectral sufficiency ≠ gradient equivalence" finding.
- Phase 1b CMMD regression → close (pre-committed in the proposal).

## Pointers

Design: `project/sigma_lowres/initial_proposal.md` · Data:
`project/sigma_lowres/bench/report.md` · Memory: `project_sigma_lowres_phase0`,
`project_tier_routing_phase3a_failed` (split-half check mandatory).
