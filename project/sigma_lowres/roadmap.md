# sigma_lowres — roadmap

Status: Phase 0 DONE (spectral mechanism refuted, σ-map measured), Phase 1a
DONE (ratio transfer FAILED). Safe set = {1024→896 @ σ>0.5}, ~13–14%
wall-clock ceiling. The line is **paused at a decision point**: the payoff is
real but small, and the next cheap probe (1280→1024) decides whether it grows.

## Next: the 1280→1024 discriminating probe (observability, cheap)

Settles ratio-vs-capacity (questions.md Q1). Steps:

1. Re-preprocess a small set of high-res sources at `--target_res` including
   the 1280 tier (probe set only — not the full corpus).
2. VRAM check at 6300 tokens on the probe harness (may need `--grad_ckpt`;
   remember [[feedback_default_block_compile]] — block-compile first on OOM).
3. `run_sigma_probe.py --tier 1280 --demote_edges 1024,896` — pre-register:
   capacity-governor predicts gap_1024 in the reenc band at σ ≥ 0.5;
   ratio-governor predicts it stays elevated like 896→768.

Outcomes:
- **PASS** → safe set gains its biggest per-draw route (0.65×); ceiling rises
  above ~14% on high-tier-heavy data; Phase 1b becomes clearly worth building.
- **FAIL** → ratio governs, the map is closed at one route, and Phase 1b is
  a judgment call on ~14% alone (likely: line closes as a finding).

## Phase 1b — trainer wiring + the gate (build only after the probe)

- σ drawn at batch-assembly time; σ > 0.5 → fetch the one-tier-down sibling
  cache (stem-suffixed, autoscale-emit pattern as design reference — its
  runtime was stripped 2026-06-28, do not resurrect blindly).
- Requires dual caches for demotable tiers (preprocess emit + reconcile
  support) — the complexity that must be paid for by the measured ceiling.
- **Gate**: fixed-steps A/B on ≥1 artist set — CMMD non-inferior (within-run
  usage only, per `project_cmmd_val_signal`) + rendered spot-check + realized
  wall-clock logged. Ship opt-in only. Pitch is wall-clock at fixed steps,
  never "more steps in the same time" (autoscale lesson).

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
