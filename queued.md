# Queued: merge-interference probe Phase 0 (calibration bench)

`docs/proposal/merge_interference_probe.md` — noise band + N-scaling +
overlapping-artist information cell (~1.5h GPU, existing ckpts, no training).
Gates the `check_merge.py` ship; kill if Tier-1 geometry predicts everything.

# Queued: EasyControl colorize — fix yellowish/sepia tone drift

# Queued: CDM Phase-0 full A/B — dynamic_schedule at 2k steps

`docs/proposal/cdm.md` — 500-step smoke (2026-07-17, valid arms verified via
`ss_turbo_dynamic_schedule`) passed all three gate axes: dynamic wins NFE=4
color/detail on all 3 styles, fixed-grid arm craters off-grid (NFE=3/2
washout), glyphs dynamic ≥ fixed. Now the real read: fresh 2k both arms
(no resume — final-step bundles skipped), ckpts 500/1000/2000, same
`configs/gui-methods/custom/turbo.toml` base; arm A needs a
`dynamic_schedule=false` copy (TOML-only knob, no CLI flag). ~4.5h/arm local
(dynamic ~25% cheaper). Render pass: fill dynamic @channel NFE3 + matched
artist NFE2 pairs. Gate to Phase 1 (L_CDM) on win.
