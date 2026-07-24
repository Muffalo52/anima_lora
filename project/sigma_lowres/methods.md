# sigma_lowres — implementation

What exists in code for the σ-conditional low-res gradient line. Everything is
**observability instrumentation** — no trainer wiring has been built (Phase 1b
is design-only; see `roadmap.md`).

The line's benches are adopted into this home (`bench/`), including the two
predecessor instruments it builds on: `bench/tier_routing/` (the closed
Phase-3a gradient-equivalence probe — `redundancy.py` still supplies probe-set
selection to both measurements below) and `bench/autotune/` (the
`blocks_to_swap` × `activation_memory_budget` → peak-VRAM/step-time surface
sweep behind the partitioner-budget operating point the gradient probe pins).

## Measurement A — latent RAPSD + closed-form crossover

`project/sigma_lowres/bench/rapsd.py` (CPU-scale, no DiT).

- Computes the radially-averaged power spectral density of the probe set's
  cached VAE latents in the DiT's spatial grid, normalized frequency
  r ∈ (0, 0.5] cycles/latent-pixel.
- Under flow-matching noising `(1−σ)x₀ + σε` with unit-variance white noise
  (PSD ≡ 1), the per-frequency signal/noise crossover has the closed form
  **σ_eq(f) = √P(f) / (1 + √P(f))**.
- Outputs: mean P(f) curve, σ_eq(f), predicted σ\*(e) for each demote edge
  (demoted Nyquist = 0.5·e/1024), above-Nyquist SNR A(σ, e), Fig-1-style plot.
- Reusable for any "what does the spectrum predict" question on Anima/Qwen-VAE
  latents.

## Measurement B — per-σ-bin gradient probe

`project/sigma_lowres/bench/run_sigma_probe.py` (~2–2.6 h GPU for 40 images × 6 arms).

The tier_routing Phase-3a instrument extended with per-σ-bin gradient
accumulators — the estimator class that was *reliable* in 3a (per-bin means
across images, SEM ~0.02), not the per-image ranking that failed there.

- **Arms per image**: native, redraw-floor null (same res, fresh noise draws —
  the "how much do gradients differ anyway" floor), re-encode control
  (decode→re-encode at native res — isolates VAE round-trip cost), and demote
  arms (pixel-space downscale → VAE re-encode → noise; SwD's validated
  "strategy B" ordering — never latent-space downsampling).
- **Binning**: B uniform σ bins × D stratified draws per bin per arm
  (shipped runs: 8 × 8). Uniform bins make the training marginal density
  irrelevant; density-weighting the bins by the trainer's sigmoid σ-density
  reproduces 3a's pooled numbers (consistency check, passed).
- **Per image × bin outputs**: cos_floor, cos_reenc, cos_e, gap_e =
  floor − cos_e, grad norms. Verdicts read off bin-mean curves with a
  mandatory split-half reliability check.
- **CLI**: `--tier <native_edge> --demote_edges <e1,e2,...>` selects the
  operating point (Phase 0 ran 1024→{896,768,512}; Phase 1a ran
  896→{768,512}).
- **Daemon-hardened**: `start_heartbeat()` (45 s stderr ticks) keeps the
  120 s daemon stall-watchdog from killing long silent accumulation loops
  (`project_sigma_lowres_phase0` gotcha).

## Operating point / caveats baked into the instrument

- Adapter under probe: `anima_soup_sincos` (trained at native tiers). A
  mixed-res-trained adapter might equalize its own gradients — untested.
- Per-bin cosines use 8 draws → absolute cosines are not comparable across
  bins (floor drops where ‖g‖ is small); **gap subtraction** is the valid
  read, with gap_reenc ≈ 0 as the instrument-validity witness.

## What deliberately does NOT exist

- Trainer wiring (σ-drawn-at-batch-assembly cache switch). Design reference
  if ever built: the autoscale-emit stem-suffixed sibling-cache pattern —
  its runtime was stripped 2026-06-28; do not resurrect blindly.
- Any latent-space downscale path (SwD found it inferior; untested here).
