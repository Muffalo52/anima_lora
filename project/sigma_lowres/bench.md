# sigma_lowres — bench digest

Canonical: `bench/sigma_lowres/report.md` (full tables + caveats), design
frozen pre-data in `docs/proposal/lowres_sigma_equivalence.md`. Runs under
`bench/sigma_lowres/results/`.

## Phase 0 (2026-07-24) — spectral mechanism REFUTED

Runs: RAPSD `20260724-1202-phase0`, gradient probe `20260724-1237-phase0`
(40 images @ native 1024, demote arms 896/768/512, 8 σ-bins × 8 draws).

- **Prediction (RAPSD)**: latents high-frequency-quiet (P(f) < 1 above
  f ≈ 0.16); predicted crossovers σ\*(896) = 0.136, σ\*(768) = 0.146,
  σ\*(512) ≈ 0.20; per-image spread tight → crossover image-generic.
- **Measurement (gradients)**: σ-dependence is real and tier-ordered
  (H2 qualitative pass, H3 ordering pass, Spearman σ→gap −0.69/−0.57/−0.36),
  **but** the collapse sits at **σ ≈ 0.5, not ≈ 0.14** (H3 quantitative fail,
  ~3.5× off) and **512 is never safe at any σ** — gap 0.29–0.47 even at
  σ = 0.94 where the latent is ~97% noise in every band 512 can't represent.
- **Interpretation**: resolution sensitivity is a property of the **network
  function** (attention structure, RoPE geometry, seq-length-dependent
  behavior), not of the latent's information content. Spectral sufficiency ≠
  gradient equivalence — a measured counterexample to the premise of
  scale-wise training schemes, at fine-tuning granularity.
- Instrument valid: gap_reenc ≈ 0 every bin (|mean| ≤ 0.054); split-half
  reliability of bin-mean curves 0.73–0.83.
- Side-finding: ‖g‖ dominated by the σ→1 tail (9.3 vs 0.4 mid-σ) — explains
  why 3a's pooled cosines read 896 as "cheap" (they mostly measured high-σ
  bins where the 896 gap happens to be small).

## Phase 1a (2026-07-24) — ratio transfer FAILS

Run: `20260724-1523-phase1a-t896` (40 native-896 images, arms 768 + 512).

- Pre-registered bar (gap_768 within reenc band ±0.04 at σ ≥ 0.5): **FAIL** —
  high-σ residual 0.06–0.12 ≈ 2× the 1024→896 plateau. Control (896→512)
  elevated flat everywhere, as predicted.
- **"One tier down" is NOT the invariant.** Safety degrades between ratio
  0.875 (1024→896, passes) and 0.857 (896→768, fails) — or the governor is
  absolute target capacity, not ratio. The two hypotheses disagree on
  **1280→1024** (ratio 0.80 — worse than the failing 0.857; capacity 4116 tok
  — better than the passing 3012): ratio says fail, capacity says pass.
  That is the discriminating probe.

## Standing result

Safe set = **{1024→896 at σ > 0.5}** only — gap 0.03–0.05, within the
reenc-control band (small residual, ~2 SEM > 0, not exact zero). Covers 96%
of the corpus (2901/3008 records) → **~13–14% wall-clock ceiling at fixed
steps**. Below the ~27–45% that motivated Phase 1; wiring decision deferred
pending a fixed-steps CMMD non-inferiority A/B (see `roadmap.md`).
