# sigma_lowres — Phase 0: σ-conditional low-res gradient equivalence

**Verdict (2026-07-24): spectral mechanism REFUTED as the governor; σ-dependence
real but the payoff collapses.** The demotion gap is genuinely σ-dependent in
the predicted direction (H2 qualitative: pass), and tier-ordered at every bin
(H3 ordering: pass) — but the collapse sits at **σ ≈ 0.5, not the
RAPSD-predicted σ\* ≈ 0.14** (H3 quantitative: **fail**, off by ~3.5×), and
**512 never becomes safe at any σ** (gap 0.29–0.47 in every bin, including
σ=0.94 where the latent is ~pure noise). The SwD noise-masking argument does
not govern LoRA gradients on Anima.

Design: `project/sigma_lowres/initial_proposal.md` (criteria frozen before
data). Runs: RAPSD `results/20260724-1202-phase0/`, gradient probe
`results/20260724-1237-phase0/` (40 images, 6 arms, 8 uniform σ-bins × 8
draws, ~2.6 h; plot `gap_vs_sigma.png`).

## Measurement A — RAPSD (the prediction)

Anima/Qwen-VAE latents are high-frequency-quiet (latent var 0.42; P(f) < 1
above f ≈ 0.16). Closed-form crossover σ_eq(f) = √P/(1+√P) predicted
σ\*(896) = 0.136, σ\*(768) = 0.146, σ\*(512) ≈ 0.20, per-image spread tight
(0.11–0.16) — crossover image-generic.

## Measurement B — per-σ-bin gradient gaps (the test)

Bin centers 0.0625 … 0.9375; mean over 40 images (SEM ~0.02); split-half
reliability of the bin-mean curves 0.73–0.83 (reliable), gap_reenc ≈ 0
everywhere (|mean| ≤ 0.054 — instrument valid).

| σ bin | 0.06 | 0.19 | 0.31 | 0.44 | 0.56 | 0.69 | 0.81 | 0.94 |
|---|---|---|---|---|---|---|---|---|
| gap_896 | .110 | .162 | .148 | .137 | **.048** | **.030** | **.030** | .053 |
| gap_768 | .144 | .216 | .208 | .223 | .164 | .163 | .115 | .063 |
| gap_512 | .348 | .410 | .355 | .469 | .430 | .391 | .289 | .296 |
| cos_floor | .84 | .65 | .51 | .65 | .70 | .77 | .80 | .83 |
| ‖g‖ native | 2.6 | 0.5 | 0.4 | 0.5 | 0.7 | 1.1 | 2.0 | 9.3 |

- **896**: elevated (~0.14–0.16) through σ ≈ 0.44, then drops to 0.03–0.05 for
  σ ≥ 0.5 — within the reenc-control band, i.e. demotion there costs no more
  than re-encoding. Crossover ≈ 0.5.
- **768**: never floors below σ ≈ 0.8; 0.16 even at σ = 0.69.
- **512**: large everywhere. No safe σ exists.
- Spearman(σ → gap): −0.69 / −0.57 / −0.36 — monotone-ish decline, real.
- Consistency with 3a: density-weighting the bins by the trainer's sigmoid
  σ-density reproduces the pooled gaps (0.092 / 0.180 vs 3a's 0.074–0.083 /
  0.147–0.151) — same instrument, σ now resolved.

## Why the spectral story fails

The gap persists far above the spectral crossover, most starkly for 512 at
σ = 0.94: the noisy latent there is ~97% noise by power in every band the 512
grid can't represent, yet the gradient still diverges by 0.3. The
resolution-sensitivity is therefore **a property of the network function, not
of the latent's information content** — different token counts change
attention structure, RoPE geometry, and the seq-length-dependent behavior the
adapter's gradients live in. Noise-masking arguments (SwD Fig 1, pyramid-flow's
premise) justify *representability*, not *gradient equivalence*, and the two
come apart at exactly the 2× downsampling that would have paid.

Grad-norm structure confirms the 3a heavy-tail diagnosis: ‖g‖ is dominated by
the σ→1 tail (9.3 vs 0.4 mid-σ), so 3a's pooled cosines were mostly measuring
the high-σ bins — where the 896 gap happens to be small — explaining why
pooled 896 read "cheap" while mid-σ demotion is actually 3–5× worse.

## What survives / practical residue

- **σ>0.5 → 896 routing** is defensible (gap ≈ reenc control): epoch cost
  ≈ 0.5·0.72 + 0.5 = **0.86 → ~14% wall-clock ceiling**. Far below the ~27–45%
  that motivated Phase 1; likely not worth the dual-cache + batch-assembly
  complexity. Decision deferred; do NOT build without a fixed-steps
  CMMD-non-inferiority A/B.
- **The instrument**: per-σ-bin binned variant of the 3a probe
  (`run_sigma_probe.py`, heartbeat-hardened for the daemon stall watchdog) +
  `rapsd.py` (latent RAPSD / σ_eq closed form) are reusable for any future
  σ-resolved gradient question.
- **The finding**: "spectral sufficiency ≠ gradient equivalence" is the
  paper-relevant residue — a measured counterexample to the assumption behind
  scale-wise training schemes, at fine-tuning granularity.

## Phase 1a addendum (2026-07-24): ratio transfer FAILS

`results/20260724-1523-phase1a-t896/` — same instrument on 40 native-896
images, arms 768 (ratio 0.857) + 512 (0.57, two-tiers-down control).
Pre-registered bar: gap_768 within the reenc band at σ ≥ 0.5.

| σ bin | 0.06 | 0.19 | 0.31 | 0.44 | 0.56 | 0.69 | 0.81 | 0.94 |
|---|---|---|---|---|---|---|---|---|
| 896→768 | .114 | .209 | .168 | .143 | **.124** | **.056** | **.092** | **.061** |
| 896→512 | .318 | .372 | .308 | .340 | .320 | .280 | .329 | .217 |

- **FAIL**: high-σ residual 0.06–0.12 ≈ 2× the 1024→896 plateau, outside the
  reenc band (±0.04). The σ-decline is still real (Spearman −0.69) but the
  route is not safe by the frozen criterion.
- Control as predicted: 896→512 elevated flat everywhere.
- **"One tier down" is NOT the invariant.** Safety degrades sharply between
  ratio 0.875 (1024→896, passes) and 0.857 (896→768, fails) — or the governor
  is absolute target capacity, not ratio. The two hypotheses **disagree on
  1280→1024** (ratio 0.80 — more aggressive than the failing 0.857, but
  target capacity 4116 tokens — higher than the passing 3012): ratio says
  fail, capacity says pass. That probe (needs a 1280-tier re-preprocess +
  6300-token VRAM check) is the discriminating experiment, and turns the
  finding into a safety-boundary map over (route, σ) — the stronger paper
  shape.
- Practical residue narrows to the single measured-safe route:
  **1024→896 at σ > 0.5**, covering 96% of the corpus (2901/3008 records) →
  ~13–14% wall-clock at fixed steps. EC/turbo extensions inherit this route
  only, pending their own operating-point probes.

## Caveats

- Single operating point (`anima_soup_sincos`, trained at native tiers). An
  adapter trained mixed-res might equalize its own gradients — untested; any
  reopen should probe a mixed-res-trained operating point first.
- Uniform σ bins; per-bin cosines use 8 draws (floor correspondingly lower at
  mid-σ where ‖g‖ is small — gap subtraction controls this, and reenc stays
  ≈ 0, but absolute cosines across bins are not comparable).
- 896-at-high-σ "safe" = within reenc band at N=40; it is a small residual
  (~0.03, ~2 SEM > 0), not exact zero.
