# tier_routing — Phase 3a gradient-equivalence probe

**Verdict (2026-07-23): FAILED — entropy-aware tier routing dies at the gate.**
Demotion has a real, reproducible gradient cost, but the cost is **not
content-modulated**: the conditional mean E[gap | redundancy] is flat. The
selective-routing thesis has no basis in gradient space; the item closes at
the price of the probe (~2.5 h GPU, no training run spent).

Design: `_archive/proposals/traj_latent_stats.md` §Phase 3a. Per image,
accumulated-gradient estimates over K (σ, ε) draws (stratified σ grid,
`flow_shift` 1.0) at a trained plain-LoRA operating point
(`anima_soup_sincos`); within-tier redraw floor as the null, re-encode arm as
the confound control, `gap = floor − cos(native, demote)`.

## Runs

Same 40-image probe set (redundancy 0.49–0.93, 20 artists), arms
{native_a, native_b, reenc, demote896, demote768}:

- **K=16** — `results/20260723-2203-phase3a/` (32 min)
- **K=32** — `results/20260723-2237-phase3a-k32/` (61 min)

Block-compile path (one dynamic-seq graph seq∈(2106, 4200), activation
budget 0.99 + aggressive_recomputation): fits 16 GB with no grad ckpt and ran
faster than the grad-ckpt smoke.

## What reproduces across the two independent runs

| quantity | K=16 | K=32 | verdict |
|---|---|---|---|
| gap_reenc mean (control) | +0.021 | +0.003 | ≈ 0 ✓ probe valid |
| gap_896 mean | +0.083 | +0.074 | **real cost, reproduced** |
| gap_768 mean | +0.147 | +0.151 | **dose-response, reproduced** |
| cos_floor (per-image) | — | cross-run ρ +0.56 | a real image property |
| gap_896 per-image ranking | — | split-half ρ **−0.18** | **unmeasurable noise** |
| gap_768 per-image ranking | — | split-half ρ +0.19 | unmeasurable noise |

Per-image gap noise did **not** shrink from K=16→32 (reenc-gap std 0.139 →
0.151): the accumulated-gradient estimator's variance is dominated by a few
high-magnitude σ-draws (heavy-tailed), so feasible K cannot resolve per-image
structure. Any per-image predictor correlation is bounded by
√(split-half reliability) ≈ 0 — the apparent K=16 "reliable-subset Spearman
−0.35 / 5× quartile contrast" (previous revision of this report) was noise
and is retracted.

## The kill: conditional mean is flat in redundancy

Bin means are immune to per-image noise attenuation (SEM ~0.015–0.02 at
n=10×2 runs). Pooled both runs, redundancy quartiles:

| | Q1 (0.49–0.69) | Q2 | Q3 | Q4 (0.80–0.93) |
|---|---|---|---|---|
| gap_896 | +0.070±.013 | +0.083±.021 | +0.079±.019 | +0.065±.015 |
| gap_768 | +0.168±.038 | +0.161±.026 | +0.118±.025 | +0.139±.029 |

Spearman(red → pooled gap_896) = −0.03; bootstrap P(top-half < bottom-half)
= 0.60. If redundancy modulated demotion cost by anything actionable
(≥0.05 between extreme quartiles) it would be visible here. It is not.
gap_768 shows a weak non-monotonic trend (−0.22) but 768 is ruled out by
absolute cost anyway.

This is the archived-autoscale picture (corpus-mean cross-resolution cosine
~uniform), now established with proper controls: the divergence is real,
roughly uniform across images, and latent redundancy does not predict it.
Static corpus redundancy (which is real and skewed) does **not** translate
into gradient-space demotion safety — plausibly because LoRA gradients are
dominated by what the *adapter* is learning, not by pixel-information
density.

## What survives

- **The numbers**: demote-one-tier costs ~0.074 gradient-cosine vs the
  redraw floor (896), ~0.147 at 768 — the honest reference for any future
  blanket-demotion wall-clock argument (which autoscale Phase 1 already
  killed at matched FLOPs; do not resurrect).
- **The instrument design**: redraw-floor null + re-encode confound control +
  cross-run split-half reliability check. The Run-1 experience is the
  cautionary tale: without the reliability check, the K=16 subset analysis
  read as a pass signal.
- **The harness**: `run_grad_probe.py` (per-image LoRA gradient comparison,
  compiled dynamic-seq fwd+bwd on 16 GB) and `redundancy.py` (per-image
  static-column scoring) are reusable for any future gradient-content
  question.

## Addendum: caption-interaction check (2026-07-23)

Post-verdict hypothesis: gradient content tracks supervision structure
(cross-attn learns labeled tags only), so a *caption-side* predictor might
succeed where latent redundancy failed. Tested on the same pooled data —
tag count (9–74) and caption length vs pooled gaps: Spearman ≈ 0 both
edges, tag-count-bin means flat (0.067–0.081 at 896), bootstrap at chance.
Cancellation between the axes is also excluded: ntags↔redundancy ρ = −0.40
(flat images carry fewer tags), yet the 2×2 (redundancy × tag density)
cell means are uniform (0.069–0.090). Demotion cost is invariant to both
content density and supervision density. Further sub-slice predictor
fishing on this probe set is noise-fishing (per-image reliability ≈ 0) —
any new predictor needs a fresh probe set and a pre-registered hypothesis.

## Addendum 2: "unlearnable ∧ redundant" two-factor check + disattenuation (2026-07-23)

Post-verdict hypothesis: route down images the model *can't learn anyway*
(operationalized as low `cos_floor` — the one per-image quantity that IS
reproducible, ρ +0.56 — i.e. gradient incoherent across redraws) AND
redundant. Result, on pooled data:

- The cell is rare: floor↔redundancy ρ = **+0.37** (redundant images have
  *more* coherent gradients), so low-floor∧high-red is 6/40 even in this
  extremes-oversampled set.
- The cell is not safer: disattenuated cross-tier alignment
  (`cos_demote / floor`, correcting noise attenuation) is 0.865 in-cell vs
  0.901 overall, and floor→true-alignment is **+0.34** — weakly-learned
  images are MORE fragile to demotion at the coherent-component level, not
  less (and the noise artifact biases this correlation the other way, so
  it's conservative). The residual "their contribution is negligible anyway"
  argument is untestable by cosines, would need a training A/B, and the
  router input (floor) requires per-image GPU probing with no cheap static
  proxy — impractical regardless.
- Useful refinement: disattenuation shows the **true coherent-component
  divergence is ~10 % at 896 and ~19 % at 768** (means 0.901 / 0.811) —
  about half the raw gap was estimator attenuation. Cost is real but
  smaller than the raw numbers read; uniformity (the kill) is unchanged.
  High-floor images are marginally the safest to demote (0.935), the
  opposite of the hypothesis.

## Not retried / out of scope

- Different operating points (other checkpoints), mean-direction estimators,
  per-σ-band gaps: possible instrument variants, but the burden was on the
  signal to appear and the bin-mean analysis is not attenuation-limited.
- Blanket (non-selective) demotion for wall-clock: that is autoscale, closed
  2026-06-28.
