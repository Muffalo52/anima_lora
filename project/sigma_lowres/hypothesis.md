# sigma_lowres — mechanism hypothesis: the two-term account

Status: pre-registered 2026-07-24 (predictions committed before the runs
finished); **Tests 1–2 landed the same day: the account HOLDS, and the Floor
is graph-dominated** (row 1 of the outcome matrix). Written to answer "can we
explain Phase 0 in principle?" (Q2/Q6 context).

## Claim

The demotion gap measured by the σ-probe decomposes into two terms:

```
gap_e(σ)  ≈  S1_e(σ)  +  Floor_e
```

- **S1 (input branch)** — the only term the SwD-style spectral argument
  describes. Decays smoothly in σ (Wiener-like posterior shrinkage), with **no
  hard gate** at the spectral crossover; it reads as "collapsed" wherever it
  sinks below the instrument's reenc band (±0.04) — measured at σ ≈ 0.5, not
  the RAPSD σ\* ≈ 0.14. The 3.5× discrepancy is expected, not anomalous.
- **Floor (target × graph)** — σ-independent, grows with demotion severity,
  and **structurally exempt from noise masking**, for two reasons:

  1. **Expected gradients are never noise-masked.** "Noise power exceeds
     signal power in band f" is a statement about a *single sample* of z_σ.
     The trained quantity is E_ε[∇θL] — an expectation that averages noise
     out and retains the signal at *any* SNR, merely attenuated. Sufficiency
     of demotion for the input licenses inference equivalence, never
     training-dynamics equivalence: mutual information is
     representation-independent, gradients are parameterization-covariant.
  2. **The target carries the clean image at unit amplitude at every σ.**
     The FM target is v = ε − x; the input gets noise-masked as σ→1 but x
     sits in the target at coefficient 1 even at σ = 0.94. At high σ the
     prediction collapses toward the prior, so the residual r ≈ x − x̄_prior
     — *the high-σ residual is the image*. The gradient g = Jᵀr then differs
     across grids through both factors: r (per-token content density changes;
     the base model's prior error is resolution-conditioned) and J (token
     count changes the graph itself: attention softmax over N, RoPE phase
     density, seq-dependent normalization — exactly Q2's candidate list).

Phase-0/1a numbers this account retrodicts: 896 plateau 0.03–0.05 (inside
instrument noise → "safe"), 768 ≈ 0.06–0.12 (just outside → 1a FAIL), 512
≈ 0.3 at σ = 0.94 where the latent is 97% noise (Floor alone). Corollary:
**1024→896 safety is an empirical smoothness statement about Anima's function
across nearby token counts, not an information statement** — there was never
a reason to expect a universal ratio invariant (consistent with 1a), and Q1's
absolute-capacity governor is the natural reading (Floor = how well the
coarse graph approximates the fine graph's computation).

The **latent-space quirk** (HF-quiet Qwen-VAE latents, non-scale-equivariant
encoder) shapes S1 only — it is exonerated as the Floor's cause by two
existing controls: gap_reenc ≈ 0 (encoder round-trip harmless) and the
σ = 0.94 persistence (input statistics there are nearly identical Gaussians).

## Pre-registered tests

### Test 1 — x-zero probe (isolates the J-term)

`run_sigma_probe.py --x_zero`: image zeroed in BOTH input and target on every
grid (input = σε, target = ε; captions + exact demoted latent shapes kept).
No content exists anywhere → any surviving gap is **pure graph-shape
sensitivity**. Run: 40 images, 4 σ-bins + σ=1 endpoint, edges 896/768/512,
8 draws/bin.

- **Read**: xz_gap ≈ endpoint gap → Floor is graph-dominated (S3) → Q2's
  per-module split is the right next probe (RoPE/attention localization).
  xz_gap ≪ endpoint gap → Floor is content-correspondence in the residual
  (S2) → fix space is content-side (e.g. resolution-conditioned targets),
  not architecture-side.
- **Secondary prediction**: the xz curve is ~flat in σ (no content to fade).
  Caveat: low-σ bins are off-manifold (input σε is norm-shrunk), so the
  σ=1 / high-σ read is primary.
- **Anomaly bar**: xz_gap substantially *above* the endpoint gap at matched
  edge would not fit the account and reopens it.

**RESULT (2026-07-24, `results/20260724-2136-xzero/`): graph-dominated.**
σ centers 0.125/0.375/0.625/0.875/1.0; SEM in parentheses at the endpoint:

| edge | xz gap across σ | xz @ σ=1 | endpoint (T2) @ σ=1 |
|---|---|---|---|
| 896 | .012 .039 .014 .031 .004 | 0.004 (.018) | −0.009 (.042) |
| 768 | .062 .046 .033 .117 .064 | 0.064 (.028) | 0.127 (.054) |
| 512 | .188 .135 .158 .260 .299 | 0.299 (.040) | 0.326 (.059) |

- **512: xz ≈ endpoint** (0.30 vs 0.33) — the Floor is essentially all
  graph/function term; per-image data content contributes ~nothing.
- **768: xz ≈ half the endpoint gap** (0.06 vs 0.13, SEM-overlapping) —
  graph term is the bulk, a possible minority content-correspondence share.
- **896: xz ≈ 0 everywhere**, matching endpoint ≈ 0.
- **Flatness**: no S1-like decline anywhere (512 mildly *rises* with σ;
  low-σ bins are the off-manifold regime, high-σ read primary). Anomaly bar
  (xz ≫ endpoint) not triggered. Split-half 0.74–0.88 (768's noisier 0.28
  driven by the wide 0.875 bin; endpoint bin is the verdict bin).
- Bonus decomposition check against Phase 0 at LOW σ: xz(512) ≈ 0.13–0.19 vs
  standard 0.35–0.47, xz(896) ≈ 0.01–0.04 vs standard 0.11–0.16 — i.e. the
  low-σ elevation in Phase 0 is mostly S1, sitting on top of exactly this
  Floor, as the two-term account requires.
- Interpretation caveat (sharpened, not weakened): with x = 0 the residual is
  ≈ −x̂_prior — the model's own grid-conditioned prior (xz ‖g‖ at σ=1 is 39.9,
  large despite zero content). So "graph-dominated" means Jᵀx̂_prior mismatch
  — the network function across token counts, including its
  resolution-conditioned prior — NOT per-image data content. This is exactly
  the object Q2's per-module split decomposes.

### Test 2 — σ=1 endpoint bin (measures Floor by construction)

`run_sigma_probe.py --bins 0 --endpoint_bin`: at σ = 1 the input is exactly ε
— the input-information term is zero by construction; any measured gap IS the
Floor. Run: standard arms (native ×2, reenc, 896/768/512), 16 draws.

- **Prediction**: gap(σ=1) matches the Phase-0 high-σ plateau per edge —
  roughly 896 ∈ [0, 0.08], 768 ∈ [0.04, 0.15], 512 ∈ [0.2, 0.4], tier
  ordering preserved, gap_reenc within ±0.04.
- **Falsifier**: endpoint gaps ≈ 0 for all edges. That would mean the σ=0.94
  persistence was carried by the residual (1−σ) input signal after all — the
  two-term account dies and a pure-input story revives. (This is the
  account's cleanest kill switch.)

**RESULT (2026-07-24, `results/20260724-2101-endpoint/`): prediction PASSES
on all three edges; falsifier not triggered.** 40 images, 16 draws, σ=1.0
exactly; cos_floor 0.86, ‖g‖ 63.9 (the σ→1 tail, as expected).

| edge | gap @ σ=1 (SEM) | predicted band | verdict |
|---|---|---|---|
| reenc | −0.038 (.032) | within ±0.04 | instrument valid |
| 896 | −0.009 (.042) | [0, 0.08] | ✓ ≈ 0 |
| 768 | +0.127 (.054) | [0.04, 0.15] | ✓ |
| 512 | +0.326 (.059) | [0.2, 0.4] | ✓ |

Tier ordering preserved. At σ=1 the input contains **zero image information
by construction**, yet the 768/512 gaps match the Phase-0 high-σ plateau —
the Floor is real and the input branch cannot be its source. The two-term
account survives its kill switch.

### Test 3 — content-loss correlation on existing data (RUN — inconclusive)

From Phase-0 `per_image.jsonl`: correlate per-image high-σ gap with content
lost to demotion (latent down-up error; HF energy above demoted Nyquist).
S2 predicts positive correlation; S3-only predicts none.

**RESULT (2026-07-24): cannot be measured at 8 draws.** The reliability
ceiling of the per-image high-σ gap — agreement between the two top σ-bins
across the 40 images — is *negative* (r ≈ −0.09..−0.18 for all edges), i.e.
the per-image gap is estimator noise with no stable image-level component
(the same per-image-ranking failure as tier_routing 3a). All correlations
null (|ρ| ≤ 0.23, p > 0.15) **against a ~0 ceiling** — reads as "instrument
blind", not "no effect". The S2-vs-S3 question falls entirely to Test 1.

## What the outcomes mean downstream

| Endpoint (T2) | x-zero (T1) | Verdict |
|---|---|---|
| ≈ plateau | ≈ endpoint | **← MEASURED OUTCOME.** Two-term account holds, **graph-dominated** → Q2 per-module split (J-decomposition) is the mechanism probe; Q1 capacity governor favored |
| ≈ plateau | ≪ endpoint | Two-term account holds, **content-dominated** → mechanism is resolution-conditioned prior error; per-module split less informative, content-side interventions open |
| ≈ 0 | (any) | **Account falsified** — input branch explains everything; spectral story revives with a slower decay constant |
| ≈ plateau | ≫ endpoint | Anomaly — account incomplete, reopen |

Paper framing if the account survives (Q6): not just "a measured
counterexample to spectral sufficiency" but *why it must be one* — the target
branch of the FM objective is structurally exempt from noise masking.

**Follow-up executed (2026-07-25)**: the Q2 per-module decomposition ran on
the indicated (x-zero) arms — see report.md "Phase Q2". Outcome: Floor
localizes in depth (early blocks ~3×, all module types uniformly; content
share late-block), RoPE refuted as a concentrated mechanism (up_q/up_k ≈
up_v). The account's J-mismatch is an early-block representation property,
not a parameter circuit.
