# sigma_lowres — open questions

## Q1 — Ratio or absolute capacity: what governs demotion safety?

1024→896 (ratio 0.875) passes; 896→768 (ratio 0.857) fails. Two candidate
governors that the existing data cannot separate:

- **Ratio**: safety is a function of edge ratio → 1280→1024 (0.80) should FAIL.
- **Absolute target capacity**: safety needs enough tokens at the demoted grid
  (~4k?) → 1280→1024 (4116 tok target vs the passing 3012) should PASS.

The **1280→1024 probe is the discriminating experiment** — and it carries the
largest per-draw payoff (0.65× tokens) if capacity wins. Answering it turns
the finding into a safety-boundary map over (route, σ) — the stronger paper
shape. Blocked on a small 1280-tier re-preprocess + a 6300-token VRAM check.

## Q2 — Where in the network does the gap live? **[ANSWERED 2026-07-25]**

Phase 0 established the gap is a network-function property (persists when the
latent is ~pure noise), but not *which* mechanism: attention structure, RoPE
geometry, seq-length-dependent normalization? A per-block / per-param-group
gap decomposition (same probe, gradient split by module) would localize it —
and might reveal a subset of parameters for which demotion IS safe.

**Answer** (`hypothesis.md` + report.md Phase Q2; runs `*-endpoint-pg` /
`*-xzero-pg`): the Floor localizes in **depth, not module type** — early
blocks (~0–9, peak 3–8) carry ~3× the late-block gap, uniformly across every
param type within a block; RoPE is refuted as a concentrated mechanism
(self-attn up_q/up_k show zero excess over up_v). The content share is a
late-block minority effect; early-block sensitivity is pure graph. The "safe
subset" is a **depth band**: late-half-only updates at 768 read gap
0.03–0.09 vs 0.12 full — a lever, not yet a pass. Remaining mechanism
question (why blocks 3–8 specifically) belongs to the paper phase (Q6), not
to safety mapping.

## Q3 — Does mixed-res training equalize its own gradients?

All measurements are at a native-res-trained operating point
(`anima_soup_sincos`). An adapter trained on mixed-res batches might close
(or widen) the gap. Any reopen of the broader low-res family should probe a
mixed-res-trained checkpoint **first** — it bounds whether the Phase-0 map is
a property of the base model or of the adapter's training distribution.

## Q4 — Is the 896 high-σ residual (~0.03) actually harmless at training scale?

"Within the reenc band at N=40" is a gradient-level statement; a full run
integrates thousands of demoted steps. Only the Phase-1b fixed-steps CMMD
non-inferiority A/B answers this. If CMMD regresses, the residual gap is the
suspect — the line closes (proposal's pre-commitment).

## Q5 — Do the bespoke loops inherit anything?

- **EasyControl**: structurally clean (frozen DiT, cached paired latents) but
  the gradient lives in the cond-LoRA stream — equivalence does not transfer
  automatically. Needs the probe re-run at an EC operating point with the
  cond stream driven.
- **turbo**: NOT clean — rollout latents are generated (no pixel-space demote
  path), and changing rollout resolution changes what the student *is* (that
  is SwD's scale-wise pipeline, a different product). Only fake/critic σ-draw
  forwards are even candidates. Own research question; no savings promised.

## Q6 — Paper question

Is "spectral sufficiency ≠ gradient equivalence" + the (route, σ) safety map
enough for a workshop paper? Needs Q1 answered (the map needs ≥3 routes) and
ideally Q2 (a mechanism sketch, not just a refutation).
