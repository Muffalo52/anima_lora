# Memorization-aware training — per-sample low-σ Δ-gap reweight

Status: **PROPOSED 2026-07-06 (Phase 0 not run).**

The first *intervention* on the memorization thread. Everything so far measured
it (`bench/memorization/`) or hoped something else would fix it (soup explicitly
does **not** — `bench/memorization/report.md`). This trains against it.

## Premise sources

- `bench/memorization/loss_gap.py` (weight-side MIA): the memorization signal is
  a per-sample **loss gap** — adapted loss vs base loss on the same
  `(x_t, ε, σ)` — and it **lives at σ ≤ 0.7**. Known-overfit reference config:
  sincos-half @ 8 epochs → AUC 0.82 (plain) / 0.77 (t1-mask); clean reference:
  plain_tenth → AUC 0.54.
- `bench/memorization/probe.py`: PE-Spatial xerox check, the independent
  OVERFIT/GENERALIZING verdict.
- In-loop `set_multiplier(0)` precedent: the VR control variate
  (`RuntimeState.vr`, `train.py::get_noise_pred_and_target`) already runs the
  trainable DiT with the adapter zeroed as a "frozen reference" — same trick,
  same code path.
- Extra-forwards-under-block-swap hazard:
  `[[project_blockswap_extra_forwards_gradcache]]` — the measurement forward is
  a second DiT forward per step, so this feature must hard-require
  `blocks_to_swap=0` (raise, not warn — same policy as the 2026-07-06
  register-tokens guard).

## The idea

Track, per dataset item, an EMA of the **Δ-gap** — exactly the quantity the MIA
probe scores, estimated online:

- On measurement steps (every K steps, or probability p per batch — knob), for
  batch items whose drawn σ ≤ 0.7: one extra no-grad forward at
  `set_multiplier(0)` on the *same* `(x_t, ε, σ)`;
  `Δ = loss_base − loss_adapted`. EMA it per item key.
- Pairing on identical `(x_t, ε, σ)` is what makes this principled: absolute low
  loss just means an easy image (flat backgrounds), but a large *gap vs base*
  means the adapter specifically learned **this** sample. The base forward
  cancels difficulty.
- Items whose Δ-EMA z-scores above threshold against the dataset distribution
  are flagged → intervention arms:
  - **Arm A (primary)**: downweight the flagged item's loss when σ ≤ 0.7,
    `w = f(z)` smoothly ↓. High-σ draws untouched — the item keeps teaching
    composition/style, it stops being pixel-copied. Local to the loss composer
    (`library/training/losses.py` registry).
  - **Arm B (fallback)**: per-item σ-floor — resample flagged items' σ draws
    away from the low band entirely. Blunter; only if A's soft weighting is too
    weak.

Cost: one extra no-grad forward on measurement steps only (K=8–16 keeps it a
few % overhead); state is a dict keyed like the caches, saved into the run dir
for post-hoc inspection.

## Phase 0 — reproduce the known overfit, ± reweight

Retrain the sincos-half @ 8ep config (the banked AUC-0.82 overfit) with the
reweight ON, versus the already-banked OFF run. All instruments **already
exist** — nothing new is built to judge this:

- **G1**: `loss_gap.py` AUC 0.82 → **≤ 0.65** (plain probe).
- **G2**: `probe.py` verdict moves OVERFIT → GENERALIZING (or the xerox matches
  visibly drop).
- **G3 (no-harm)**: standard 28-step/CFG-4 render grid, matched seeds, eyeball —
  the flagged items' artist must not lose fine detail (low σ *is* the detail
  band; this is the real risk of Arm A).
- **Sanity**: the online Δ-EMA ranking at end of training should correlate with
  the offline probe's per-item gap (they estimate the same thing; if they
  disagree, the online estimator is broken — fix before reading G1/G2).

## Phase 1 (only if Phase 0 gates pass)

Defaults + opt-in surface: `memorization_reweight = true` per-subset knob,
threshold/K/floor knobs in `configs/base.toml`, docs page, and the flagged-item
report dumped next to the checkpoint (useful on its own as a "which of my images
is this LoRA xeroxing" report, even with the reweight off — a measure-only mode
ships first).

## Non-goals / guards

- **Not an eval proposal.** Instruments are frozen at the existing bench; no new
  metric is introduced anywhere in this line.
- Not a T-LoRA interaction (that lever is rank-side; composable, test later).
- Doesn't touch the bespoke distill loops (turbo/rsd) — LoRA-family `train.py`
  only.

## Kill criteria

- G1 unmoved with the sanity check passing → memorization is not
  per-step-gradient-local at low σ (it accretes some other way); bank that,
  close the line.
- G3 fails at any threshold that moves G1 → the detail/memorization trade-off is
  inherent at the loss level; close Arm A, note Arm B's result, and the line's
  successor would be data-side (dedup/augment), not loss-side.
- Flag set unstable epoch-to-epoch (churn > ~50%) → the online estimator is too
  noisy at trainable batch sizes; measure-only mode may still ship.
