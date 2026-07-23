# traj_stats — trajectory-resolved latent statistics

## Phase 1 — the anime-domain atlas

**Run**: `results/20260723-2100-phase1/` (`run_atlas.py --label phase1`).
Generation arm: 8 prompts (mignon prompt-set families detailed×4 / sparse×2 /
no_trigger×2) × 4 seeds, 1024², 28-step er_sde, CFG 4 → 32 traces. Inversion
arm: DirectEdit inversion (`invert(..., traj_recorder=...)`, guidance 1.0,
same 28-knot schedule) of 8 real cached-corpus images, one per artist,
1024-tier band. Static column: 512 corpus latents, same quantizer (k=4) +
normalization, no trajectory. All traces canonicalized to σ-descending knot
order; commit recomputed from stored codes in that shared order.

### (a) Commitment is front-loaded — the n=1 preview survives aggregation

Aggregate commit-CDF (mean [p25, p75] over 32 generation traces):

| σ | 0.96 | 0.86 | 0.80 | 0.72 | 0.625 | 0.50 | 0.333 |
|---|---|---|---|---|---|---|---|
| gen | 0.03 | 0.12 | 0.18 [.09,.19] | 0.24 | 0.34 | 0.47 [.40,.50] | 0.65 [.61,.68] |
| inv | 0.01 | 0.06 | 0.11 [.07,.16] | 0.18 | 0.30 | 0.48 [.44,.52] | 0.74 [.71,.75] |

Roughly half of all tokens have taken their final k=4 code by σ=0.5, ~two
thirds by σ=1/3, with a tight cross-seed/prompt spread — **not** uniform in σ,
so falsifier 1 does not fire and the late-step headroom for Phase 3
inference items is real. E(σ) confirms from the activity side: at τ=q95 the
active-token fraction falls 1.00 → 0.73 (σ=0.78) → 0.29 (σ=0.50) → 0.12
(σ=0.26). τ-sensitivity (q90/q99 band): the curve shifts level (E@σ=0.50 =
0.38/0.29/0.18) but not shape — every τ in the band shows the same
front-loaded decay. Per-family: sparse prompts commit *earlier* (0.28 by
σ=0.80 vs 0.14 for detailed / no_trigger — less text signal to integrate,
earlier lock); detailed and no_trigger are indistinguishable. Effective
guidance (`guide`) is even more front-loaded: post-combine ‖v_f − v_u‖
drops 2.14 → 0.44 → 0.26 over σ=1.0 → 0.80 → 0.33, consistent with the
cross-attn front-loading finding (`bench/crossattn_drive`, archived).

### (b) Channel usage is skewed ~4× and stable across arms

`cbits(c, σ_min)` (generation, mean over 32): high channels 13 (1.50),
4 (1.40), 11 (1.19), 15 (1.18), 5 (1.10) vs idle channels 8 (0.37),
14 (0.53), 6 (0.57), 10 (0.70) — a ~4× per-channel entropy range at k=4.
The ordering is essentially the corpus ordering (Spearman vs static column
0.94; inversion arm 0.98). The skew exists already at σ≈0.9 and the channel
*profile* is frozen from σ≈0.92 down (per-knot gen↔inv Pearson ≥ 0.89 below
σ=0.92) — channels don't take turns, the domain just uses a fixed subset
hard. σ-scheduled channel truncation therefore has a target, but a
compute-irrelevant one (16 latent channels are mixed into 1024-dim tokens at
patch embed; see proposal Phase 3 scoping).

### (c) Generation matches inversion — corpus statistics transfer

The structural-disagreement falsifier does not fire:

| cross-arm check | value |
|---|---|
| commit-CDF max gap (gen vs inv means) | 0.086 |
| E(q95) mean abs gap | 0.094 |
| cbits channel-profile Pearson, median over knots | 0.90 (≥0.89 for σ ≤ 0.92) |
| cbits channel Spearman at σ_min | 0.95 |
| static vs gen final cbits (Pearson / Spearman) | 0.95 / 0.94 |
| static vs inv final cbits (Pearson / Spearman) | 0.99 / 0.98 |

The only divergence is the σ→1 extreme (channel-profile r = 0.42/0.60 at the
two highest knots), which is expected and mechanical: at σ≈1 generation's
x̂₀ is the model's prior guess from pure noise while inversion's is the
nearly-destroyed image estimate. Below σ≈0.92 the two processes are
statistically the same object, so corpus statistics license
generation-side claims (not just img2img/editing ones). Inversion is
mildly *more* front-loaded at low σ (0.74 vs 0.65 at σ=1/3) — real images
lock detail slightly earlier than the sampler does.

Two side observations worth keeping: (1) real-image trajectories end at
higher `hf` than generated ones (0.20 vs 0.13 token-Laplacian energy) — the
generated corpus is measurably smoother than the training corpus under
identical normalization, a usable gauge baseline; (2) the static column's
within-image redundancy is substantial (modal joint-code share 0.11, unique
code fraction 0.26 at k=4) and skewed per image — the entropy-aware tier
routing input exists.

**Verdict**: exploitable structure confirmed on both axes the proposal
gated on (front-loaded commit-CDF, skewed cbits) and statistics transfer
from corpus to generation. Phase 2 (the intactness gauge) proceeds;
Phase 3's committed-token compute-reuse item keeps its audition, channel
truncation is demoted to measurement-only.

Repro:

    uv run python bench/traj_stats/run_atlas.py --label phase1

---

## Phase 0 — passive recorder: bit-exactness + overhead

**Verdict: PASS** (run `results/20260723-2042-phase0/`, 1024², 28 steps,
er_sde, CFG 4.0, seed 42; spectrum smoke arm included).

Implements Phase 0 of `docs/proposal/traj_latent_stats.md` (PR #74): the
`--traj_stats` recorder (`library/inference/traj_stats.py`), hooked into the
main inline loop, the tiled loop, and `spectrum_denoise` via
`SamplerSideChannels.traj_stats`. Invariant tests: `tests/test_traj_stats.py`.

| gate | result |
|---|---|
| determinism control (2× recorder-off, bit-identical) | PASS (main + spectrum) |
| bit-exactness (recorder on vs off) | PASS (main + spectrum, both recorder runs) |
| overhead ≤ 2 % at 1024² | PASS — **0.12 %** main (0.56 ms/step), 0.31 % spectrum |

Sidecar: 2.4 MB npz (28 steps × 4096 tokens × 6 traces, k=4).

## Performance notes (the two traps)

1. **Never pass `float(sigmas[i])` to `record()`** — the float() read is a
   stream sync that kills the loop's CPU run-ahead (~28 ms/step, 5.8 %
   overhead). The baseline loop never syncs per step (its `float(sigmas[i])`
   reads all sit behind short-circuited `None` checks). Hook sites pass the
   0-d tensor; conversion happens once at flush. Isolated `record()` cost is
   0.39 ms/step.
2. **`np.savez`, not `savez_compressed`** — flush runs inside the generation
   wall time; zlib on the ~4 MB payload cost >100 ms per generation.

## First trace (Phase 1 preview, single prompt/seed — not yet a claim)

- E(σ) (τ = σ→0 activity floor, provisional 95th-pct): 1.00 at σ≈0.95 →
  0.85 at σ=0.80 → 0.49 at σ=0.69 → 0.15 at σ=0.33 → 0.05 at the end.
- commit-CDF: 10 % of tokens committed by σ=0.80, 32 % by σ=0.55, 56 % by
  σ=0.33 — i.e. commitment is meaningfully front-loaded on this render,
  which is the exploitable-structure direction. Phase 1 (seed × prompt grid
  + inversion arm) decides whether it holds corpus-wide.

Repro:

    uv run python bench/traj_stats/run_bench.py --with_spectrum --label phase0
