# traj_stats Phase 0 — passive recorder: bit-exactness + overhead

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
