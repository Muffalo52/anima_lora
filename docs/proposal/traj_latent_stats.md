# Trajectory-resolved latent statistics — effective token usage measured *during* generation

Status: **Phase 0 SHIPPED & PASSED 2026-07-23** (recorder + invariant tests +
bench — `bench/traj_stats/results/20260723-2042-phase0/`, report in
`bench/traj_stats/report.md`). **Phase 1 (atlas) is the active phase.**
Phase 2 (intactness gauge) is the actual product; Phase 3 (interventions) is
explicitly speculative and gated on Phase 1/2 evidence.

Originally proposed in PR #74; this copy supersedes the draft there.

## Premise

The Anima DiT works in the Qwen Image VAE latent space (`z_dim = 16`,
per-channel `latents_mean` / `latents_std` shipped in the model config). The
anime domain is latent-sparse: flat cel shading, line art, and uniform
backgrounds mean most spatial tokens of a *finished* image carry near-zero
unique information under quantization. Static corpus statistics (quantize the
cached `.npz` latents, measure per-token / per-channel entropy) would confirm
that — but they are **reconstruction** statistics of clean images. They say
nothing about *when* along the σ trajectory each token's information
materializes, and the one intervention this repo already tried on the strength
of redundancy signals — the deferred-foveated merge — died precisely because
its damage was invisible to final-result framing until benched hard:

> the periphery soft blur is **constitutive** (P4t: no tail treatment recovers
> it) — `docs/inference/foveated.md`, archived 2026-07-03.

A region denoised at reduced token count never receives its high-frequency
detail. The failure lives in the **process**, not the endpoint: final-image
metrics on the fovea looked fine, the periphery *trace* was what flatlined.

This proposal inverts the order of operations: build the measurement layer
before proposing any new efficiency intervention.

1. **Measure statistics while generation runs** — per-step, per-token,
   per-channel — using signals the loop already computes for free. *(Phase 0,
   done.)*
2. Derive an **effective token usage profile** `E(σ)`: at each noise level,
   what fraction of tokens is still actively receiving information? *(Phase 1.)*
3. Turn the same traces into a **trajectory-intactness gauge**: a candidate
   intervention must keep the *whole generation process* statistically intact
   relative to baseline, not merely score well on the final image. *(Phase 2.)*

Everything in Phases 0–2 is observability. No generation behavior changes.

## Phase 0 — shipped (reference)

`--traj_stats` on any `inference.py` / `make test` / `make gen` invocation
records one `.npz` sidecar per generation (`--traj_stats_dir`, default
`output/traj_stats/`; `--traj_stats_k` quantization bits, default 4).

- **Recorder**: `library/inference/traj_stats.py::TrajStatsRecorder` —
  observes `x̂₀ = z − σ·v` post-CFG-combine / pre-sampler-step, on `.float()`
  copies, per-channel normalized with the VAE's `latents_mean`/`latents_std`
  (constants pinned against `qwen_vae.py` by test). Traces per the table
  below; `derive_summary()` in the same module produces `E(σ)` + commit-CDF.
- **Hook sites**: `generate_body` inline loop, the tiled loop, and
  `networks/spectrum.py::spectrum_denoise` (threaded via
  `SamplerSideChannels.traj_stats`). SPD and the archived foveated runner are
  **not yet hooked** — the foveated hook is a Phase 2 task (known-bad arm).
- **Invariants** (pinned by `tests/test_traj_stats.py` + the bench):
  recorder on/off latents bit-identical; explicit dim-2 squeeze (5D → 4D);
  off = `None` short-circuit, zero allocations.
- **Measured** (1024², 28 steps, er_sde, CFG 4): overhead **0.12 %**
  (0.56 ms/step; budget was 2 %), spectrum arm 0.31 %; sidecar 2.4 MB.
- **Perf traps for future hook sites** (both bitten during Phase 0): pass
  `sigmas[i]` as the 0-d **tensor** — `float(sigmas[i])` is a stream sync
  that breaks the loop's CPU run-ahead (~28 ms/step); and the sidecar is
  written with uncompressed `np.savez` — zlib cost >100 ms inside the
  generation wall time.

### The statistics

All computed on `x̂₀`, normalized per channel. "Token" = one DiT patch = a
2×2 latent-pixel cell (16 px), so maps are directly comparable to
attention/token-count reasoning.

| trace | definition | what it answers |
|---|---|---|
| **code(p, i)** | k-bit uniform quantization code of x̂₀ (k=4/channel default) | the quantized-VAE view: which discrete cell the token estimate is in |
| **commit(p)** | last step where `code(p, ·)` changed | when did this token's content lock? |
| **activity(p, i)** | ‖x̂₀(p, i) − x̂₀(p, i−1)‖ | is information still flowing into this token? |
| **hf(p, i)** | Laplacian energy of x̂₀ in the token's neighborhood | foveation's x0var, kept for continuity — the trace that flatlined |
| **guide(p, i)** | ‖v_final − v_uncond‖ per token (post-combine; CFG runs) | where the prompt is steering (g-scaled vs the draft's pre-combine form) |
| **cbits(c, i)** | entropy of channel c's code histogram over all tokens | per-channel information ramp — which of the 16 channels the anime domain actually uses, and when |

Derived scalars:

- **E(σ)** — effective token usage: fraction of tokens with
  `activity(p, i) > τ`. τ is provisionally the 95th percentile of final-step
  activity (the σ→0 noise floor); **Phase 1 calibrates it** across the grid
  before any cross-run comparison leans on it.
- **commit-CDF** — distribution of `commit(p)` over σ. The quantified
  headroom for *any* late-step token intervention; if flat, the efficiency
  thesis dies cheaply in Phase 1.

## Phase 1 — the atlas (ACTIVE)

Goal: turn single-render traces into anime-domain trajectory statistics, and
answer three questions: **(a)** how front-loaded is token commitment in this
domain, **(b)** which channels carry it, **(c)** does generation match
inversion (i.e. do corpus statistics transfer to the generation process).

Preview evidence from the Phase 0 run (one prompt, one seed — not a claim):
commit-CDF hit 10 % by σ=0.80, 32 % by σ=0.55, 56 % by σ=0.33; E(σ) fell to
0.15 by σ=0.33. Direction is front-loaded — Phase 1 decides whether that
survives aggregation.

Harness: `bench/traj_stats/run_atlas.py` (to build; shares `bench/_common.py`
envelope + the existing `run_bench.py` render pattern via
`build_inference_bundle`).

1. **Generation arm** — seed × prompt grid over anime prompts with
   `--traj_stats` on. Two routing options: in-process via the bundle (matches
   `run_bench.py`, one model load for the whole grid — preferred), or
   `make gen ARGS="--traj_stats ..."` when a training run owns the GPU (the
   recorder composes with the daemon path unchanged). Grid floor: ~8 prompts
   (reuse `bench/tag_dropout/prompt_sets_*.json` families for spread:
   detailed / sparse / no_trigger) × 4 seeds × {28-step er_sde CFG 4}.
   Aggregate E(σ), commit-CDF, `cbits(c, ·)` with per-cell spread, not just
   means — the τ-sensitivity of E(σ) gets reported alongside (τ swept ±1
   quantile band).
2. **Real-image arm** — DirectEdit inversion
   (`library/inference/editing/directedit.py`) replays *real* cached-corpus
   images as trajectories. Task item: the inversion loop is a separate code
   path — thread the recorder through it the same way as the forward loops
   (side-channel field already exists; keep the sigma-as-tensor rule).
   Recorder on the inversion pass gives the same traces for ground-truth
   anime images.
3. **Static baseline column** — zero-cost corpus statistics computed directly
   from `post_image_dataset/lora/*.npz` cached latents (same quantizer, same
   normalization, no trajectory): per-token/per-channel entropy of clean
   images. This is what redundancy intuitions were always based on; the atlas
   report shows it next to the trajectory-resolved columns so the difference
   is explicit.
4. Deliverable: `bench/traj_stats/report.md` §Phase 1 answering (a)/(b)/(c).
   (c) failing — generation and inversion traces structurally disagreeing —
   is an important negative result on its own: generated trajectories are
   off-manifold and corpus stats only license img2img/editing claims.

## Phase 2 — the trajectory-intactness gauge

The product. For a candidate intervention X and a fixed (prompt, seed, steps):

    D(X) = per-σ divergence profile between X's traces and baseline's traces
           — token-wise x̂₀ RMSE(σ), |E_X(σ) − E_base(σ)|, hf-trace delta,
           commit-CDF shift — reported as a *curve*, not one scalar.

Calibration, both directions:

- **Known-bad**: the archived foveation runner (`--fovea_sigma_c 0.75`).
  Requires hooking the recorder into `networks/foveated.py` (not done in
  Phase 0 — the runner records via the same side-channel field). The gauge
  must show the periphery `hf` trace flatlining below σ_c and a commit-CDF
  hole — i.e. rediscover P4t from traces alone, *without looking at the
  final image*. This is the acceptance test.
- **Known-good**: Spectrum and SMC-CFG composes (shipped, quality-neutral by
  their own benches) must show small, structureless D. The spectrum hook
  already exists and passed bit-exactness in the Phase 0 bench, so this arm
  is ready as soon as the gauge script exists. If the gauge flags
  shipped-good methods, τ / normalization get re-tuned before anyone uses it.

Output: `bench/traj_stats/gauge.py --baseline <dir> --candidate <dir>` →
verdict bands (calibrated in this phase, provisional: intact · perturbed ·
process-broken), riding the same `result.json` envelope.

## Phase 3 — interventions (gated, speculative)

Only if Phase 1 shows exploitable structure (front-loaded commit-CDF, skewed
`cbits`), ranked by the Phase 2 gauge before any quality bench is spent:

- **Entropy-aware tier routing** (training-side, safest): feed per-image
  latent entropy into `choose_edge` (`library/datasets/buckets.py`) so flat
  cel-style images train a tier down. No inference-time process risk — the
  gauge is irrelevant here, Phase 1's static column suffices.
- **Committed-token compute reuse** (inference-side): below a σ threshold,
  reuse the previous step's velocity for tokens whose code has been stable
  for m steps (delta-token / cache-skip — tokens keep their *identity and
  resolution*, unlike merging; staleness is bounded and refreshable, unlike
  foveation's constitutive pooling). Gauge-gated.
- **σ-scheduled channel truncation**: if `cbits` shows late-σ channels idle,
  drop them from the stats/guidance path — *measurement first*.

Explicitly **not** proposed: re-running token *merging* with better selection.
P3/P4t closed that line — the defect was the mechanism (reduced-resolution
denoising), not the mask, and a better redundancy prior does not change that.

## Non-goals / scope guards

- Phases 0–2 never change a generated image. The recorder's bit-exactness is
  pinned by test and re-verified by every bench run.
- No new models, no extra forwards, no training-loop involvement (the
  training-side Phase 3 item consumes Phase 1 *outputs*, not the recorder).
- `k`, τ, and verdict bands are bench-calibrated knobs, not shipped defaults —
  nothing here touches `configs/base.toml`.
- Tiled path support is post-blend only (shipped as such in Phase 0);
  per-tile traces are out of scope.

## Falsifiers (cheap exits)

1. Phase 1 commit-CDF ≈ uniform in σ → no late-step headroom → Phase 3
   inference items die; only tier routing survives (and only if the *static*
   corpus stats are skewed). *(Phase 0's single trace points the other way,
   but n=1.)*
2. Generation-arm vs inversion-arm traces disagree structurally → corpus
   statistics don't transfer to generation → restrict all claims to img2img /
   editing paths.
3. Gauge can't separate foveation (known-bad) from Spectrum (known-good) →
   the trace set is insufficient; stop before anyone trusts it.
