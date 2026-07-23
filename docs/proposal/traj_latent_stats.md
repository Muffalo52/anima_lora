# Trajectory-resolved latent statistics — effective token usage measured *during* generation

Status: **Phase 0 SHIPPED & PASSED 2026-07-23** (recorder + invariant tests +
bench — `bench/traj_stats/results/20260723-2042-phase0/`).
**Phase 1 (atlas) DONE 2026-07-23** — `run_atlas.py`, run
`bench/traj_stats/results/20260723-2100-phase1/`, report §Phase 1: (a)
front-loaded commitment confirmed at aggregate scale (~half of tokens
code-committed by σ=0.5, ~2/3 by σ=1/3, tight spread), (b) channel usage
skewed ~4× and corpus-stable, (c) generation ↔ inversion traces structurally
agree below σ≈0.92 — corpus statistics transfer; neither falsifier fired.
**Phase 2 (intactness gauge) DONE & PASSED 2026-07-23** —
`bench/traj_stats/gauge.py` + `run_gauge_calibration.py`, run
`results/20260723-2130-phase2-recal/`: one exemplar per verdict band
(SMC-CFG intact · Spectrum perturbed · foveation process-broken). P4t
rediscovered from traces alone via commit-CDF hole 0.17 + in-loop hf
blow-up ~30× (the predicted *flatline* only exists post-readout — detector
retained). Key calibration finding: **quality-neutral ≠
process-transparent** (Spectrum's forecast steps are process-visible;
verdicts are driven by distributional metrics only, pointwise divergence is
descriptive). Falsifier 3 does not fire. **Phase 3 (interventions) is now
unblocked but stays gated per-item** — compute-reuse must clear this gauge
before any quality bench is spent; channel truncation demoted to
measurement-only.
**Phase 3a (tier routing gate) RUN & FAILED 2026-07-23** —
`bench/tier_routing/report.md`: demotion gradient cost real but flat in
redundancy; the tier-routing item is closed (see §Phase 3). Compute-reuse
and the decode probe remain the open Phase 3 items.

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

## Phase 1 — the atlas (DONE 2026-07-23 — see report §Phase 1)

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

## Phase 2 — the trajectory-intactness gauge (DONE 2026-07-23 — see report §Phase 2)

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
  gauge is irrelevant here, Phase 1's static column suffices. **CLOSED 2026-07-23
  — FAILED at the 3a gate** (`bench/tier_routing/report.md`): demote-one-tier
  gradient cost is real and reproducible (gap ≈ 0.074 at 896, 0.147 at 768,
  vs re-encode control ≈ 0) but **flat in redundancy** (quartile means
  indistinguishable, bootstrap P = 0.60) — static corpus redundancy does not
  translate to gradient-space demotion safety. Per-image gap ranking has
  ~zero split-half reliability at K≤32 (heavy-tailed σ-draw variance), so
  selective routing has no measurable basis; 3b was never built. The probe
  harness + the redraw-floor/re-encode-control/split-half methodology are
  the reusable artifacts.
- **Committed-token compute reuse** (inference-side): below a σ threshold,
  reuse the previous step's velocity for tokens whose code has been stable
  for m steps (delta-token / cache-skip — tokens keep their *identity and
  resolution*, unlike merging; staleness is bounded and refreshable, unlike
  foveation's constitutive pooling). Gauge-gated. **Design constraint from
  the Phase 1 addendum: detect, don't predict** — which tokens commit late
  is not predictable a priori (commit vs final-hf Spearman ≈ 0.17), only
  online code-stability detection is reliable; any pre-drawn-mask variant
  is refuted twice over (P4t + the addendum).
- **σ-scheduled channel truncation**: if `cbits` shows late-σ channels idle,
  drop them from the stats/guidance path — *measurement first*. **Demoted
  to measurement-only at Phase 2**: the 16 latent channels mix into
  1024-dim tokens at patch embed, so there is no DiT compute here.
- **Subspace-truncated decode probe** (VAE-side, measurement-only, no
  gauge needed): the Phase 1 addendum shows token-level statistics live in
  a fixed ~3-dim channel subspace (effective rank 2.7–3.4 / 16, top-4
  directions ≈ 90–94 % var, static↔generation principal angles ≈ 1).
  Project cached latents onto top-k PCA directions, decode, compare
  (LPIPS + eyes) against full decode — bounds how much of the channel
  space decode *quality* actually needs, informing latent-storage
  compression and nothing else. Cheap, zero process risk; "statistics
  live low-dim" ≠ "decode survives projection" until this runs.

Explicitly **not** proposed: re-running token *merging* with better selection.
P3/P4t closed that line — the defect was the mechanism (reduced-resolution
denoising), not the mask, and a better redundancy prior does not change that.

### Phase 3a — the gradient-equivalence probe (the gate for tier routing)

The routing input exists (Phase 1: modal joint-code share 0.11, unique-code
fraction 0.26 at k=4, skewed per image). Before shipping any routing, validate
the *claim* behind it: that for high-redundancy images, training on the
tier-down cache produces the same gradient the native-tier cache does. This is
a per-image, training-free instrument — no quality bench, no CMMD.

**Prior art / how this differs from archived autoscale.** The autoscale
resolution curriculum (`_archive/proposals/autoscale_resolution_curriculum.md`,
reverted 2026-06-28) already measured cross-resolution gradient alignment —
corpus-mean instantaneous cosine ≈ 0.75–0.78 (896↔1024, barely above
noise-seed variation) — and then failed Phase 1 at matched FLOPs: the
artist-LoRA regime is data/plateau-bound, not compute-bound. Two lessons
inherited here: (1) this probe's thesis is that autoscale's *corpus-mean*
cosine hides per-image heterogeneity predicted by static redundancy —
routing is selective where the curriculum was blanket; if the per-image
spread turns out flat (everyone ≈ 0.77 regardless of entropy), the routing
signal doesn't exist and the item dies at 3a. (2) The 3b payoff claim is
**wall-clock only** (same steps, cheaper steps). "Reinvest the saved FLOPs
into more steps for quality" is exactly the claim autoscale falsified —
never resurrect it.

**Design.** For each probe image, expected-gradient estimates at a fixed
trained-LoRA operating point, each accumulated over K draws on a shared
stratified σ grid (quantile midpoints of the training `shift` distribution —
matched σ across arms; ε cannot be matched across resolutions since the
latent shapes differ, so every estimate uses its own disjoint seed set and
the floor is built by the identical protocol):

- `ḡ_T^A`, `ḡ_T^B` — native tier, two disjoint draw sets. Their cosine is the
  **within-tier redraw floor**: the best agreement any two estimates of the
  same input can reach at this K. This is the null — *not* cos = 1.
- `ḡ_{T−1}` — the demoted-tier arm: the native-tier resized PNG down-resized
  to the lower tier's free-fit bucket and VAE-encoded (same caption / TE
  cache; text is resolution-independent). Verdict per image:
  `gap = cos(ḡ_T^A, ḡ_T^B) − cos(ḡ_T, ḡ_{T−1})`. Gradients live in the shared
  LoRA parameter space regardless of token count, so the cross-resolution
  cosine is well-defined, and cosine absorbs the per-token-mean loss
  normalization difference between tiers. Probing two demote depths
  (1024→896 and 1024→768) gives the dose-response for free.
- `ḡ_reenc` — confound control: the native PNG re-encoded through the probe's
  own resize→VAE chain *at the native bucket*. If cos(native, reenc) sits
  below the floor, the encode chain (not resolution) is contaminating the
  demoted arms and the probe is invalid.

**Headline metric**: Spearman(per-image static redundancy → gap). If the
static entropy scalar predicts gradient-content loss, the routing threshold
is read directly off that curve (demote where the predicted gap vanishes
into the redraw floor) and 3b proceeds. If it doesn't predict, the item
dies for the price of the probe.

**Pinned pitfalls** (from design review): (1) fresh-LoRA B=0 init makes
grad(A) identically zero — probe at a *trained* checkpoint (answers "does
demotion change what a real run learns"); (2) K must be validated by the
floor itself (floor ≪ 1 ⇒ raise K; start K≈16–32); (3) σ grid shared and
stratified across arms — commit front-loading means low-σ draws carry
little signal; (4) probe-set caches for the demoted tier are built in a
scratch dir, never written into `post_image_dataset/` (sidecar-clobber
class of bug).

**Residual the probe cannot see**: curriculum effects — whether shifting the
corpus resolution mix changes what multi-scale training converges to in
aggregate. Bounded by demote-one-tier + conservative threshold; a
throughput-framed full-run comparison is a 3b-ship follow-up, not a gate.

Harness: `bench/tier_routing/run_grad_probe.py` (standard `bench/_common.py`
envelope), ~30–50 images spanning the redundancy range across 2–3 artists.

### Phase 3b — ship shape (gated on 3a)

Two-pass preprocess; training untouched (`make_buckets` treats on-disk caches
as the source of truth, so a demoted image is indistinguishable from one that
natively landed a tier down — and multi-scale 512–1536 training is already
the normal regime, so demotion stays inside the known distribution):

1. Normal `make preprocess` (caches exist at today's `choose_edge` tier).
2. Routing pass: read cached `.npz`, quantize with the *identical* Phase-1
   quantizer (`library/inference/traj_stats.py`), compute the per-image
   redundancy scalar, write a demotion manifest.
3. Re-resize with the manifest consumed by `choose_edge` as a
   **demote-one-tier gate** (threshold from 3a; deliberately not a rewrite of
   the cost function — multi-tier demotion is unmeasured), then
   `make preprocess-reconcile ARGS="--delete"` sweeps the moved buckets.

Opt-in flag; nothing touches `configs/base.toml` (scope guard above).
Invariant test: routing off ⇒ bucket assignment bit-identical; routing
deterministic given manifest. Payoff is deterministic and quality-bench-free:
1024→768 is 4200→2160 tokens (~2× attention cost per demoted image), and the
demotion fraction at the chosen threshold is computable from the corpus.

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
