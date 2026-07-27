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

## Mechanism addendum (2026-07-24): two-term account confirmed

`hypothesis.md` pre-registered gap_e(σ) ≈ S1_e(σ) (input branch, Wiener-decay,
no hard gate) + Floor_e (target × graph, σ-independent). Both discriminating
probes passed: the σ=1 endpoint bin (`results/20260724-2101-endpoint/` —
input pure ε, gaps −0.01/0.13/0.33 for 896/768/512, all in pre-committed
bands) and the x-zero probe (`results/20260724-2136-xzero/` — x=0 in input
AND target; 512 xz ≈ endpoint ⇒ Floor graph-dominated). See hypothesis.md
for the account and groundings.md for the full test records (G1/G2) +
outcome matrix.

## Phase Q2 (2026-07-25): per-module / per-block Floor localization

Runs: `results/20260724-2306-endpoint-pg/` (standard arms, σ=1 only, 16
draws) + `results/20260724-2343-xzero-pg/` (x-zero arms, 4 bins + endpoint),
both `--per_group`: 15 module-type groups (incl. `lora_up` row-splits of the
fused qkv/kv projections — the RoPE q/k-vs-v discriminator) × 28 block
groups. Validity: per-group gap_reenc −0.045..−0.01 for every group;
split-half of block curves 0.61–1.00.

**Verdict: the Floor localizes in DEPTH, not module type — and RoPE is
refuted as a concentrated mechanism.**

- **Type axis ≈ flat.** At 512/σ=1 every module type sits in 0.22–0.31
  (endpoint) / 0.14–0.30 (x-zero) — no clean type, no dominant type.
  Notably `self_attn_up_q` +0.28 ≈ `up_k` +0.30 ≈ `up_v` +0.30 (endpoint;
  x-zero 0.23/0.24/0.24): **q/k show zero excess over v**, so RoPE geometry
  is not where the sensitivity is concentrated. `adaln_up_*` are among the
  highest (0.26–0.31) with ep−xz ≈ 0 — global-statistics shift, not content.
- **Depth axis is the signal.** Early blocks carry ~3× the late-block gap
  (means at σ=1, early = blocks 0–9, late = 14–27):

  | edge | ep early | ep late | xz early | xz late |
  |---|---|---|---|---|
  | 512 | .357 | .223 | **.351** | .125 |
  | 768 | .164 | .085 | .121 | .033 |
  | 896 | .023 | .015 | .044 | .012 |

  The 512 x-zero profile peaks at blocks 3–8 (up to 0.48 at block 4) and
  decays to ~0.12 by block 14. Early-block gap is σ-flat in the x-zero run
  (no S1-style decline) — pure Floor.
- **The content share (ep − xz) lives in LATE blocks**: at 512, early
  ep−xz ≈ 0.006 (nothing) vs late ≈ 0.10; same shape at 768. Graph
  sensitivity is created early; content correspondence is a late-block,
  minority effect.

**Interpretation**: per-type uniformity within a block + early concentration
+ q/k ≈ v says the mechanism is not a parameter circuit but **early-block
representation building**: the first ~10 blocks construct grid-calibrated
token statistics (attention over N tokens of ~noise differs in its softmax
statistics with N), the divergence propagates into every param type's
gradient in those blocks roughly equally, and deeper blocks inherit a
partially washed-out version. This is Q2's "subset of parameters for which
demotion IS safe" answer inverted: the safe subset is a **depth band, not a
module type** — late-half-only updates would see gap ≈ 0.03 (768 x-zero) to
0.09 (768 endpoint) vs 0.12 full. Quantitatively that does NOT yet clear the
reenc band at 768 on its own; it is a lever, not a free win.

## Pooled-gradient addendum (2026-07-25): the SGD-aggregate view agrees, harder

Run: `results/20260725-2155-pool4/` — the probe's new `--pool` mode (40
images sorted by redundancy, strata of 4, 4 uniform bins + σ=1 endpoint,
4 draws/bin). Per stratum and for the all-images aggregate, per-image
bin-gradients are **summed across images before cosines** — the
batch-gradient object SGD actually follows — in two variants: unweighted
(training-realistic, gnorm-weighted) and per-image-normalized. Pooled floors
carry their own noise-redraw + image-split-half nulls; pooled numbers are NOT
comparable to the per-image ±0.04 band.

Aggregate (n=40) pooled gap curves, bins 0.125 / 0.375 / 0.625 / 0.875 / 1.0:

| | 0.125 | 0.375 | 0.625 | 0.875 | 1.0 |
|---|---|---|---|---|---|
| cos_floor | .921 | .837 | .982 | .992 | .992 |
| gap_reenc | .028 | −.008 | .009 | .001 | .001 |
| gap_896 | .210 | .447 | **.019** | **.005** | **−.002** |
| gap_768 | .132 | .308 | .104 | **.013** | **.023** |
| norm_gap_896 | .103 | .115 | .015 | .009 | .013 |
| norm_gap_768 | .137 | .256 | .054 | .012 | .042 |

- **The σ-structure sharpens in the aggregate.** Pooled gap_896 collapses to
  ≈ 0 (within the pooled reenc control) at every bin σ ≥ 0.625 — the
  per-image residual (~0.05–0.10 per-image in this run) largely **averages
  out across images**: what survives pooling at high σ is the shared
  cross-image gradient component, and demotion preserves it. Even pooled
  gap_768 nearly vanishes at σ ≥ 0.875. Low σ is the mirror image: pooled
  gaps 0.10–0.45, far above floor noise.
- **Gnorm-weighting artifact**: unweighted pooled gap_896 > gap_768 at low σ
  (0.45 vs 0.31) — an inversion driven by a few large-gnorm images dominating
  the unweighted sum; the normalized side-channel restores the expected
  tier ordering (0.115 vs 0.256). Read tier ordering from `norm_`, magnitude
  realism from unweighted.
- **Redundancy trend: null.** Stratum-level Spearman(redundancy → pooled gap
  @ σ=1) = −0.07 (896) / −0.05 (768) across 10 strata spanning redundancy
  0.49–0.93. Consistent with the tier_routing 3a closure: demotion cost is
  flat in redundancy — there is no "demote the redundant images first" lever
  in the pooled view either. Individual strata are unstable at n=4 (s3–s5
  show floors 0.74–0.88 and wild gaps incl. one +1.0 read); only the
  aggregate and the 10-stratum trend are verdict quantities.
- Validity: pooled gap_reenc ≈ 0 everywhere (|·| ≤ 0.028); imgsplit floor
  tracks the redraw floor (0.78–0.99, same shape).

**Implication for Q4**: the gradient-level case for the {1024→896 @ σ>0.5}
route is now stronger than "within the reenc band per-image" — in the
aggregate gradient the route costs ≈ nothing at σ ≥ 0.625 while per-image
deviations cancel. This is still not the Phase-1b CMMD A/B (integration over
thousands of steps + optimizer state is untested), but the most SGD-like
static object we can measure now agrees.

Ops note: `--pool` initially OOM-killed the box (two ~19 GB fp32 accumulator
sets + ~8 GB per-image temps vs 46 GB RAM). Fixed in `run_sigma_probe.py`:
the aggregate accumulator is disk-memmap-backed (`release()`/`ensure_open()`
between merges — ~19 GB transient under the run dir, deleted at the end) and
per-image gradient lists are freed eagerly. Verified bit-identical to the
in-RAM path.

## 1280→1024 probe (2026-07-26): ratio governor REFUTED; the threshold is route-dependent

Run: `results/20260726-2017/` — the cheap variant (24 images, 4 uniform
σ-bins × 4 draws + σ=1 endpoint, arms native×2 + reenc + 1024, `--pool 8`;
**47 min vs Phase 0's 2.6 h**). Data: probe-local 1280-tier cache built by
`prep_1280_probe.py` — **no corpus re-preprocess** (and none wanted: on-disk
caches are the source of truth for training buckets, so an in-corpus 1280
tier would silently leak into the next run). 36 sources ≥ 6300 native tokens
(2454/3008 of the corpus qualify), redundancy-stratified ≤3/artist,
production resize + VAE chains (so reenc stays a genuine encode-chain
control), TE caches symlinked (text-only). `--data_root` threads the
alternate root through the probe. Instrument valid: gap_reenc |mean| ≤
0.048, split-half 0.56–0.89. No OOM at 6300 tokens under block compile
(budget 0.99).

| σ bin | 0.125 | 0.375 | 0.625 | 0.875 | 1.0 |
|---|---|---|---|---|---|
| gap_1024 | .176 | .160 | **.096** | **−.015** | .077† |
| gap_reenc | .048 | −.005 | .047 | .020 | −.011 |
| pooled gap_1024 | .080 | .082 | .031 | **.006** | **.007** |
| pooled norm_gap_1024 | .101 | .093 | .029 | .002 | .025 |

† endpoint mean is one outlier (`asou_(asabu202)/6278695`, gap +1.13, the
highest-redundancy image in the set); median 0.008, mean without it ≈ 0.03,
pooled 0.007 — the σ=1 bin is in-band by every robust read.

- **Both single-governor pre-registrations miss, asymmetrically.** The
  capacity prediction ("in band at σ ≥ 0.5") fails at the 0.625 bin (0.096
  ± 0.028, ~2 SEM above its control). The ratio prediction ("stays elevated
  like 896→768") fails harder: 896→768 never reached the band at any σ,
  while 1280→1024 is cleanly in-band at 0.875 and σ=1, per-image and pooled.
- **The ordering discriminates: ratio is refuted as the governor.**
  1280→1024 (ratio 0.80) floors by σ ≈ 0.75 despite being *more aggressive
  by ratio* than the never-flooring 896→768 (0.857). Absolute target
  capacity (4116 vs 3012 tokens) predicts exactly this ordering.
- **What replaces "σ > 0.5" is a route-dependent crossover σ\*(route)**:
  ≈ 0.5 for 1024→896, somewhere in (0.625, 0.875) for 1280→1024, > 0.95 (or
  nonexistent) for 896→768. Consistent with hypothesis.md's smoothness
  reading — no universal invariant, safety = how well the coarse graph
  approximates the fine one, which degrades with both ratio *and* absolute
  coarseness.
- Pool view agrees and sharpens (as in pool4): pooled gap ≈ 0 at σ ≥ 0.875,
  0.031 at 0.625. Redundancy trend at 3 strata: −0.5, too few strata to
  read (noted only).
- **Practical residue is gate-position-sensitive**: per-draw saving on a
  1280-tier image is 0.65×, but at the trainer's σ-density the mass above
  the gate matters — σ > 0.75 captures ~14% of draws (~5% epoch saving) vs
  ~27% if σ\* resolves near 0.65 (~9%). Hence the σ-window refinement below.
  NB the corpus currently has **no 1280 tier** (`target_res = [1024, 896]`);
  the route pays only if a 1280 tier is adopted — today's value is the
  ratio-vs-capacity discrimination and the third point on the (route, σ) map.

Follow-ups: the σ-window refinement (`--sigma_window 0.5,1.0`, bins packed
into the crossover region) was started then **deprioritized at 5/24 images**
(partial rows in `results/20260726-2109/`; same command re-runs it) in favor
of the **forward-only prior-distance probe** (`run_prior_distance.py`,
`results/20260726-2120/`, 4.5 min) — result in groundings.md G6
("prior-distance probe"): no discontinuity at the training-distribution edge
(the checkpoint's prior distinguishes 1280-vs-1024 as much as trained
pairs), and prior distance dissociates from the gradient Floor → the
Floor's route-ordering lives in the graph (J) factor, not the prior.
Its σ-resolved v2 (`--sigmas`, mean-residual verdict object;
`results/20260726-2133/`, 16 min) found the residual-distance σ-shape is
strong but **route-uniform** — groundings.md G7 ("σ-resolved residual
curve"): gap_e(σ) ≈ A_e·s(σ) + Floor_e with s universal and both A_e and Floor_e
J-side; the σ\* ordering is not prediction-side. The **iso-severity
discriminator** (1280→1120, ratio 0.875 matched to 1024→896;
`results/20260726-2153/`) landed the same day — groundings.md G9:
**A ~ ratio** (ratio-matched routes coincide despite 1.6× target-capacity
difference; σ\*(1280→1120) ∈ (0.375, 0.625) ≈ 1024→896's). NB this does
not conflict with "ratio is refuted as the governor" above — that verdict
concerns the Floor/σ\*-route-ordering; G9 pins the *S1 amplitude*. The
division of labor: ratio → S1 amplitude (A_e), absolute target capacity →
Floor_e.

## Phase 1b in-vivo weight-space A/B (2026-07-27): displacement orders by demoted fraction; the demote signature lives in LATE blocks

First full-training read of the wiring (`--sigma_lowres`, tenth preset ×
4 epochs, identical 10% subset). Four arms: base (never demote),
σ>0.75 (~14% of steps demoted), σ>0.5 (~48%), threshold-0 (**every**
1024-tier step demoted — the outside-safe-region control). Two sweeps:

- **Unpaired (no seed — `tenth4p_*`)**: `train.py` draws a random seed when
  none is set (`train.py:2282`), so every run gets fresh random-init
  directions for the non-frozen modules + unseeded stochastics. All pairwise
  ΔW cosines land at **0.09–0.10 regardless of arm** — the init/noise
  lottery floor. Weight space is unreadable without pairing. (The lottery
  was previously masked in whole-checkpoint cosines by the frozen Ortho/SVD
  bases — compare ΔW = scale·up@down, never raw checkpoints.)
- **Paired (`--seed 42 --paired_step_rng` — `tenth4s_*`)**: CRN mode (σ +
  noise from per-step-seeded generators, `library/runtime/noise.py::
  draw_flat_sigmas(generator=…)`). Lockstep witnessed in the logs: base ≡
  σ>0.5 arm bit-exact at step 2, all arms tracking to 3–4 decimals through
  step 10; residual ~1e-4 wobble between should-be-identical prefixes is
  hardware nondeterminism (flash backward), which chaos then amplifies —
  absolute cosines below understate similarity; the ordering is the read.

Paired global ΔW cosines (rank-space `(UaᵀUb) ⊙ (DaDbᵀ)` accumulation —
never materialize up@down; the full-ΔW version eats all RAM):

| pair | cos | demote-set difference |
|---|---|---|
| base ↔ σ>0.75 | **0.395** | ~14% of steps |
| base ↔ σ>0.5 | **0.320** | ~48% |
| σ>0.75 ↔ σ>0.5 | 0.343 | ~34% |
| σ>0.5 ↔ thr-0 | 0.245 | ~52% |
| σ>0.75 ↔ thr-0 | 0.195 | ~86% |
| base ↔ thr-0 | **0.184** | 100% |

Monotone in demoted-fraction difference, all far above the 0.09 unpaired
floor. **Depth profile**: the gated arms keep base's late-block structure
(σ>0.75 vs base climbs to 0.62–0.80 over blocks 22–27; σ>0.5 to
0.48–0.72) while the always-demote control collapses exactly there
(0.15–0.19 across blocks 20–26 — its worst blocks, vs 0.42+ at block 27
for everyone). Unconditional 896 training relearns the late
(content/detail) blocks; σ-gated demotion largely preserves them. Blocks
3–5/9–11 are everyone's most chaotic (0.09–0.21) — low-signal directions,
not a demote effect (uniform across arms).

Verdict: in-vivo confirmation of the map's shape at training scale —
demote-induced displacement is real, scales with demoted fraction, and is
concentrated where unconditional low-res training should bite; the σ>0.75
arm is nearly as close to base as hardware chaos allows. NOT yet the gate:
CMMD non-inferiority + rendered comparison still decide (renders:
`output/tests/sigma_ab/` for the unpaired arms, seeds 0–3). Checkpoints:
`output/ckpt/anima_lora_tenth4{p,s}_*.safetensors`.

## Caveats

- Single operating point (`anima_soup_sincos`, trained at native tiers). An
  adapter trained mixed-res might equalize its own gradients — untested; any
  reopen should probe a mixed-res-trained operating point first.
  **In-pool overlap checked (2026-07-26)**: the probe pool contains 2/40
  `sincos/*` stems (the adapter's fine-tune set; Phase 1a pool has none).
  No stem-level bias: their per-image gap percentiles scatter 0.0–0.9 with
  no direction (the one high endpoint read, `7082181` gap_768 = 1.19, is
  the G3 per-image estimator noise — the same stem ranks *lowest* in
  Phase-0's high-σ bins), and leave-both-out shifts endpoint means ≤ 0.026
  (768: 0.127→0.101, 512: 0.326→0.307) — every verdict stays in its
  pre-registered band.
- Uniform σ bins; per-bin cosines use 8 draws (floor correspondingly lower at
  mid-σ where ‖g‖ is small — gap subtraction controls this, and reenc stays
  ≈ 0, but absolute cosines across bins are not comparable).
- 896-at-high-σ "safe" = within reenc band at N=40; it is a small residual
  (~0.03, ~2 SEM > 0), not exact zero.
