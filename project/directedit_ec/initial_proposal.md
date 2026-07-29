# EasyControl-enhanced DirectEdit — a learned preservation prior on the edit trajectory

Status: **Phase 0 PASSED**; **Phase 1a PASSED, amended**; **Phase 1b PASSED
(3/3 in-place edit types ≥ vinj_t6)** (2026-07-24, all zero-training;
`project/directedit_ec/bench/report.md`). Phase 2 is unblocked. The 1a amendment: the cond
hole alone landed the edit on 1/3 images — the global Δz anchor pulls the
hole back to the source. Dropping the anchor inside the edit region (paper
Eq. 12's anchor-side half, now wired as `--mask` in `scripts/edit.py` /
`directedit.edit_forward`) makes the recipe land 3/3 (incl. the hard image)
at b_offset 0, with outside-hole preservation 2.6–17× better than
best-of-{vinj_t6, ec_b-1, ec_b-2}. The winning recipe is **cond hole +
anchor mask, same file** (`--easycontrol_mask m.png --mask m.png`).
This proposal covers the follow-on phases: masked-cond probe, edit-type
generalization, the cross-image **subject descriptor** (an EasyControl adapter —
same network, new data pairing), and the feed-forward editor endgame.

## One-paragraph pitch

DirectEdit preserves the source through two training-free mechanisms — the Δz
anchor (exact but overpowered by CFG) and V-injection (positional hard V-swap,
`t_inj` steps × block set to tune, kills `compile_blocks`). EasyControl's
extended self-attention is the *learned, gated* generalization of V-injection:
target queries retrieve source appearance from cached cond K/V under a
per-block learned softmax-mass gate (`b_cond`). Phase 0 showed that loading an
existing EasyControl adapter (inpaint, fed a hole-free cond = the source) onto
the edit pass composes **exactly** with the Δz anchor and, at the right gate
offset, **beats V-injection on composition preservation while landing the
edit** — zero training, one interpretable scalar (`--easycontrol_b_offset`),
one KV prefill instead of a per-step parallel src forward.

## Phase 0 results (done — grounding, not plan)

Runs: `project/directedit_ec/bench/results/20260724-1731-phase0-full` (3 img × 8 arms),
`…-1749-phase0b-boffset` (2 img × 10 arms). Edit = caption + ", glasses",
CFG 4, 28 steps. Adapter: `output/ckpt/methods/anima_inpaint.safetensors`.

| Finding | Evidence |
|---|---|
| EC cond stream composes exactly with the Δz anchor | recon gate (ψ_tar==ψ_src, CFG 1): recon_ec/recon_base = 0.85–0.97 on all images |
| `cond_scale` is **near-binary** — not a dial | 0.25/0.5 ≈ no-EC baseline (MSE 0.136–0.137 vs 0.140); 1.0 = total clamp (0.0005–0.002, edit suppressed) |
| `b_cond` offset **is** the dial | live logit bias, not baked into the KV cache; each −1 ≈ e× less cond mass; useful range −1..−2, −3/−4 disengaged |
| EC @ sweet spot beats vinj_t6 on composition | dan@−1: glasses land + full source background survives; vinj_t6 landed glasses but invented a fireworks background |
| Sweet spot is image-dependent | second image needed −2 (at −1 the edit didn't land — vinj_t6 also failed there) |
| Pure anchor at CFG 4 loses composition | base_t0 MSE 0.06–0.14 — the fragility this line exists to fix |

Mechanism note for everything below: EC and V-injection **cannot stack** — the
EC-patched `Block.forward` routes attention through
`_extended_target_attention`, bypassing the `Attention.forward` that
`_v_injection_scope` patches. The design is *replace*, not compose.

## Why the sweet spot is narrow (and what widens it)

The inpaint adapter is used off-label: it was trained "cond is authoritative"
(`b_cond_init=-6`, `easycontrol_drop_p=0`, hole-free cond slightly OOD), so its
operating point is cliff-shaped — full copy or disengaged, with the usable
band squeezed into b_offset ∈ [−2, −1]. Two orthogonal wideners:

1. **Give the prior its trained exception channel back** (Phase 1a, zero
   training): the inpaint adapter *already knows* "copy everything except the
   hole". Punch the hole over the edit region.
2. **Train a prior whose loss can't be satisfied by positional copying**
   (Phase 2): cross-image pairs of the same character force content-based
   retrieval — the capability none of the shipped aligned-pair adapters
   (inpaint / colorize / sanitize) has.

## Phases

### Phase 1a — masked-cond probe (zero training, ~1 GPU-hour) — **DONE, PASSED AMENDED**

Result (`project/directedit_ec/bench/report.md` Phase 1a): exception-driven
preservation works exactly as pitched, but the edit only lands once the Δz
anchor is ALSO masked inside the edit region (`ec_mask_anch` arm). The
literal ≤2× outside-MSE gate is mis-calibrated (recon is near-pixel-exact;
ratios 2.4–61× at 0.0003–0.003 absolute drift, still far below every
alternative arm) — judged on renders + vs-alternatives, the probe passes.
Known artifact: on 7538087 the hole regenerates with a flat saturated style
(inpaint-prior artifact, present in all EC arms on that image). Original
plan below.

Feed the inpaint prior what it was trained on: cond = source latent with a
**gray hole over the intended edit region** (`easycontrol_adapters/inpainting/`
already owns the hole-drawing code). Outside the hole the prior clamps
(trained behavior, no gate fiddling); inside it generates freely, steered by
ψ_tar through cross-attn. This is DirectEdit's never-implemented Eq. 12 mask
blending, obtained for free from a trained prior — and it should dissolve the
per-image b_offset tuning for localized edits.

- Mask source: user box first (bench: fixed face-region box); the cfgdelta
  subject localizer (`bench/` foveation line's reusable artifact) as the
  automatic upgrade.
- Wiring: `--easycontrol_mask <path>` in `scripts/edit.py` — gray-fill the
  masked region of the cond *image* before VAE encode (matches training
  distribution; do NOT zero the latent).
- **Gate**: on the Phase-0 image set + the hard image (10473210), edit lands
  inside the mask AND outside-mask MSE ≤ 2× recon level, at b_offset 0
  (no tuning). Compare against best-of-{vinj_t6, ec_b-1, ec_b-2}.

### Phase 1b — edit-type generalization (zero training, ~2 GPU-hours) — **DONE, PASSED 3/3**

Result (`project/directedit_ec/bench/report.md` Phase 1b): ec_mask_anch ≥ vinj_t6 on
all three in-place types — REMOVE and REPLACE land only under ec_mask_anch
(vinj_t6 landed neither anywhere); expression parity. Geometry control failed
as expected, with the twist that the full-frame box DOES produce the pose
(suppression is preservation-owned) while keeping nothing — Phase 2's
falsifiable target. Hard-image ceiling: 10473210's halo-removal and
hair-recolor fail for every method. Original plan below.

Phase 0 tested one additive edit. Sweep the b_offset (and 1a mask) recipe
over: REMOVE (drop a tag present in ψ_src), REPLACE (hair color), expression,
and a geometry edit (pose tag) as the expected-failure control. n=3 images ×
4 edit types × {ec_b-1, ec_b-2, ec_mask, vinj_t6, base_t0}.
- **Gate**: EC recipe ≥ vinj_t6 on (edit lands, composition held) judged on
  renders, for ≥ 2 of 3 in-place edit types. Geometry edits are *expected* to
  fail (position-locked prior) — record, don't gate.
- Optional metric upgrade: tag-readback edit-success once the readback judge
  ships (`tag_readback_reward.md` Phase 0a passed; blocked only on a trained
  tagger checkpoint being present).

### Phase 2 — cross-image **subject descriptor** (one standard EasyControl train)

A new descriptor `configs/easycontrol/subject.toml` (colorize/inpaint shape) —
**the same `EasyControlNetwork`**, only the data pairing changes:

- **Pairs**: cond = image A of a character, target = image B of the same
  character, mined from `caption_index.json` character tags (fallback: same
  artist + shared character-defining tags). Staging emits a pair manifest;
  cond latents reuse the shared LoRA cache (both sides are corpus images —
  cheapest staging of any descriptor yet, no synthetic tree).
- **Anti-shortcut knobs**: `cond_res_scale = 0.5` (starves the positional
  shortcut, pays for itself in speed); mild `easycontrol_drop_p = 0.05` and
  `b_cond_init = -8` so the gate learns a softer operating point than
  inpaint's all-in −6.
- **Text**: full captions of the *target* (the prompt must keep owning
  layout/pose; the cond should own identity/appearance only).
- Cost: inpaint-recipe scale — ~4 epochs, 16 GiB-friendly, same daemon flow
  (`make easycontrol EASYADAPTER=subject`).
- **Gate** (the Phase-0 harness re-run, `project/directedit_ec/bench/run_bench.py
  --ec_weight <subject.safetensors>`): (a) sweet-spot width — the b_offset
  range where the edit lands with composition held spans ≥ 2 units (inpaint:
  ~1); (b) at least parity with vinj_t6 on the geometry edit from 1b that the
  inpaint prior failed (the associative-retrieval claim, falsifiable).

### Phase 3 — feed-forward editor (endgame, gated on 1b)

Distill DirectEdit itself: use the Phase-1b recipe + tagger to synthesize
`(source, edited, edit-caption)` pairs at scale, train an EasyControl editor
descriptor on them (cond = source, target = edited). Inference becomes **one
cached-cond generation — no inversion pass, no anchor, no patching**.
InstructPix2Pix's recipe with our own tag vocabulary and a training-free
teacher. Proposed only if 1b shows the teacher is reliable enough to label its
own data; separate proposal when reached.

## Relation to BYG (and the cost argument)

BYG (`docs/experimental/byg.md`, demoted 2026-07-02, validation-gated) is the
in-tree training-based editing line: no paired data needed, but the price is a
bespoke multi-forward training step (four losses, staged identity backward,
STE) — and its inference conditioning patch was never wired. This line gets a
working preservation/edit tradeoff at **zero** training (Phases 0–1), and its
first training phase (2) is a *standard* EasyControl recipe on real image
pairs — no bootstrap loop, no reward model, ordinary descriptor plumbing.
BYG's niche (free-form *instruction* following beyond tag edits) is untouched;
if its Phase-0 gate ever passes, a BYG arm belongs in this bench.

## Paper-readiness (honest version)

The novel claim is narrow but real: *a pretrained image-conditioning adapter's
attention gate, offset at inference, is a continuous preservation dial for
flow-inversion editing — replacing hand-tuned attention injection with one
scalar, composing exactly with residual-anchored inversion.* What a paper
additionally needs (none exists yet): external baselines under matched NFE
(RF-Inversion / RF-Solver / FireFlow / FlowEdit class), a public edit
benchmark (PIE-Bench) beside the in-house set, quantitative edit-success +
identity metrics (CLIP-dir / DINO-identity, or our tagger-readback +
PE-Core), and the Phase-2 adapter so the story isn't "one off-label inpaint
checkpoint". Per the FSG lesson (`project_fsg_golden_path_phase0`): no
free-quality claims without the matched-NFE table. Phases 1a–2 produce
exactly that table; decide on writing after Phase 2's gate.

## What would kill this line

- Phase 1a mask probe failing on the hard image → the prior can't be
  exception-driven even in-distribution → preservation is all-or-nothing and
  V-injection keeps the crown; line closes as "EC = recon-only tool".
- Phase 2 sweet-spot width not improving over inpaint's → aligned-vs-cross
  pairing wasn't the binding constraint; the cliff is architectural
  (gate granularity), and the next lever would be per-block/per-σ gate
  schedules — a different, smaller proposal.
- Every phase gates on renders (small n) — per repo policy, no CMMD at this
  scale (`project_seed_floor_cmmd_fragile`); MSE-vs-source is a preservation
  proxy only, never an edit-quality metric.

## Costs

| Phase | Compute | New surface |
|---|---|---|
| 1a | ~1 GPU-h | `--easycontrol_mask` in edit.py + bench arm |
| 1b | ~2 GPU-h | bench edit-type matrix |
| 2 | 1 descriptor train (~inpaint-scale) + bench re-run | `configs/easycontrol/subject.toml` + pair-mining stage |
| 3 | pair synthesis + 1 train | separate proposal |
