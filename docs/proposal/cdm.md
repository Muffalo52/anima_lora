# CDM — continuous-time distribution matching for the turbo loop

Status: **PHASE 0 (dynamic schedule) WIRED 2026-07-17, ON by default in
`configs/methods/turbo.toml` — 500-step smoke A/B was a decisive dynamic win:
every gate axis, every eval style, visible at a glance (details in the Phase 0
section). The verdict-grade 2k read is queued (`queued.md`); Phases 1–3 stay
gated on it, but the smoke also validated the line's core thesis — see "Why
this line".**

- **Phase 0 shipped**: `dmd.dynamic_schedule` — per-iteration random continuous
  rollout grid (`primitives.sample_dynamic_sigmas`, consumed as `sigmas_it` /
  `n_steps_it` in `distill.py`'s student rollout). `false` = bit-identical
  legacy fixed-grid loop; old snapshots resolve to `false` and reproduce.
  Guards: refuses `per_step_expert` (heads keyed to fixed grid steps) and
  dpdmd with `student_steps < 2`. Resume warns across a flip (`_WARN_FIELDS`);
  ckpt metadata carries `ss_turbo_dynamic_schedule`. Invariant tests:
  `tests/test_turbo_dynamic_schedule.py`.
- **Untouched by design**: the DP diversity anchor (t₁ = 1 stays pinned, so
  `v_target` math is unchanged), the diversity validation and inference (both
  stay on the static `flow_shift` grid), the GAN, `teacher_cfg=4`.
- **Next**: the verdict-grade 2k A/B (fresh both arms — the 500-step runs
  wrote no resume bundle, final-step skip). The 500-step smoke already went
  decisively to dynamic (see Phase 0), but a single early checkpoint is never
  a verdict ([[project_turbo_T_sweep_verdict]]: the 1k ckpt of T2 was voided
  and healed by 2k).

## What CDM is

**Continuous-Time Distribution Matching** (arXiv:2605.06376, Liu et al.,
Nankai/Alibaba, May 2026; code `github.com/byliutao/cdm`) is the direct
successor to D-DMD (arXiv:2511.22677, *"CFG augmentation as the spear,
distribution matching as the shield"* — the CA-DM decomposition our
`teacher_cfg` knob was originally introduced for). It keeps the decomposition
and makes three moves:

1. **Dynamic continuous schedule** (§3.2). Vanilla DMD2/D-DMD backward-simulate
   on the *fixed inference grid*, so the student's velocity field is only ever
   supervised at N discrete t's. CDM re-samples the grid every iteration:
   length N ~ U{1..N_max}, anchors a strictly-decreasing continuous sequence
   with t₁ = 1. Their empirical claim: distribution matching is
   *schedule-independent*, and the fixed grid was a restrictive constraint,
   not a necessary anchor — dynamic scheduling alone lifts HPSv3 10.08 → 10.65
   on Longcat (their Fig. 2), with finer detail and fewer artifacts.
2. **Role decoupling finding** (§1, Fig. 3). A student distilled with the DM
   loss alone converges to the teacher's **CFG-free** distribution; the CA term
   (the CFG-direction delta of the *real* teacher) is what carries text-image
   alignment. Neither is a "regularizer" for the other — they own different
   axes and want independent weights and independent τ draws.
3. **The L_CDM loss** (§3.3) — the actual novelty. From an on-trajectory latent
   x_t and its predicted velocity v, Euler-extrapolate a *large random stride*
   to an off-trajectory point x_{t'} = x_t + (t' − t)·v, t' ~ U(0,1). Supervise
   the student's prediction *there* with a real-vs-fake delta anchored to the
   **local** clean estimate x̂₀^{(t')} (re-noised to a fresh τ̂). This penalizes
   invalid velocity off the ideal manifold — exactly the truncation-drift
   region few-step Euler actually traverses. Ablations: velocity-driven
   extrapolation > Gaussian re-noising; local x̂₀ target > full-trajectory x̂₀.

Full objective L = L_CA + L_DM + L_CDM. Result: 4-NFE SOTA on SD3-Medium and
Longcat-Image, beating DMD2 and D-DMD on every metric, **with no GAN and no
reward model**.

## Why this line, here

- **It's the one open lever.** [[project_superturbo_nfe2_line]] and
  [[project_turbo_R_plateau]] both closed with the same verdict:
  init/LR/schedule/rank exhausted, *objective* is the only lever left
  (js/sf, identity aux, per_step_expert were the listed candidates). CDM is an
  objective-side change with a real ablation table behind it.
- **It attacks our two known failure axes.**
  [[project_turbo_teacher_gap_2026_06_29]]: pose-diversity collapse (CDM's
  mode-seeking mitigation) + text lost (CDM's CA-owns-alignment finding — see
  Phase 3).
- **GAN removal is a feature for us.** The GAN is at a spent plateau
  ([[project_turbo_R_plateau]]) and carries real plumbing hazards
  ([[project_turbo_view_ckpt_recompute_hazard]]: `--grad_ckpt`+GAN broken;
  half-depth feature forwards for VRAM). If L_CDM substitutes for what the GAN
  was buying, the loop gets simpler *and* cheaper.
- **Our fused real score has no independent alignment knob.** Our DM real
  score is CFG-guided at `teacher_cfg=4` (`_teacher_cfg_velocity`), i.e. our
  "DM" is an implicitly-weighted CA+DM with the ratio tied to α. The paper's
  decoupling gives alignment pressure its own weight and its own τ draw.
- **Phase 0 validated the line's core thesis** (added 2026-07-17, post-smoke).
  The smoke A/B showed empirically that the student's field is only good where
  supervision lands, *actively drifts* where it doesn't, and the sampler
  visits the drift region — proven on the t-axis, at a 500-step dose, visible
  to the naked eye. L_CDM is the identical argument on the x-axis: dynamic
  rollouts still only visit self-trajectory states, while few-step Euler's
  large strides traverse off-manifold points that remain unsupervised after
  Phase 0. Circumstantial pointer: the dynamic arm's one residual defect at
  NFE=2 is text garble while image quality holds — the largest-stride,
  maximal-truncation-drift regime, i.e. exactly where L_CDM aims.

### Map: paper term → our loop

| Paper | Ours today | Gap |
|---|---|---|
| Backward simulation on fixed grid | student rollout on `student_sigmas` | closed by Phase 0 (`dynamic_schedule`) |
| Continuous teacher perturbation τ | `tau_dm` already uniform-continuous | none |
| Gradient over all grid points | `grad_step="random"` (routing only — grid stays fixed) | routing ≠ schedule; Phase 0 adds the schedule |
| L_CA (real cond vs real uncond, weight α, own τ) | fused into the DM real score via `teacher_cfg` | Phase 3 splits it |
| L_DM (real cond vs fake cond, own τ̃) | `delta_dm` (but real side CFG'd) | Phase 3 |
| L_CDM (off-trajectory) | — | Phase 1 |
| GAN / reward aux | teacher-feature GAN + f-distill | Phase 2 removes if L_CDM covers it |
| Diversity mechanism | DP first-step teacher anchor | **kept — ours is stronger** ([[project_official_turbo_v10_eval]]: V10 mode-collapsed vs ours). CDM has no equivalent; never trade the anchor for L_CDM. |

## Phase 0 — dynamic schedule A/B (500-step smoke: decisive dynamic win)

One variable. Arm A = `dynamic_schedule=false` (legacy fixed grid), Arm B =
`true`. Same V10 warm start, same GAN, same `div_weight`, same seed/data. Arm B
is ~25% cheaper per iteration (mean rollout length 3 vs 4).

### Smoke read — 2026-07-17, 500 steps, seed 42, V10 adaln warm start

Arms verified via `ss_turbo_dynamic_schedule` ckpt metadata (the first attempt
trained two identical fixed-grid arms — a `[network]`-vs-`[dmd]` sectioned-
config misplacement; those grids survive as the matched-config noise floor).
**Dynamic won every axis, and not narrowly:**

- *NFE=4* (12 matched pairs, 3 artist styles): dynamic keeps saturation,
  background detail (waves / sand / foliage / signatures) and per-style
  separation; the fixed-grid arm is systematically washed out across all 12
  pairs — well above the noise floor, which flips content (bikini colors,
  bubble garble), never global saturation.
- *Off-grid ladder (NFE=3/2)*: the fixed arm craters exactly as grid-locking
  predicts — fog, translucent bubbles, Japanese glyph soup on @sincos.
  Dynamic degrades gracefully: at NFE=2 image quality is nearly intact, only
  the glyphs garble. (NFE=2 pairs were artist-mismatched — qualitative only.)
- *Glyphs*: dynamic ≥ fixed at every NFE. *Diversity*: seed spread comparable,
  no collapse signature.

**Mechanism read**: the delta is less "dynamic improved" than **"fixed-grid
training actively degrades the field off its 4 supervised t's"** — shared
LoRA weights mean on-grid updates are unconstrained everywhere else. The eval
sampler amplified this honestly: `er_sde_cns`'s σ grid ≠ the flow_shift-3.0
training grid, so even NFE=4 is slightly off-grid for arm A. That mismatch is
the deployment condition, not a bench artifact — Comfy users run arbitrary
samplers/schedules, so schedule-independence is real-world robustness.

**Still pending for the verdict**: the 2k read below — 500 steps / one
training seed per arm is a smoke peek. The step-trend to watch: if arm A's
off-grid degradation deepens 500 → 2000, the mechanism read is confirmed
directly. Render gaps to fill next pass: dynamic @channel NFE=3, and
matched-artist NFE=2 pairs.

**Protocol**: 2k steps, ckpts at 500/1000/2000, compared at matched steps.
- *Primary*: rendered 4-step grids at `--cfg 1.0` (`make gen`), fixed prompt
  set × seed sweep, human A/B. No paired-holdout CMMD gating at this n
  ([[project_seed_floor_cmmd_fragile]]); within-run CMMD as a trend line only.
- *Mechanism*: the **off-grid NFE ladder** — render both arms at
  `--infer_steps 2 3 6 8` alongside 4. Arm A is grid-locked and should crater
  off-grid; if CDM's mechanism is real, Arm B degrades gracefully. This is
  readable even if the NFE=4 A/B ties.
- *Diversity*: seed-grid spread per prompt + `val/div_*` — a schedule change
  must not regress what the DP anchor bought.
- *Text*: glyph probe (`bench/turbo/`) — regression watch only.

**Gate to Phase 1**: B ≥ A on rendered NFE=4 at 2k AND clearly better on the
off-grid ladder AND no diversity/glyph regression.
**Kill**: B ties at NFE=4 and shows nothing off-grid → the schedule axis is
dead on our teacher; decide whether L_CDM alone merits its own Phase 0 or the
line closes (flip the TOML back to `false` either way if B loses).

## Phase 1 — L_CDM off-trajectory loss (gated on Phase 0)

The payload. Wiring sketch, one new branch in the student update, reusing the
step-g machinery:

1. After the grad step g produces `v_g` at σ_g: draw t' ~ U(0,1) (CPU RNG),
   extrapolate `x_off = x_g + (t' − σ_g) · v_g` — same Euler form as the
   rollout (our grids decrease, so this is the paper's Eq. 7 verbatim).
   **Detach the extrapolation input** (fresh leaf, mirroring the DM renoise
   path) — memory-flat, no second BPTT chain; note this is a deliberate
   deviation to keep the loop's one-grad-forward budget per branch.
2. One grad-bearing student forward at (x_off, t') → local prediction
   `x0_off = x_off − t' · v_off` (paper Eq. 1 in our v-param).
3. Re-noise `sg[x0_off]` to a fresh τ̂ (existing `renoise`), then the
   real-vs-fake delta at τ̂ — **variant A** uses our CFG'd real score
   (`_teacher_cfg_velocity`, consistent with our fused DM), **variant B** uses
   cond-only real (paper-faithful; only meaningful after Phase 3 splits CA
   out). Start with A: one variable at a time.
4. Loss = MSE(x0_off, sg[x0_off + w·Δ]) with the same `dm_x0_norm` policy as
   `grad_dm`; weight `cdm.weight` (new `[cdm]` TOML table, default 0 = off).

Cost: +1 student grad forward, +1–2 teacher no-grad, +1 fake no-grad per
iteration (~+40–50% step time with GAN still on — Phase 2 claws it back).
Compile: t' is a tensor input, no shape change, no new graphs; the dynamic
schedule already de-keyed everything from grid membership. Student forwards at
arbitrary t are exactly what Phase 0 normalized.

**Gate**: rendered win at NFE=4 or on the ladder, diversity intact.
**Kill**: no rendered delta at matched steps → L_CDM doesn't transfer to our
teacher/domain; keep dynamic schedule (if Phase 0 won), close the rest.

## Phase 2 — GAN-off arm (gated on Phase 1)

CDM's headline claim is that continuous supervision replaces the adversarial
patch. Arm: Phase-1 winner with `gan.weight_gen = 0` (the existing kill switch
— disc never constructed, byte-identical DP-DMD+CDM). Compare against the
GAN-on twin at matched steps. If it holds: delete a two-optimizer moving part,
reclaim the disc VRAM + the grad-bearing teacher feature forward, and the
`--grad_ckpt`+GAN hazard class goes away entirely.

**Kill**: GAN-off regresses texture/sharpness (what the GAN historically
bought) → keep GAN, CDM stays additive.

## Phase 3 — CA/DM decoupling (optional, independent)

Split `_teacher_cfg_velocity`'s fused score into the paper's Eq. 3/4 form:
- **L_CA**: real-cond vs real-uncond delta, scale α = `teacher_cfg`, own τ
  draw, own weight — the explicit text-alignment knob.
- **L_DM**: real-cond vs fake-cond, own τ̃ — pure distribution pressure.

Same forward count as today (the CFG'd score already pays for cond + uncond +
fake). Motivation: the "text lost" axis of the teacher gap gets its own lever
instead of riding α; also unlocks paper-faithful variant B of L_CDM. This
phase is orthogonal to Phases 1–2 and can run as its own A/B whenever the
text axis becomes the binding constraint. Caution: the fused form's implicit
CA:DM ratio is what all shipped checkpoints trained under — treat the split
as a re-tune of both weights, not a free refactor.

## Standing eval guards (all phases)

Rank by rendered 4-step at `--cfg 1.0` ([[project_turbo_lr_instability_threshold]]);
no paired-holdout CMMD gates at n≈24/96 ([[project_seed_floor_cmmd_fragile]]);
diversity is non-negotiable (the DP anchor is the moat —
[[project_official_turbo_v10_eval]]); warm-start stays fixed across arms
([[project_turbo_warmstart_scope]]: init picks the mode — never let an
objective A/B double as an init A/B).
