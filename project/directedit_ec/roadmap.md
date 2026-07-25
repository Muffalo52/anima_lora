# directedit_ec — roadmap

Status: Phases 0, 1a, 1b all PASSED (2026-07-24, zero training). The shipped
recipe (cond hole + anchor mask, b_offset 0) beats V-injection on every
in-place edit type. **Phase 2 arm 1 ran 2026-07-25 and both gates failed — but
on a run that never exercised the hypothesis** (see below). Next work item is
Phase 2 arm 2, a single retrain with the cond gate open.

## Phase 2 — cross-image subject descriptor (one standard EasyControl train)

**Status: arm 1 RUN, gates FAILED, hypothesis UNTESTED (2026-07-25).** The
train itself was clean (8928 steps, loss 0.0797, pair data verified), but
`b_cond_init=-8` with `cond_res_scale=0.5` left the cond stream at ~8.4e-5
attention mass for all 8928 steps, and **`b_cond` does not learn** (saved
exactly at init, as inpaint's did). So the cond path got almost no gradient and
cross-image pairing was never actually exercised. Full data:
`bench/report.md#phase-2`.

- Gate (a) sweet-spot width: **FAIL** — 0 usable units (preserve *and* land the
  edit) vs inpaint's ~1. The preservation band exists but is displaced ~7
  offset units (+6…+8) and the edit is suppressed throughout it.
- Gate (b) geometry parity: **FAIL to demonstrate** — EC ties `vinj_t6` only by
  both failing to land the pose.
- Decisive: `bench/run_subject_probe.py` (DirectEdit-free, cond = image A,
  prompt = caption of image B) shows **no identity transfer at any offset** —
  inert at the trained point, image degradation at +6/+7/+8. What engages at
  +7/+8 is the *architectural* aligned-cond copy path, not learned retrieval.

**Do NOT invoke the kill criterion on this run** — it does not discriminate the
hypothesis. Arm 2 first: retrain with the gate open (`b_cond_init ≈ -2`, mass
3.3e-2, and/or `cond_res_scale=1.0`), same cost (~1h45m), then re-run the same
three gate benches. If arm 2 also shows no retrieval, the kill criterion applies
with evidence behind it.

Surface as shipped: `configs/easycontrol/subject.toml` (descriptor with knobs +
generated blueprint tail), `easycontrol_adapters/tools/subject_pairs.py`
(the miner — near_twins contract, CPU-only), `EASYADAPTER=subject` registered
in `scripts/tasks/training.py`. First mining run: **1116 pairs over 283
characters, 813 (73%) cross-artist** (solo 1girl/1boy single-character only,
cap 16 targets/character, seed 42; manifest at
`post_image_dataset/easycontrol/subject/pairs.json`).

```bash
make easycontrol-staging    EASYADAPTER=subject   # mine pairs → staging/ + cond/ (CPU, done)
make easycontrol-preprocess EASYADAPTER=subject   # rebuild cond/ only (after corpus re-preprocess)
make easycontrol            EASYADAPTER=subject   # the Phase-2 train (--queue for daemon)
```

Design (as proposed, now implemented):

- **Pairs**: cond = image A of a character, target = image B of the same
  character, mined from `caption_index.json` character tags. Staging emits a
  pair manifest; cond + target latents/TE reuse the shared LoRA cache — no
  synthetic tree and **no encode pass** (both steps are pure symlinks;
  cheapest descriptor staging yet). The same-artist + shared-tags fallback
  was skipped — character tags alone cover ~1.1k targets. Each target's cond
  partner prefers a different artist dir (starves the style shortcut too).
- **Anti-shortcut knobs** (in `[training]`): `cond_res_scale = 0.5` (starves
  the positional shortcut, and is faster); `easycontrol_drop_p = 0.05`;
  `b_cond_init = -8`. **Arm 1 showed these two compose into a trap**: they
  multiply into ~8.4e-5 cond attention mass (0.25·e⁻⁸), 29.5× below inpaint's
  2.5e-3, and `b_cond` does not self-correct — it stays at init. The
  anti-shortcut intent was right, the dose closed the mechanism. `b_cond_init`
  is NOT a "softer learned operating point"; it is a fixed hyperparameter.
- **Text**: full captions of the *target* — prompt keeps owning layout/pose,
  cond owns identity/appearance only.
- Cost: inpaint-recipe scale (8 epochs over the 1.1k pair set ≈ inpaint's
  4 epochs over the 3k corpus in optimizer steps; 16 GiB-friendly).
- **Gate** — three benches, all re-runnable against any future arm:

```bash
W=output/ckpt/anima_easycontrol_subject.safetensors
# (a) b_offset band: sweep WIDE — the band's location depends on b_cond_init,
#     and the arm-1 band sat at +6..+8, outside every inpaint-era offset.
python project/directedit_ec/bench/run_bench.py --ec_weight $W --ec_scales 1.0 \
    --ec_b_offsets "-2,-1,1,2,4,6,8"
# (b) geometry / associative retrieval (whole cond, anchor released)
python project/directedit_ec/bench/run_bench.py --phase 2 --ec_weight $W \
    --ec_b_offsets "<engaged offsets from (a)>"
# (c) DECISIVE: retrieval with no DirectEdit in the loop
python project/directedit_ec/bench/run_subject_probe.py --n_pairs 3 \
    --b_offsets "<engaged offsets from (a)>"
```

  (a) sweet-spot width ≥ 2 b_offset units where the edit *both* preserves and
  lands (inpaint: ~1) — answers Q1; (b) ≥ parity with vinj_t6 on the geometry
  edit — answers Q2. **Run (c) first on any new arm**: it is cheap, has no
  composition confound, and a null there makes (a)/(b) uninterpretable.

## Phase 3 — feed-forward editor (endgame, gated on Phase 2)

Distill DirectEdit itself: synthesize `(source, edited, edit-caption)` pairs
at scale with the Phase-1b recipe + tagger readback as the label filter,
train an EasyControl editor descriptor (cond = source, target = edited).
Inference becomes one cached-cond generation — no inversion, no anchor, no
patching. InstructPix2Pix's recipe with our tag vocabulary and a
training-free teacher. **Separate proposal when reached**; only if Phase 2
shows the teacher reliable enough to label its own data (the hard-image
ceiling, questions.md Q3, bounds this).

## Parallel / opportunistic

- Wire tag-readback edit-success into `run_bench.py` once a trained tagger
  checkpoint exists (Q6) — retro-scores existing result dirs too.
- Swap the manual hole box for the cfgdelta subject localizer (Q5) — small
  edit.py flag, benchable on the existing 1a set.
- Paper go/no-go **after** the Phase-2 gate (Q7): if written, the matched-NFE
  baseline table is the first work item, not the last.

## Kill criteria

- Phase 2 sweet-spot width ≤ inpaint's AND geometry parity fails → pairing
  wasn't the constraint; close this proposal's training arc, keep the
  zero-training recipe as the shipped artifact, spin the gate-schedule idea
  (per-block/per-σ) as a separate smaller proposal only if demand exists.
- Phase 3 never starts unless Phase 2's adapter demonstrably widens the
  operating band — a feed-forward editor distilled from a cliff-shaped
  teacher inherits the cliff.

## Relation to BYG

BYG (demoted 2026-07-02, validation-gated) keeps its niche: free-form
*instruction* edits beyond tag space. This line's Phases 0–2 stay tag-edit
scoped. If BYG's Phase-0 gate ever passes, a BYG arm belongs in this bench.

## Pointers

Proposal: `project/directedit_ec/initial_proposal.md` · Data:
`project/directedit_ec/bench/report.md` · Memory: `project_directedit_ec_phase0` ·
Recipe + component map: `methods.md`.
