# directedit_ec — roadmap

Status: Phases 0, 1a, 1b all PASSED (2026-07-24, zero training). The shipped
recipe (cond hole + anchor mask, b_offset 0) beats V-injection on every
in-place edit type. **Phase 2 is unblocked** and is the next work item.

## Phase 2 — cross-image subject descriptor (one standard EasyControl train)

**Status: PREPARED (2026-07-24)** — descriptor + miner + task wiring shipped
and staging validated; the remaining work is the train itself + the gate
bench. Surface: `configs/easycontrol/subject.toml` (descriptor with knobs +
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
  `b_cond_init = -8` (softer learned operating point than inpaint's all-in −6).
- **Text**: full captions of the *target* — prompt keeps owning layout/pose,
  cond owns identity/appearance only.
- Cost: inpaint-recipe scale (8 epochs over the 1.1k pair set ≈ inpaint's
  4 epochs over the 3k corpus in optimizer steps; 16 GiB-friendly).
- **Gate** (re-run the existing harness:
  `project/directedit_ec/bench/run_bench.py --ec_weight
  output/ckpt/anima_easycontrol_subject.safetensors`):
  (a) sweet-spot width ≥ 2 b_offset units (inpaint: ~1) — answers
  questions.md Q1; (b) ≥ parity with vinj_t6 on the 1b geometry edit —
  answers Q2, the associative-retrieval claim.

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
