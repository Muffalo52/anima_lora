# Mask-quality survivor selection in detection dedupe

Status: **Phase 0 complete (2026-08-17); Phase 1 ready to implement; Phase 2
dead.** Evidence recorded in
[`docs/experimental/multiview_audit.md`](../experimental/multiview_audit.md)
§4–5.4 (root cause on `5847152`, NMS-pair replay over `ama_mitsuki`, batch mask
probe over the 10 smoke findings, 8-prompt sweep, full-corpus Phase 0). Phase 0
outcomes: gate 1 **passed** — the ratio axis has an empty band around 2.0 at
both floors, fixing **R = 2.0**; gate 2 **failed** — the shape-B fill gap
disappears at corpus scale, so the degenerate guard does not ship.

## TL;DR

`dedupe_detections` (`library/preprocess/position_captions.py:608`) is greedy
NMS ranked on **SAM3 score alone**. SAM3's score is box-level confidence and
says nothing about mask coherence, so when two proposals are duplicates of the
same object, a ~0.03 score edge decides which pixels every downstream consumer
sees. Two failure shapes, both measured:

- **A — wrong survivor.** A near-empty duplicate outscores the clean mask and
  suppresses it; the tagger then reads confident garbage off a white speckle
  crop (`5847152`: garbage 0.389/fill 0.077 beat clean 0.354/fill 0.560 →
  `extra-character` verdict on one girl drawn twice). 8 pathological pairs now
  measured across the prompt sweeps; fill *ratio* inside the matched pair
  separates them from benign inversions in every case.
- **B — phantom body.** A whole-canvas near-empty proposal that overlaps
  *nothing* at the IoU threshold survives as an extra instance (`5847182`:
  fill 0.001; under `girl` it missed the 0.35 floor by 0.018, under
  `woman`/`female`/`person` it clears the floor and is counted).

**The fix for A**: when NMS has already judged two proposals to be the same
object, choose the survivor by relative mask fill, not score. Shared code, so
it fixes `caption-position` and the multiview audit at once. **B** gets a
narrowly-scoped degenerate-input guard, gated on a corpus measurement (Phase 2)
because it brushes against a settled negative.

Prompt engineering is measured and dead as an alternative (§5.3): instruction
prompts fall off SAM3's grounding distribution entirely; longer noun phrases
collapse recall on exactly the close-up population the pipelines exist for;
`woman`/`female`/`person` make shape A *more* reliable (junk outscores the
clean mask by up to 0.184) and make shape B real. The prompt stays `girl`.

## The change

### A. Fill-ratio swap inside a matched pair (shared, `dedupe_detections`)

Ranking stays score-descending, collision stays IoU ≥ `iou_threshold`. What
changes is what happens on collision: instead of unconditionally dropping the
candidate, compare mask fill within each box's own bounds and **swap the
survivor** when the loser's mask is decisively better:

```
on collision of candidate c with kept box k (score(k) ≥ score(c) by ranking):
    if fill(c) / fill(k) ≥ R:   replace k with c      # swap, count unchanged
    else:                        drop c                # today's behaviour
```

Properties that make this safe:

- **Instance count is invariant by construction.** The rule can only change
  *which* of two already-matched duplicates represents the object — it can
  never add or drop an instance. Skips/proposals counts cannot move; only the
  pixels behind some crops can.
- **No absolute cut-point anywhere.** This is not the settled negative
  (`position_captions.md` — mask fill rejected as an *absolute* gate because a
  clean 0.87-score figure sits at fill 0.267). A ratio inside one pair never
  asks "is this fill low", only "is this duplicate's mask far better than the
  one that outscored it".
- **Masks optional.** `Detection.mask` is `None` for stub tests and for part
  boxes (which don't pass through `dedupe_detections` anyway —
  `merge_part_detections` is a separate path and is **not** touched); when
  either mask is missing, fall back to today's score-only drop.
- **Single-pass, deterministic.** The swapped-in box has different geometry
  and could in principle newly collide with a third kept box; we accept the
  single-pass result (bounded, order-stable) rather than iterating to a fixed
  point. If the corpus probe surfaces a real cascade case, revisit.

Implementation: a small `mask_box_fill(det) -> float | None` helper (binary
mask at 0.5, mean inside the clipped box — exactly what
`scripts/preprocess/probe_nms_pairs.py` computes today) + the swap branch in
`dedupe_detections`, knob `--dedupe_fill_ratio` (float; `0`/unset = off =
byte-identical current behaviour). Both the clause pipeline and the audit pick
it up through `detect_subjects` with no further wiring.

**Choosing R — settled by Phase 0 at R = 2.0** (audit §5.4). Over the full
3008-image corpus under the shipping `girl` prompt, every suppressed pair at
ratio ≥ 2.0 has a degenerate survivor (kept fill ≤ 0.149) and every pair below
2.0 has a clean one (kept fill ≥ 0.213), except one pair where both fills are
degenerate and no ratio rule helps. The ratio axis is *empty* on (1.87, 2.75)
at the retry floor and (1.75, 3.70) at the primary floor, so 2.0 sits in a
measured gap: {1.5, 2.0, 3.0} were swept and 1.5 adds 4 swaps of clean
survivors (risk, no measured benefit) while 3.0 misses 2 real pathological
pairs. R stays deliberately conservative: a missed swap is today's behaviour,
a wrong swap is a regression. The pre-Phase-0 evidence (pathological 7.25 /
3.54 / … / 1.50 across the prompt sweep, benign ≤ 1.33) pointed at the same
default; the 1.50-ish pathological pairs all came from non-shipping prompts.

### B. Degenerate-proposal guard (Phase 2, gated)

Shape B is not reachable by the pair rule (nothing collides). The candidate
guard: drop a proposal whose mask supports essentially none of its own box,
e.g. `area_frac ≥ 0.95 and fill < 0.10`. The argument that this is *not* the
settled negative: downstream, the mask is what blanks the crop — a fill≈0
proposal produces an **all-white crop by construction**, so this is a
degenerate-input guard, not a quality judgment. The rejected absolute gate was
litigated at fill ≈ 0.267 (clean figures live there); observed garbage sits at
0.000–0.077 and observed real whole-canvas-ish views at 0.148+ (`6494927`,
area 0.930) and 0.363 (`5847152` under an alt prompt).

But the margin (0.077 garbage vs 0.148 real) is thin and the sample is one
artist directory, so this ships **only** if the Phase-0 fill distribution over
the full corpus shows a clean gap — and if it doesn't, B stays handled the way
it is today (the score floor plus, in the audit, `count-explained` /
spot-checking). Under the shipping `girl` prompt B has never been *observed* to
fire, only counterfactually — which is why it is Phase 2, not Phase 1.

## Interaction with the interim mitigation

The audit currently sets `reliable=False` on every retry-recovered box
(score < `score_threshold`), silencing its identity vote — a symptom patch that
also silences retry boxes whose masks are fine (`multiview_audit.py:424`).
After A lands, the recovered clean mask makes the identity read *real* again on
the 5847152 shape. Keep `reliable=False` as-is in Phase 1 (it guards the
5847168 headless-crop shape, which is a different failure), but re-run the
smoke with A on and check whether any finding's witness set improves; loosening
it is a possible Phase-3 follow-up, not part of this proposal.

## Phases

**Phase 0 — corpus measurement (GPU, daemon-routed). DONE 2026-08-17.**
`probe_nms_pairs.py` over the full resized corpus (not just `ama_mitsuki`), at
both floors (0.5 primary, 0.35 retry), recording every suppressed pair's
scores + fills **plus** the fill/area distribution of *all* proposals (for B).
Gates: (1) benign-vs-pathological ratio separation holds at scale and fixes R;
(2) fill distribution answers whether B's gap is real. Also quantifies, for the
first time, how often shape A fires in `caption-position`'s population — the
"unmeasured" line in the audit doc.

*Results* (full numbers in audit §5.4; payload
`post_image_dataset/captions/nms_pairs_full.json`): gate 1 **passed**, R fixed
at **2.0** (empty ratio band around it at both floors; catches 13 of 14
degenerate-survivor pairs, the 14th being both-fills-degenerate). Gate 2
**failed** — whole-canvas fills form a continuum through the proposed cut, so
Phase 2 is dead. Shape-A frequency: 0.07% of images at floor 0.5, 0.37% at
floor 0.35.

**Phase 1 — the swap.**
- `mask_box_fill` helper + swap branch in `dedupe_detections`, knob
  `--dedupe_fill_ratio` on both CLI shells (audit + caption-position), default
  from Phase 0.
- Unit tests (`tests/test_position_captions.py`): swap happens on a ≥R pair;
  no swap below R; count invariant under swap; `mask=None` falls back to
  score-only; part-box path untouched.
- Regression: re-run the audit smoke (expect: `5847152` reaches
  `multiple views` with a *real* identity read instead of via the mitigation;
  all other verdicts unchanged) and a `caption-position` dry-run diff on
  `ama_mitsuki` (expect: proposals/skips identical, crops changed only on
  swap-affected images).

**Phase 2 — the degenerate guard. DEAD** — Phase 0's gate failed (no fill gap
at corpus scale; low-fill whole-canvas proposals with scores up to 0.79 sit on
a continuum with real sparse-subject views). B stays handled as today: the
score floor plus, in the audit, `count-explained` / spot-checking.

## Non-goals / settled things not reopened

- **No absolute mask-fill quality gate** (`position_captions.md` settled
  negative) — A is pair-relative; B is a degenerate-input guard with an
  explicit measurement gate and dies quietly if the gap isn't there.
- **Containment suppression stays off** (measured: breaks 34 rows to recover
  ~2). The 5847152 pair happens to be 0.991 contained, but that is not the
  lever.
- **No prompt change** — §5.3 measured every plausible variant.
- **No score-margin term in the rule** — margin measurably does not separate
  the pathology (§5.1: the 0.035 pathological margin is matched exactly by a
  benign pair).

## Evidence index

| Artifact | What it shows |
|---|---|
| `multiview_audit.md` §4 | Root cause on 5847152, verified end-to-end |
| §5.1 + `post_image_dataset/captions/nms_pairs_ama.json` | 7 suppressions / 106 images, ratio separates |
| §5.2 + `mask_probe/smoke_batch.json` | Garbage proposals in 3/10 findings; both counterfactuals; absolute-cut refutation re-confirmed (6494927) |
| §5.3 + `mask_probe/prompt_sweep*.json` | Prompt engineering measured and declined; 7 more pathological pairs; benign band tops at 1.33 |
| `mask_probe/<stem>/*.png` | Per-image overlays behind every number above |
| §5.4 + `post_image_dataset/captions/nms_pairs_full.json` | Phase 0 full corpus: R = 2.0 fixed (gate 1 pass), B gap refuted (gate 2 fail), shape-A frequency 0.07% / 0.37% |
