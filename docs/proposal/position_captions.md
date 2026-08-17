# Position-aware captions — natural caption enhance → smart caption rewrite

Status: **PROPOSAL — Phase 0 (feasibility probes) DONE, both green.** Probes +
result envelopes live in `bench/position_captions/` (runs `20260817-1122
-autocaption`, `20260817-1123-binding`). No production code written yet.

## TL;DR

Two staged captioning features that give multi-subject images spatially bound
captions in the dataset's existing hand-written convention
(`On the left, <tags>. On the middle, <tags>. …`):

1. **Natural caption enhance** (v1, additive): for captions carrying
   `multiple views` or a girls-count > 1, detect `girl` instances with SAM3,
   order them left→right, tag each mask-blanked crop with the Anima Tagger,
   and **append** positional clauses to the caption. The flat tag bag is
   untouched.
2. **Smart caption rewrite** (v2, advanced): additionally **move** each tag
   that is attributable to exactly one instance out of the flat bag into its
   positional clause, so the duplication between bag and clause is resolved
   and the flat bag stops asserting unbound attributes.

## Why

The flat tag bag cannot express which attribute belongs to which subject:

- `channel_(caststation)/channel6` — `3girls, blonde hair, aqua hair, red
  hair, …`: who is blonde? (This image already has hand-written clauses —
  they are the convention v1 automates.)
- `channel_(caststation)/8090164` — `1girl, multiple views, maid, playboy
  bunny, swimsuit, pink jacket, …`: four outfit views of one character, all
  outfits unbound. Same ambiguity, same fix, and the gate must therefore be
  **detected instance count**, not the girls-count tag.

Populations at time of writing: **258** multi-girl captions (12 with
hand-written clauses = ground truth), **~350** `multiple views` captions.

## Phase 0 evidence (both probes green)

**Probe A — the base model already obeys positional clauses.**
`probe_binding.py` rendered 24 no-LoRA images from counterbalanced prompts
(`…, {a} hair, {b} hair. On the left, {a} hair. On the right, {b} hair.`,
4 color pairs × both orders × 3 seeds), split each render into halves, and
asked the tagger which color won per side: **48/48 sides correct (chance
50%)**. Spot-checked renders confirm honest wins (two girls, correct color on
the prompted side, either order). Positional captions therefore *reinforce an
existing base-model capability* — there is no teach-from-scratch risk, and
LoRA training on such captions inherits a working conditioning channel.

**Probe B — the detection→crop→tag pipeline works on first contact.**
`probe_autocaption.py` on the 12 ground-truth images + the 8090164 showcase:

- Instance count vs hand-written clause count: **10/12**. Misses: `chicke2`
  (a two-view sheet whose left view is an extreme close-up SAM scores < 0.5
  for `girl`) and `4615461` (2 of 3 found).
- Hair-color-to-position: **8/10**. Both misses are *neighbor hair intruding
  into the bbox crop* (plus genuinely borderline colors: near-black vs brown,
  pale pink vs white).
- Character-name-to-position: 4/7 — names are the weakest signal on crops.
- The 8090164 proposal came out exactly right: maid → bunny leotard → pink
  jacket → one-piece swimsuit, left to right, from a `1girl` image.

The two systematic errors have known mechanical fixes, wired into v1 below:
mask-blanking (SAM already returns per-instance masks; blank non-instance
pixels before tagging) kills crop contamination, and a lower score threshold
with a count-consistency check recovers weak detections.

## Feature 1 — natural caption enhance (v1)

Pipeline per image (candidates prefiltered by caption: `multiple views` OR
girls-count tag > 1; skip any caption that already contains `On the `):

1. **Detect**: SAM3 `girl` on the resized image → per-instance boxes + masks
   + scores (`Sam3Processor.set_text_prompt`, same API as
   `scripts/preprocess/generate_masks.py`). Score filter (default 0.5, retry
   pass at 0.3 when detected count < caption girls-count), greedy IoU dedupe
   (0.65).
2. **Order**: left→right by box center-x. Position vocabulary: N=2 →
   `left/right`; N=3 → `left/middle/right`; N≥4 → `leftmost / second from
   left / … / rightmost`. **Row-aware extension for grid sheets**: cluster
   center-y first; if >1 row, prefix `top`/`bottom` (`on the top left, …`) —
   multiple-views sheets are often 2×2, and pure x-ordering interleaves rows.
3. **Crop + blank**: padded bbox crop (6%), non-instance pixels blanked to
   white using the instance mask (the probe-B contamination fix).
4. **Tag**: Anima Tagger per crop. Clause tags = character name (only if kept
   with high confidence — probe B says names are weak), hair-color group
   winner (fall back to top kept hair tag — the group is `softmax_when_solo`
   and stays silent on multi-person crops), eye color, hair-shape tags,
   outfit tags. Cap ~8 tags/clause.
5. **Append**: `<flat caption>. On the left, <tags>. On the right, <tags>.`
   Written to the caption master (`image_dataset/*.txt`) and mirrored to
   `post_image_dataset/resized/*.txt` the same way existing caption tooling
   does.

Gate on **detected** count ≥ 2. If detection and the girls-count tag disagree
after the retry pass (probe-B: 2/13), *skip and log* — never write clauses we
can't ground.

**Ops sequencing (both are silent-failure traps):** caption edits do NOT
invalidate TE caches — the run must end with an explicit `make preprocess-te`;
and `*.variants.txt` tag-dropout sidecars OVERRIDE the CLI rate, so variants
must be regenerated after the rewrite (decision needed on clause dropout
semantics — see open questions).

**Code layout** (per the layering contract): orchestration in
`library/preprocess/position_captions.py`, thin argparse shell
`scripts/preprocess/position_captions.py`, `make caption-position` target,
daemon-routed (GPU: SAM3 + tagger). `--dry-run` is the default: emit a review
artifact (proposed clause per image + crops) without writing any caption;
`--apply` writes.

## Feature 2 — smart caption rewrite (v2)

v1 leaves each bound attribute in two places (flat bag + clause). v2 resolves
the duplication by **moving** tags into clauses:

- A flat tag moves into a clause when the tagger keeps it on **exactly one**
  crop (with margin) — i.e. it is attributable. `blonde hair` kept only on
  the left crop → leaves the bag, lives in `On the left, …`.
- Tags kept on multiple crops (shared uniform, `smile`), scene/meta tags,
  count/rating/artist/copyright, and anything unattributable stay flat.
- Result for channel6-style images: `3girls, <scene tags>. On the left,
  akita neru, blonde hair, …` — each attribute asserted exactly once, bound.

This is the riskier half: it changes the flat-bag token distribution the base
model was pretrained on (probe A validated clause *comprehension*, not
bag-*removal* tolerance). Ship it behind a flag, phase-gated on a training
A/B (Phase 3), never as a default until the A/B clears.

## Phases

- **Phase 0 — probes: DONE** (numbers above).
- **Phase 1 — v1 implementation + dry-run sweep**: build the pipeline, run
  `--dry-run` over all ~600 candidates, spot-check ~30 proposed clauses
  (weighted toward grids, overlapping pairs, N≥4). Exit: clause proposals
  look right at ≥90% on the spot-check; count-disagreement skip rate
  reported.
- **Phase 2 — apply + train A/B**: `--apply`, regenerate variants + TE
  caches, train the standard LoRA recipe on a multi-girl-dense slice, twin
  control on unmodified captions (same seed — mind the paired-ΔW chaos
  floor: compare renders, not raw ΔW cosines). Eval: probe-A-style render
  test through the trained LoRA (does binding hold under the artist style?)
  + multi-character sample grids. Exit: binding ≥ control, no regression on
  single-girl renders.
- **Phase 3 — v2 behind `--rewrite`**: same A/B protocol, v2 vs v1 corpus.

## Open questions

1. **TE token budget**: appended clauses lengthen captions; verify the long
   tail stays inside the text-encoder `max_length` (the padding invariant
   means overflow truncates silently — count tokens in the dry-run report).
2. **Clause dropout semantics** in `variants.txt`: drop the whole clause as a
   unit (recommended — a positionless attribute fragment is exactly the
   ambiguity we're removing), or never drop clauses?
3. **Character names in clauses**: include only above a confidence floor, or
   leave names to the flat bag entirely (probe B: 4/7)?
4. **Boys/POV**: prompt is `girl` only, matching the hand-written convention.
   Extend to `person` for the rare on-screen-boy image, or leave out of v1?
5. **Vertical layouts**: row-clustering covers grids; tall single-column
   stacks (rare) would need `top/bottom` vocabulary — punt until seen.
