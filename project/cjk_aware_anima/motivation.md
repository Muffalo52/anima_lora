# CJK-aware Anima — motivation

*Line home: [`done.md`](done.md) (what is measured and built) ·
[`plan.md`](plan.md) (what remains, phases, gates).*

## The problem

The LLM Adapter's T5-side query stream is the only thing standing between
Anima and native Japanese prompts. Anima conditions on two streams: Qwen3
hidden states (the content channel) and a T5-id query table that
cross-attends into them. The Qwen3 side handles JA fine. The T5 side is a
stock English SentencePiece — fed native JA it collapses to
`▁ <unk> </s>`, and conditioning dies with it (cos ~0.02 vs the all-EN
reference, discrimination 0.91 = the pathway carries no prompt-specific
signal at all).

Feed that same T5 side an **English translation** of the JA prompt and
conditioning recovers to cos **0.69–0.76** with healthy discrimination
(~0.11). So the T5 stream doesn't need to be *Japanese* — it needs to be
*semantic*. That asymmetry is the whole opening: the broken piece is small,
isolated, and has a working reference to imitate. Full measurements:
`bench/cjk_adapter/results/20260815-1836/result.json`.

**Scope is Japanese only for now.** The extended vocab and the probe cover
ko/zh too, but data building, distillation and gates are ja-first — zh/ko
become "rerun the same pipeline with a different corpus" once ja passes.

## The shape of the fix

Give the T5 side real ids for CJK characters (an extended vocab, built by
borrowing Qwen's CJK tokens and mapping them into the T5 embedding space),
then **distill** those new rows so the student `(Qwen ja, T5-ext ja)`
matches the teacher `(Qwen ja, T5 en-translation)` at the adapter output.

Two properties make this cheap rather than a model fork:

- The two arms **share the Qwen side**, so the loss isolates exactly the
  broken piece.
- The trainable surface is **only the new rows** — the original 32,128 rows
  and the EN tokenize path are untouched, so **EN prompts are bit-identical
  by construction**.

And the data question dissolves, because the teacher is computed at
embedding level: we don't need human parallel corpora, we need **JA
caption-domain text + any decent local MT model**. MT quality only has to
preserve meaning.

Design details are in [`plan.md`](plan.md); what has already been measured
and built is in [`done.md`](done.md).

## Why distill instead of shipping an inference-time MT shim

`t5en` works today, so "run local MT on the prompt, feed the translation to
the T5 side" is the zero-training baseline (B0) — and for **offline** uses
(TE-caching JA captions at preprocess time) it is genuinely viable. It is not
the destination because:

1. **Inference UX**: it puts an MT model in the live inference path (VRAM +
   latency + a ComfyUI node dependency). The real user base is Arca Live
   (KR) / CN / JP / Civitai — native prompting has to work in a stock
   pipeline.
2. **Verbatim text identity**: translation destroys quoted strings
   (`「おはよう」` → `"GOOD MORNING"`). The OCR-caption and text-render lines
   need the exact JA string to survive into conditioning; only the ext vocab
   provides per-character identity.
3. **Ceiling**: t5en itself sits at 0.69–0.76, not 1.0 — the distilled
   student targets the teacher, and can in principle be *improved past* the
   teacher later (image-level training, Phase 4).

B0 stays in every eval grid as the bar to match.

## What this unlocks downstream

Deferred, but the reason this is worth more than an inference-time MT shim:
each JA character gets a **unique, stable T5-side id** instead of `<unk>`,
which is the prerequisite for

- appending OCR'd in-image text to captions instead of masking it, and
- an eventual JA text-rendering phase.

Neither is reachable through a translation shim, which destroys the verbatim
string. Both live in Phase 4 ([`plan.md`](plan.md#phases--gates)), explicitly
deferred and explicitly ordered *after* the vocab exists.

## Directions already ruled out

Closed with measured evidence — read before reopening any of them. Probe arms
are in `bench/cjk_adapter/results/20260815-1836/result.json`; the data-side
verdicts are argued in [`datasets/README.md`](datasets/README.md).

- **Romanization** (any transliteration scheme) — `t5rom` cos ~0.06. Non-unk
  is not enough; the T5 stream must be semantic.
- **Reverse routing** (`q_en`: EN on Qwen, JA on T5) — cos ~0.07/0.15,
  discrimination 0.92. The Qwen side is the content channel.
- **Cleverer zero-shot maps** (procrustes, per-block, nonlinear) — not worth
  another dry run: a 0.75 anchor holdout cos still yielded ~0.05 end-to-end,
  so the gap is contextual, not linear-map quality. This is what makes
  training necessary.
- **A fresh SentencePiece over T5** — borrowing Qwen tokenization keeps the
  source and query streams token-aligned over CJK spans; a new spiece would
  destroy that structural property.
- **Fixing tag register with a better prompt or a bigger MT model** — few-shot
  moved the 1.8B 8/28 → 9/28; the 7B reaches 15/28. The misses are knowledge
  of what the community calls things, so the fix is a source that has it (the
  Danbooru wiki), not more MT.
- **Translating captions whole** — superseded by composing them from the tag
  glossary: proper-noun pinning and idiom become true by construction, and
  each tag is translated once instead of once per occurrence.
- **Leaving danbooru tags latin** as the cheap alternative — strictly worse,
  not equivalent: latin tags route to the original spiece rows, training *no*
  ext rows at all.
- **Unguarded flat Wikidata matching for single-token given names** (`aru`,
  `ann`) — returns railway stations and news agencies. Not a refutation of
  flat matching in general: with ≥2 tokens plus a `P31/P279* Q95074` guard it
  is the route that works.
- **Expecting Wikidata to cover artists** — 0/89; pixiv/danbooru handles are
  not encyclopedia entities. Artist names go through D2 metadata.
- **Bake-in** (expanding `embed.weight` inside a forked DiT checkpoint) — see
  [`plan.md`](plan.md#deployment--a-vocab-pack-not-a-lora).

Two invariants any future change must preserve: **verified ≠ Japanese**
(棕毛, 藍眼睛, 汗液 all back-translate perfectly and are Chinese — keep both
script filters and the kana-first ranking), and **selection is pure
post-processing** (`tag_glossary.py --reselect`, ~1 s, never a re-translation).
