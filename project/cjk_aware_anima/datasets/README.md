# `datasets/` — corpus builders for the CJK distillation

Data-side tooling for Phase 2a of [`../plan.md`](../plan.md). This file is the
home of the measured results for everything below; [`../done.md`](../done.md)
only lists what is finished.
Everything writes into `assets/` (gitignored — regenerate rather than commit)
except `tag_overrides.json`, which is hand-authored and committed. Run from the
repo root. The two builders that need a GPU (`mt.py`, and the `--mt` passes)
must go through the daemon — `make daemon-run ARGS="<script> …"` — never a
background shell.

The distillation teacher is computed at **embedding level** from an EN string,
so nothing here needs human-quality parallel text; what it does need is *JA that
is right*, in the *registers users actually prompt in*, hitting *ext rows the
corpus will really visit*.

| Script | Source | Licence | Output |
|---|---|---|---|
| `wikidata_lexicon.py` | Wikidata SPARQL, entity list from our `caption_index.json` | CC0 | `assets/wikidata_lexicon.json` — EN↔JA(↔KO/ZH) proper nouns |
| `lovehina.py` | [PLippmann/multimodal-manga-translation](https://github.com/PLippmann/multimodal-manga-translation) (COLING 2025, arXiv:2411.02589) | MIT | `assets/lovehina_ja.json` — 3,705 native JA speech-bubble lines |
| `mt.py` | `tencent/Hy-MT2-1.8B` / `-7B` | Apache-2.0 | library (`MTEngine`) + `--probe` / `--smoke` self-checks |
| `tag_glossary.py` | Danbooru wiki dump + the two sources above + MT | WTFPL (dump) | `assets/tag_glossary_ja.json`, `assets/tag_glossary_review.md` |
| `build_pairs.py` | the glossary | — | `post_image_dataset/cjk_distill/{pairs.jsonl,coverage.json,spotcheck.md}` |

## Why MT is the *last* source, not the first

The obvious plan — "translate the captions with a local MT model" — fails on
register, and register is not cosmetic here: ext rows only train where the
corpus visits them, so JA that says 二本の髪 where users type ツインテール
trains the wrong rows. That is the proper-noun distribution argument
(below) applied to the whole general vocabulary.

Measured on a held-out 28-tag idiom probe (`mt.py --probe`, greedy):

| arm | hit rate |
|---|---|
| Hy-MT2-1.8B, plain prompt | 8/28 (29%) |
| Hy-MT2-1.8B + 16 few-shot exemplars | 9/28 (32%) |
| Hy-MT2-7B, plain prompt | 13/28 (46%) |
| Hy-MT2-7B + few-shot | 15/28 (54%) |

Prompting barely moves it and size only halves the gap: these are *knowledge*
misses (ツインテール, カメラ目線, 割座, セーラー服), not instruction-following
misses. **Do not re-propose fixing tag register with a better prompt or a
bigger MT model** — the fix is a source that already knows the community's
words, i.e. the Danbooru wiki.

The 7B needs `--gpu-budget 13GiB` to fit a 16 GB card (bf16 weights are 15.2 GB
against ~15.5 GB usable); accelerate then keeps the overflow blocks on CPU and
streams them, costing ~4.2 gen/s. Keep `--max-new-tokens` tight: a batch runs
until every sequence finishes, so the 512 default made tag translation ~14×
slower than necessary.

## `tag_glossary.py` — the EN→JA tag table

Sources in priority order, per tag: `tag_overrides.json` (hand fixes) →
artist passthrough (pixiv handles stay latin — users type them latin) → rating
band → Wikidata lexicon → Danbooru wiki `other_names` → MT.

`other_names` mixes languages, so script detection matters twice over. Kana
proves Japanese. Han-only strings are kept only if Shift-JIS can encode them,
which rejects simplified Chinese (小鸟游星野 out, 火宮チナツ in) — but *not*
traditional Chinese (棕毛, 藍眼睛 both encode fine and both scored a perfect
back-translation), so Han-only candidates are additionally checked against a
JA-kanji inventory learned from the dump itself: every `other_names` entry
containing kana is Japanese by construction, so the kanji inside those entries
are a free census of the JA repertoire.

The wiki is authoritative on idiom but *not* a synonym list — it also carries
narrower compounds (`underwear` → 下着コート) and different concepts
(`multiple girls` → 群像). So every candidate, **including the MT rendering**,
is back-translated to English and scored (token F1) against the original tag;
the best scorer above `--accept-f1` wins, and ties break toward brevity so
`黒髪ロング` loses to `黒髪`. Exemplars for the MT prompt come from a
hand-checked list in `mt.py`, never from the unverified wiki head — drawing
them automatically fed 下着コート back into the prompts and MT reproduced it.

Disagreements between the two sources land in `assets/tag_glossary_review.md`,
ordered by occurrence count, for the sign-off the phase gate asks for. Fixes go
in `tag_overrides.json` and beat every automatic source on the next build.

## `build_pairs.py` — the corpus

Captions are tag strings, so the JA side is **composed from the glossary**
rather than translated wholesale: idiom and pinned proper nouns are then true
by construction instead of by hope. Registers: `tags` (primary wording),
`tags_alt` (verified synonyms swapped in — unverified wiki alternates would
inject 女スパイ for `1girl`), `natural` (`--mt`, prose rewrite with
character/copyright terms pinned), plus D6 quoted-text templates seeded from
the LoveHina lines.

Coverage is reported in **ext rows**, encoded with the same
`HybridT5Encoder` training will use — the histogram is what says which rows a
run can actually move, and never-visited rows are the ones the plan says to
flag rather than ship as noise.

## Network note

`danbooru.donmai.us` is unreachable from this machine (connection reset — a
network-level block, not the sandbox), so the wiki arrives via the HF mirror
and **D2 (artist commentary) cannot be pulled directly**. Any D2 route has to
go through an HF-hosted metadata dump.

## `wikidata_lexicon.py` — proper nouns

MT transliterates names, and our captions are made of names: unassisted,
`acheron (honkai: star rail)` comes out アケロン instead of **黄泉**,
`silver wolf` シルバーウルフ instead of **銀狼**.

This does **not** corrupt the training pairs — teacher and student share the
Qwen side and the teacher's T5 side carries the original EN caption, so
アケロン→Acheron is a self-consistent thing to learn. The cost is
*distribution*: a JA user types 黄泉, whose ext rows the corpus then never
visited, so the failure lands on the token that identifies the character. Plus
coverage — name kanji go unvisited entirely. Applied as a **pre-substitution on
the EN side with the JA target pinned**.

Entity list is ours, not the world's — `caption_index.json` (`make caption-index`)
already types every caption tag, so we only query names that actually occur in
training data (1,314 character / 150 copyright / 89 artist tags over 3,008 images).

**Two routes, because names come in two shapes.** Qualified names
(`aru (blue archive)`) go through the **franchise roster**: resolve the
copyright tag to candidate QIDs, keep only candidates that have a character
roster (the precision filter — it rejects `chimera` → キマイラ), union their
rosters, then token-subset match the paren-stripped name
(`aru` ⊆ {rikuhachima, aru} → **陸八魔アル**). Unqualified full names
(`aisaka taiga`, 607 of the tags) go through a **global label match** guarded
by ≥2 tokens plus `P31/P279* Q95074` (fictional character) — the type
constraint is what makes flat matching safe here.

An unguarded flat lookup on *single-token given names* is the trap: it looks
like 67.6% recall and mostly returns the wrong entity (`aru` → アランデル駅,
a railway station; `ann` → Anime News Network).

Measured 2026-08-15: 57/150 franchises resolved; **499/1314 characters**
(257 via roster, 242 via full name) = 38.0% of types and **42.9% of tag
occurrences**; **0/89 artists** — pixiv handles are not encyclopedia entities,
expect nothing there. Contributes **556 unique CJK codepoints**, including
rare name kanji (鈎 錠 霄 梔 棗 楯 芬) that ordinary caption prose never reaches.
Rows carry `via` provenance so the two routes stay separable.

Precision evidence is 25/25 correct on a spot-check of each route. The
`character_ambiguous: 0` stat is **not** independent evidence — ambiguity is
only flagged when two candidates carry different JA strings at equal EN-token
length, so a shorter label wins silently.

`--langs ja ko zh` pulls the other labels from the same rows, so the deferred
zh/ko phases inherit their proper-noun layer for free.

## `lovehina.py` — native register

Professional JA→PL translations of Love Hina vols 1 and 14. We keep the
**Japanese side only** — the Polish target is useless (our teacher needs EN),
but the JA is the one thing no other source supplies: native, professionally
written manga text that no MT system produced.

Its primary job is as a **held-out native-register eval set**. The proposal's
register-mismatch risk (student learns translated-sounding JA better than real
JA) is otherwise unmeasurable, because every other corpus source is either
MT output or MT input. Secondary job: 40 corner-bracket-quoted lines seed the
D6 quoted-text constructions, and reading order is preserved so consecutive
bubbles can be concatenated into multi-utterance samples.

Measured: 3,705 lines, 51,575 chars, **904 unique kanji + 161 kana**. The two
sources are complementary rather than redundant — each holds codepoints the
other misses, the lexicon's being name kanji (梔 棗 楯 鈎 錠 霄 芬).

Volume is small and it is **not** bulk training data — treat it as eval plus
seed. Only annotation JSON is fetched; the pages live in Manga109-s, which
forbids redistribution and needs a separate application. That matters only if
Phase 4 (OCR + glyph rendering) ever wants images — the text-only distillation
does not.
