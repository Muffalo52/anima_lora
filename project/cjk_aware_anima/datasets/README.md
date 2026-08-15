# `datasets/` — corpus builders for the CJK distillation

Data-side tooling for Phase 2a of [`../initial_proposal.md`](../initial_proposal.md).
Each script is stdlib-only, CPU-only, and writes into `assets/` (gitignored —
regenerate rather than commit). Run them from the repo root.

The distillation teacher is computed at **embedding level** from an EN string,
so nothing here needs human-quality parallel text; what it does need is *JA that
is right*, in the *registers users actually prompt in*, hitting *ext rows the
corpus will really visit*. The two builders below cover the parts machine
translation cannot supply on its own.

| Script | Source | Licence | Output |
|---|---|---|---|
| `wikidata_lexicon.py` | Wikidata SPARQL, entity list from our `caption_index.json` | CC0 | `assets/wikidata_lexicon.json` — EN↔JA(↔KO/ZH) proper nouns |
| `lovehina.py` | [PLippmann/multimodal-manga-translation](https://github.com/PLippmann/multimodal-manga-translation) (COLING 2025, arXiv:2411.02589) | MIT | `assets/lovehina_ja.json` — 3,705 native JA speech-bubble lines |

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
