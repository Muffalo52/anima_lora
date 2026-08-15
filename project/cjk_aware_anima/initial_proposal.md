# JA-native conditioning — distill the extended T5 vocab against the EN-translation teacher

Status: **PROPOSAL.** Phase 0 (probe) and Phase 1 (zero-shot ext vocab) are
done and measured (`bench/cjk_adapter/`, results `20260815-1836`), as are the
Phase 2a data assets that MT cannot produce — the proper-noun lexicon (2a-lex)
and the native-register eval set (D7), both under [`datasets/`](datasets/),
2026-08-15. The rest of Phase 2 is unstarted. Scope is **Japanese only** for now — the ext
vocab and probe cover ko/zh too, but data building, distillation, and gates
below are ja-first (zh/ko become "rerun the same pipeline with a different
corpus" once ja passes).

## TL;DR

The LLM Adapter's T5-side query stream is the only thing standing between
Anima and native Japanese prompts: with the T5 side fed an **English
translation** (`ja_t5en` arm), conditioning recovers to cos **0.69–0.76** vs
the all-EN reference with healthy discrimination (~0.11); with the stock
spiece it collapses to `▁ <unk> </s>` (cos ~0.02, discrimination 0.91 = dead
pathway). Phase 1 built a **58,968-row extended vocab** (Qwen-borrowed CJK
tokens, anchor-ridge-mapped into the T5 embedding space) — zero-shot it is
injective (discrimination 0.18) but off-manifold (cos ~0.05). Phase 2 =
**distill the ext rows** so that the student `(Qwen ja, T5-ext ja)` matches
the teacher `(Qwen ja, T5 en-translation)` at the adapter output. The two
arms **share the Qwen side**, so the loss isolates exactly the broken piece,
and the trainable surface is only the new rows — the original 32,128 rows and
the EN tokenize path are untouched, so **EN prompts are bit-identical by
construction**.

The data question dissolves once the teacher is embedding-level: we don't
need human parallel corpora, we need **JA caption-domain text + any decent
local MT model**. Primary source = our own caption master translated EN→JA;
secondary = native-JA artist commentary translated JA→EN; public
caption-parallel sets are optional ballast. Two things MT alone cannot supply
are built first: a Wikidata-derived EN↔JA **proper-noun lexicon** scoped to
the entities our own `caption_index.json` says we actually train on (MT
transliterates names, so 黄泉 would never appear where users will type it —
2a-lex, done), and a **native-register eval set** from professionally written
manga text, since every other source in the mix is either MT output or MT
input and therefore cannot measure register drift (D7, done).

Downstream (deferred, but the reason this is worth more than an inference-time
MT shim): each JA character gets a **unique, stable T5-side id** instead of
`<unk>`, which is the prerequisite for (a) appending OCR'd in-image text to
captions instead of masking it, and (b) an eventual JA text-rendering phase —
neither is reachable through a translation shim, which destroys the verbatim
string.

## Premise sources (measured)

- `bench/cjk_adapter/run_bench.py`, run `20260815-1836` (seed 42, cfg 3.5,
  30 steps, 1024², compiled, images rendered):
  - `ja_t5en` cos_vs_en **+0.765 / +0.692** (p1_sign / p2_beach),
    discrimination **0.114** → content flows from the Qwen3 side; the T5
    stream only needs to be a *meaningful* anchor. This is the teacher.
  - `ja_native` cos **+0.022 / +0.053**, discrimination **0.909** → today's
    pipeline is dead for ja (3 non-pad T5 tokens, 1 unk).
  - `ja_t5rom` cos **~0.06** → **romanization refuted**: non-unk is not
    enough, the stream must be semantic. Do not re-propose.
  - `ja_q_en` cos **~0.07/0.15**, discrimination 0.92 → reverse routing
    (EN on Qwen, ja on T5) equally dead; the Qwen side is the content channel.
  - `ja_ext` (Phase 1 zero-shot) cos **+0.057 / +0.051**, discrimination
    **0.184** → the mapped rows are injective (different prompts separate)
    but off-manifold. Zero-shot anchor mapping is insufficient; **training is
    required** — that gap is exactly what Phase 2 closes.
- `bench/cjk_adapter/build_ext.py` assets (`assets/ext_embed.*`): 58,968 rows
  = 30,951 clean Qwen CJK tokens + 28,017 per-char fallback rows; anchor fit
  on 20,873 surface-identical tokens, held-out cos **0.752**; rows rescaled
  ×1.658 to the T5 table's mean row norm. ~241 MB fp32 sidecar (all of CJK —
  a ja-only subset is much smaller, see Phase 2b).
- `library/anima/models.py::LLMAdapter` (~:2599): 6-block cross-attn bridge,
  T5-id query table `embed [32128, 1024]`, queries cross-attend into Qwen3
  hidden states. The ext table appends rows to `embed`; nothing else changes.
- Mojibake in `process_escape` fixed in c8cf3ce2 — native-CJK prompt strings
  survive the CLI/config path now (prerequisite, done).
- MIT / ComicTextDetector is already integrated in `make mask`
  (`scripts/tasks/masking.py`) — the same upstream tool
  (manga-image-translator, in the parent dir) carries the OCR + translator
  stages Phase 4 needs. No new external dependency.

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
   teacher later (Phase 4 trains against images, not embeddings).

B0 stays in every eval grid as the bar to match.

## Phase 2 — distillation design

### Objective

Student and teacher share the Qwen3 hidden states (same native-JA source
text); only the T5-side ids differ:

```
teacher:  adapter(qwen_hidden(ja), t5_ids(en_translation))     # no grad
student:  adapter(qwen_hidden(ja), t5ext_ids(ja))              # grad → ext rows only
loss:     match(student_out, teacher_out)
```

Trainable surface, in order of escalation (each step gated on the previous
plateauing below target):

- **2-i (default): ext embedding rows only** (~60 M fp32 for full CJK; ja
  subset less). Frozen adapter, frozen everything. EN path provably
  unaffected — original rows and spiece tokenization untouched, and
  `HybridT5Encoder` is bit-identical for pure-EN text (asserted by a unit
  test, see gates).
- **2-ii: + low-rank adapter on `q_proj`/`in_proj` of the adapter blocks.**
  Only if 2-i plateaus. This *breaks* the EN-bit-exactness guarantee (shared
  weights move), so it must clear an EN-regression gate and would ship as a
  separate optional sidecar.
- **2-iii: full adapter finetune** — not proposed; it risks the pretrained
  EN behavior for marginal gain and turns a sidecar into a checkpoint fork.

### The alignment problem (main technical risk)

Adapter output is indexed by **T5-token position**. Teacher positions carry
EN tokens (e.g. 31 non-pad), student positions carry JA ext tokens (e.g. 25
non-pad) — a position-wise MSE compares *different tokens of different
counts*. Three loss candidates, to be settled by a small ablation at the top
of Phase 2c:

- **L_flat** — MSE/cosine on the full padded `[512, d]` output. Crude but the
  probe's own metric; misalignment may act as tolerable noise because the
  output at every position is Qwen-context-dominated (pad-position outputs
  already track the Qwen side — that's why `ja_native` vs `en` cos ≈ 0 even
  on 481 shared pad positions).
- **L_attn (recommended if L_flat stalls)** — distill at the interface the
  DiT actually consumes: cross-attention is permutation-invariant over KV, so
  match `Attn(Q_probe, K(out), V(out))` for a fixed probe-query bank (e.g.
  the first DiT cross-attn block's frozen K/V projections + a few hundred
  cached image-token queries at 2–3 σ levels). One attention op, no DiT
  forward, set-level by construction.
- **L_pool** — mean/max-pooled sequence statistics. Weakest; control arm only.

### Training loop

Bespoke small loop, **not** `train.py` — mirror the `scripts/distill_mod/`
pattern: `scripts/distill_cjk/`. Notes:

- The adapter can be instantiated **standalone** from the DiT checkpoint's
  `net.llm_adapter.*` keys (build_ext.py already reads tensors this way) — no
  full DiT load. Memory footprint = Qwen3 0.6B encoder + 6-block adapter;
  batches of hundreds of prompts on one GPU.
- Qwen3 hidden states are computed **once per pair** and fed to both
  teacher and student adapter passes (teacher no-grad).
- Padding invariant applies (max-pad 512, no `crossattn_seqlens` masking) —
  the bench's `SplitTokenizeStrategy` / `HybridT5Encoder.encode` already
  encode this correctly; promote them out of `bench/` into the strategy
  module when the loop lands.
- GPU work through the daemon (`make daemon-run` / `--queue`), result
  envelope + eval via the existing bench.

Estimated cost: with ~10⁵–10⁶ pairs and a 6-block adapter forward, this is
hours on one GPU, not days. The surface (≤60 M params) saturates well before
the corpus does.

## Phase 2a — data build (the actual open question)

Because the teacher is computed from the EN side *at embedding level*,
**synthetic translations are first-class training data** — MT quality only
needs to preserve meaning, comfortably above the bar for modern local LLMs.
Sources, in priority order:

| ID | Source | Direction | Domain fit | Est. size | Notes |
|---|---|---|---|---|---|
| D1 | Our caption master (`image_dataset/*.txt`) | EN→JA via local LLM | exact | ~dataset size × variants | The backbone. Generate 2–3 register variants per caption: tag-string JA / natural-sentence JA / mixed, since real users prompt in all three. Reuse `caption_index.json` (`make caption-index`) for vocabulary coverage stats. |
| D2 | Danbooru/pixiv artist commentary (native JA on the same images) | JA→EN via local LLM | exact, native phrasing | 10⁴–10⁵ | Covers word order / casual register / kaomoji adjacency that EN→JA MT never produces. Needs a metadata pull; keep local, research use. |
| D3 | STAIR Captions (~820k JA on COCO, aligned to EN COCO captions); YJ Captions (~131k) | pre-aligned | caption-domain, photographic | ~1 M | Real parallel captions, zero MT. **Check licenses before mixing** (STAIR is CC BY 4.0 last checked; verify). |
| D4 | JESC (~2.8 M subtitle pairs) | pre-aligned | casual dialogue | subsample | Matches the speech-bubble / quoted-text register. Subsample; don't let it dominate. |
| D5 | **Wikidata SPARQL**, entity list taken from our own `caption_index.json` | lexicon (EN↔JA↔KO↔ZH) | anime proper nouns | 499 characters + 56 franchises **built** | Pins the JA form of names MT would otherwise transliterate — a **coverage/distribution** lever, not a correctness gate (see 2a-lex). CC0, no licence question. `datasets/wikidata_lexicon.py`. |
| D7 | **LoveHina manga annotations** ([PLippmann/multimodal-manga-translation](https://github.com/PLippmann/multimodal-manga-translation), MIT) | native JA (JA→EN via local LLM) | manga speech bubbles | 3,705 lines / 904 kanji **built** | **Eval instrument first, data second** — the only native, non-MT-touched JA in the mix, so it is what measures risk 3. `datasets/lovehina.py`. |
| D6 | Quoted-text templates | synthetic | embedded-text constructions | 10⁴ | `「X」と書かれた看板/Tシャツ/黒板…` with X drawn from D1–D5 fragments. Teacher side in **two variants**: fully translated ("a sign that says GOOD MORNING") *and* quote-preserved ("a sign with 「おはよう」 written on it") so the quote-carrying construction isn't foreign to the student. |

### 2a-lex — the proper-noun lexicon (BUILT, measured)

`datasets/wikidata_lexicon.py` (run 2026-08-15,
`datasets/assets/wikidata_lexicon.json`). This is the one part of 2a that is
done.

**What it actually buys — distribution, not loss correctness.** The first
draft of this section claimed raw MT would make D1 *wrong*. That is not true
under the objective above, and the distinction matters. Teacher and student
share the Qwen side, and the teacher's T5 side carries the **original EN
caption**: if MT renders `acheron` as アケロン, the pair
`(qwen(アケロン…), t5(acheron…))` is perfectly self-consistent and the student
learns アケロン→Acheron correctly. Nothing is corrupted. The real costs are:

1. **Train/inference distribution mismatch.** A JA user types **黄泉**, not
   アケロン. The ext rows for 黄泉 were never visited, so at inference the T5
   stream for exactly the token that identifies the character falls back to
   the zero-shot regime (cos ~0.05, the `ja_ext` Phase-1 result) — the failure
   lands precisely on the content word that matters most.
2. **Coverage.** The 157 name kanji below are never visited at all if every
   name arrives as katakana transliteration.

So the lexicon is a **coverage-and-distribution** instrument, applied as a
pre-substitution on the EN side before MT with the JA target pinned. That is
still the highest-value single data intervention here — character names are
the most frequent content tokens in our captions — but it is not a
correctness gate, and Phase 2c should not be blocked on it.

A note on the obvious cheaper alternative: **leaving danbooru tags
untranslated in the JA caption is not equivalent and is strictly worse.**
Latin-script tags route to the original spiece rows, so those positions train
*no ext rows at all* — the opposite of the goal.

**Entity list is ours, not the world's.** `caption_index.json` already parses
every caption into typed axes, so we only ever query names that actually
occur in training data: 1,314 `character`, 150 `copyright`, 89 `artist` tags
over 3,008 images. This keeps query volume trivial and the lexicon scoped to
rows the corpus will really visit.

**Query design.** Two routes, because one does not cover both name shapes:

1. **Franchise-constrained** (for names qualified by a copyright, `aru (blue
   archive)`): resolve the `copyright` tag to candidate QIDs by label/altLabel
   over case variants (romaji altLabels are why `ane naru mono` finds
   姉なるもの); pull every candidate's roster — items pointing in via `P1441`
   (present in work) / `P179` (series) / `P361` (part of) with a `ja` label;
   **a candidate with no roster at all is discarded**, which is what rejects
   `chimera` → キマイラ. Then join by token-subset match on the paren-stripped
   base name (`aru` ⊆ {rikuhachima, aru} → **陸八魔アル**). Rosters from *all*
   surviving candidates are unioned: franchises routinely split across a
   series item, an anime item and a game item.
2. **Global full-name** (for unqualified names, `aisaka taiga` — 607 of the
   tags): flat label/altLabel match, guarded by requiring ≥2 tokens **and**
   `P31/P279* Q95074` (fictional character). The type constraint is what does
   the work — it is what stops the `Anime News Network` class of match.

Measured on the live dataset:

| Axis | Resolved | Note |
|---|---|---|
| copyright → franchise QID | **57 / 150** | misses are doujin circles, magazine titles, small games with no Wikidata item |
| character via franchise roster | **257 / 488 (52.7%)** within a resolved franchise | |
| character via global full-name | **+242** | the route that unqualified names need |
| character total | **499 / 1314 (38.0%)** types, **1292 / 3013 (42.9%) occurrences** | occurrence coverage runs ahead of type coverage — popular characters are better documented. 1,079 / 3,008 images have ≥1 covered character |
| artist | **0 / 89** | pixiv/danbooru handles are not encyclopedia entities — expect nothing, use D2 metadata |
| unique CJK codepoints contributed | **556** | kana + name kanji |

Precision evidence is a 25-sample spot-check per route, both 25/25 correct
(`陸八魔アル`, `黄泉`, `銀狼`, `食蜂操祈`, `雪ノ下陽乃`, `胡桃`). Note the
reported **0 ambiguous is weak evidence**: ambiguity is only flagged when two
candidates carry *different* JA strings at equal EN-token length, so a shorter
label wins silently. Treat the spot-checks as the precision claim, and widen
them in the 2a sign-off pass.

The codepoint row is why this matters beyond name accuracy: name kanji are
precisely the **rare rows** a caption corpus otherwise never visits
(鈎 錠 霄 梔 棗 楯 芬 陸八魔 銀鏡 伊原木…). Several hundred names buy a tail
that millions of ordinary caption sentences would not.

**Free ko/zh.** The same rows carry `ko`/`zh` labels (`--langs ja ko zh`), so
the deferred zh/ko expansion inherits its proper-noun layer at zero extra
cost — one more reason those phases are "rerun the pipeline", not "redo the
work".

**Known residue** (accepted, small): franchise resolution still lets a few
wrong-but-rostered items through (`ananta` → クリシュナ, the Hindu deity), and
some labels carry subtitles (`blue archive` → `ブルーアーカイブ -Blue
Archive-`). Franchise labels are substitution *candidates* and get the same
user spot-check as the MT sample; the character join is the part that
dominates caption text and it was clean. Rows carry `via` provenance
(`franchise_roster` / `full_name`) so the two precision profiles stay
separable downstream.

**Remaining lift** (cheap, not done): `pull_rosters` follows the three
properties directly with no path, so characters attached to a *sub-work*
(a season, an individual game) rather than the series item are invisible —
plausibly a large share of the 231 within-franchise misses. Beyond that, a
`wbsearchentities` fuzzy fallback and a danbooru tag-wiki alias cross-check
would attack the 93 unresolved copyrights. The genuine Wikidata limit is
per-franchise roster depth (Pokémon trainers, Arknights operators and most
Azur Lane ships are not individually itemised); those fall back to MT plus
human spot-check.

MT engine: a local LLM (larger Qwen3 / Swallow / plamo-translate class),
batch-offline, greedy or low-temperature; a spot-checkable sample (~200
pairs) goes in the run dir for the user (who reads JA) to eyeball. No paid
API dependency.

**Coverage is a sampling constraint, not a data-volume constraint.** Ext rows
only move where the corpus visits them. Deliverables of 2a therefore include
a coverage report (ext-row visit counts over the assembled corpus). Policy:

- kana + punctuation + fullwidth rows: must be saturated (they will be).
- kanji: train the visited set; expect D1+D2 to cover jōyō + our domain tail,
  with D5 supplying the name-kanji tail (556 codepoints measured) that
  ordinary caption prose never reaches.
- **never-visited rows stay at zero-shot init but get flagged**; if eval shows
  they poison nearby generations, demote them to the `<unk>` fallback in the
  encoder mapping rather than shipping noise rows (cheap: it's a JSON edit).

Ja-only scoping: build the corpus ja-only, but keep the ext table's zh/ko
rows physically present (they're inert without zh/ko input text). The
distilled sidecar ships with per-row provenance (`trained` / `zero-shot`) in
the mapping JSON.

## Phases & gates

- **Phase 2a — corpus.** Deliverable: `post_image_dataset/cjk_distill/`
  (or similar) with pairs + coverage report + 200-pair human spot-check
  sample. The D5 lexicon and the D7 native-register set are **done**; what remains is the MT pass
  with lexicon pre-substitution applied and the D2 pull. Gates: user signs off
  on the translation sample; kana/jōyō coverage saturated; **no proper noun
  from the lexicon appears MT-transliterated in the sampled JA** (i.e. the
  substitution actually took — spot-check that 黄泉 shows up and アケロン
  does not).
- **Phase 2b — loop + unit gates.** `scripts/distill_cjk/` lands with:
  (G1) EN bit-exactness test — `HybridT5Encoder` on pure-EN text produces
  identical ids to stock spiece, and a pure-EN prompt's conditioning is
  bitwise unchanged with the ext table attached; (G2) loss-ablation harness
  (L_flat vs L_attn vs L_pool on a 1k subset).
- **Phase 2c — train + eval.** Acceptance harness is **the existing bench
  unchanged**: `run_bench.py --ext --languages ja` pointed at the distilled
  sidecar. Gates:
  - `ja_ext` cos_vs_en ≥ **0.6** on both content prompts (teacher sits at
    0.69/0.77 — get within noise of it; stretch: match it),
  - discrimination ≤ **0.2**,
  - held-out corpus: student-vs-teacher cos ≥ 0.9,
  - **native-register readout (D7)**: student-vs-teacher cos on the held-out
    LoveHina lines, reported *separately* from the MT-derived held-out set.
    Not a hard gate on the first pass — a large gap between the two is the
    register-drift signal (risk 3) and calls for reweighting D2/D7 in the mix,
    not for failing the run,
  - rendered same-seed grid: `en` / `ja_t5en` (B0) / `ja_ext` — user judges
    ja_ext ≈ ja_t5en on prompt adherence. A wider JA prompt set (~20, incl.
    quoted-text) rendered for the eyeball pass.
  - EN regression: G1 stays green post-training (trivially true for 2-i).
- **Phase 3 — ship.** Promote encoder + strategy shim out of `bench/` (auto
  route CJK spans through the ext encoder when the sidecar is present; flag
  to disable), extend the adapter embed at DiT load, sidecar as a release
  asset (pattern: the CNS γ npz), ComfyUI loader-node touchpoint (see
  Deployment). TE-cache regeneration note applies (padding-invariant §: any
  tokenizer change invalidates cached `.npz`/TE sidecars for JA captions —
  EN caches are unaffected by construction).
- **Phase 4 (deferred, separate proposal when reached) — OCR captions +
  glyph rendering.** OCR in-image JA via manga-image-translator (already the
  `make mask` MIT backend), append verbatim strings to captions, and train
  image-level so the DiT binds ext-token ids to glyphs. **Explicitly deferred
  and explicitly ordered**: keep text-masking ON until this phase — unmasking
  makes the loss pay for glyph pixels the model can't yet produce. Phase 2
  only needs masks left as they are.

## Deployment — a "vocab pack", not a LoRA

The Phase 2 artifact **cannot be a LoRA**: LoRA expresses a low-rank delta on
an *existing* matrix, while this artifact is (a) new rows appended to the
adapter's `embed [32128, 1024]` table (a shape change) and (b) a tokenizer
mapping (segmentation rules + char/token → row-id JSON), which is behavior,
not weights. It ships as the two files Phase 1 already emits —
`ext_embed.safetensors` + `ext_embed.json` — as a release asset (pattern:
CNS γ npz). Per surface:

- **In-repo**: strategy shim routes CJK spans through `HybridT5Encoder`;
  `load_dit_model` appends the rows to `llm_adapter.embed`. Auto-discovery
  of the sidecar (flag to disable). Composes with any checkpoint and any DiT
  LoRA by construction (disjoint parameters). Preprocess/TE-caching uses the
  same strategy, so JA captions cache with ext ids transparently.
- **ComfyUI** (verified against core, 2026-08-15): tokenization lives in the
  CLIP object (`comfy/text_encoders/anima.py::AnimaTokenizer`, stock
  `T5TokenizerFast`; ids ride conditioning extras as `t5xxl_ids`), and the
  MODEL holds the adapter with a **hardcoded** `Embedding(32128, …)`
  (`comfy/ldm/anima/model.py:159`). One custom node — "Anima JA Vocab":
  `(MODEL, CLIP, vocab_pack) → (MODEL, CLIP)` — wraps the CLIP's t5xxl
  tokenize path (pure Python) and object-patches the adapter embed with the
  extended table (ModelPatcher object patch, per the Adapter Loader repo's
  forward_hook-not-override invariant — don't clobber the class). Home:
  `ComfyUI-Anima_lora-Adapter`. Endgame: upstream to ComfyUI core (core
  already owns Anima natively) → node-free.
- **Bake-in rejected**: expanding `embed.weight` inside a forked DiT
  checkpoint forks the base model, *breaks* stock ComfyUI load (32128 is
  hardcoded in core), and still doesn't carry the tokenizer — the
  irreducible piece is code, which a weights file can't ship.
- The optional 2-ii fragment (low-rank `q_proj`/`in_proj` deltas) *is* a
  genuine LoRA and rides the existing Anima Adapter Loader — but always
  alongside the vocab pack, never instead of it.
- Phase 4's eventual glyph-rendering training, by contrast, outputs an
  **ordinary DiT LoRA** on every existing loader — it merely *depends* on
  the vocab pack being loaded, like any LoRA depends on its base.

## Reuse inventory

| Piece | Where | State |
|---|---|---|
| Ext vocab build (anchor map, table, mapping) | `bench/cjk_adapter/build_ext.py`, `ext_vocab.py` | shipped, assets built |
| Hybrid encoder (CJK spans → ext ids, EN bit-identical) | `ext_vocab.HybridT5Encoder` | shipped; promote to `library/anima/` in Phase 3 |
| Split tokenize strategy (probe plumbing) | `run_bench.py::SplitTokenizeStrategy` | teacher-side encoding for the loop |
| Acceptance harness (diagnostics + rendered arms) | `run_bench.py --ext` | shipped — Phase 2c reuses unchanged, just swap the sidecar |
| Adapter tensors without DiT load | `build_ext.py::read_tensor` pattern | extend to full `net.llm_adapter.*` |
| Bespoke distill-loop pattern | `scripts/distill_mod/` | template for `scripts/distill_cjk/` |
| OCR / text detection | MIT backend of `make mask` (`scripts/tasks/masking.py`) | shipped; OCR+translate stages unused so far (Phase 4) |
| Tag vocabulary | `caption_index.json` (`make caption-index`) | coverage stats + D5 entity list (typed axes are exactly the query keys) |
| EN↔JA proper-noun lexicon (Wikidata) | `datasets/wikidata_lexicon.py` → `datasets/assets/wikidata_lexicon.json` | **built + measured**; CC0; `--langs ja ko zh` for the deferred phases |
| Native-register JA eval set (manga) | `datasets/lovehina.py` → `datasets/assets/lovehina_ja.json` | **built**; MIT; 3,705 lines / 904 kanji; hold out, never train |

## Risks / open questions

1. **Position misalignment** (see L_flat vs L_attn above) — the one real
   unknown; bounded by the G2 ablation before committing GPU time.
2. **Teacher ceiling**: distilling to t5en caps at t5en. Accepted for this
   phase; image-level training (Phase 4) is the lever past it.
3. **Register mismatch**: if D1's MT-JA dominates, the student may handle
   translated-sounding JA better than native phrasing — D2 exists exactly to
   hedge this; keep it ≥ ~20% of the mix. This was previously unmeasurable
   (every source was MT output or MT input); **D7 is the instrument** — hold
   it out entirely and report student-vs-teacher cos on it separately from
   the MT-derived held-out set. A gap between the two *is* register drift.
4. **Proper-noun distribution gap**: raw MT transliterates character names
   (アケロン where a user types 黄泉), so those ext rows go unvisited and fall
   back to the zero-shot regime at inference. Bounded by the 2a-lex
   substitution for the 499 names it resolved; the remaining ~815 character
   tags still ride raw MT. Track proper-noun coverage separately in the
   spot-check rather than folding it into overall translation quality. Not a
   loss-correctness issue — see 2a-lex.
5. **Row-norm / manifold**: zero-shot rows needed a ×1.66 norm rescale;
   training may drift norms — log row-norm distribution vs the original
   table as a diagnostic.
6. **Discrimination as a trap**: optimizing cos-to-teacher could collapse
   diversity if the corpus is template-heavy — the discrimination gate (≤0.2)
   and register variants guard this.

## Anti-re-proposal notes

- Romanization (any transliteration scheme) is **refuted** (`t5rom` ~0.06).
- Reverse routing (`q_en`) is **refuted** — the Qwen side must carry native.
- Zero-shot mapping (any cleverer W: procrustes, per-block, nonlinear) is
  **not worth another dry run** — 0.75 anchor holdout cos still yielded 0.05
  end-to-end; the gap is contextual, not linear-map quality.
- **Unguarded flat Wikidata label matching is refuted for single-token given
  names** (`aru`, `ann`, `ako`): the apparent hits are railway stations and
  news agencies. Scope this correctly — it is *not* a refutation of flat
  matching in general. For **multi-token full names** with a
  `P31/P279* Q95074` (fictional character) type guard, flat matching is the
  route that works and it supplies 242 of the 499 entries (25/25 spot-check).
  Qualified names go through the franchise roster; unqualified full names go
  through the guarded global query.
- Expecting Wikidata to cover **artists** is refuted (0/89) — pixiv/danbooru
  handles are not encyclopedia entities. Artist names go through D2 metadata.
- Training a new SentencePiece over T5 is out: borrowing Qwen tokenization
  keeps the source and query streams **token-aligned over CJK spans**, a
  structural property a fresh spiece would destroy.
