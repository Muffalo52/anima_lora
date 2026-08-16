# CJK-aware Anima — done

What is finished. Details and measured numbers live with the code that
produced them — this file only says *what exists* and *where*.

*Line home: [`motivation.md`](motivation.md) (why, incl. directions already
ruled out) · [`plan.md`](plan.md) (what remains) ·
[`report.md`](report.md) (Phase 2b measured verdicts).*

## Phase 0 — probe (2026-08-15)

- [x] Language/routing arm sweep — `bench/cjk_adapter/run_bench.py`,
      results `results/20260815-1836/result.json` (+ rendered grids).
      Established `t5en` as the teacher and closed romanization and reverse
      routing.
- [x] `process_escape` mojibake fix (c8cf3ce2) — native-CJK prompt strings
      survive the CLI/config path.

## Phase 1 — zero-shot ext vocab

- [x] Ext table build — `bench/cjk_adapter/build_ext.py` →
      `assets/ext_embed.{safetensors,json}`.
- [x] Hybrid encoder (`ext_vocab.HybridT5Encoder`) — CJK spans → ext ids,
      bit-identical on pure EN. Promote to `library/anima/` in Phase 3.
- [x] Acceptance harness — `run_bench.py --ext`. Phase 2c reuses it unchanged,
      swapping the sidecar.
- [x] Zero-shot arm measured (same result dir) — anchor mapping alone is
      insufficient, so Phase 2 training is required.

## Phase 2a — data build

Builders and their measured results: [`datasets/README.md`](datasets/README.md).

- [x] `datasets/wikidata_lexicon.py` — EN↔JA/KO/ZH proper-noun lexicon (CC0).
- [x] `datasets/lovehina.py` — native-register JA eval set (MIT). Held out,
      never trained on.
- [x] `datasets/mt.py` — `MTEngine` (Hy-MT2, Apache-2.0) + `--probe` idiom
      benchmark + `--smoke`; per-batch resume cache, `--gpu-budget`.
- [x] `datasets/tag_glossary.py` — EN→JA tag glossary from the Danbooru wiki,
      lexicon and MT → `assets/tag_glossary_ja.json` + `_review.md`.
      `--reselect` re-derives choices with no GPU.
- [x] `datasets/tag_overrides.json` — hand-pinned wording (committed; beats
      every automatic source on rebuild).
- [x] `datasets/build_pairs.py` → `post_image_dataset/cjk_distill/`
      (`pairs.jsonl`, `coverage.json`, `spotcheck.md`).
- [x] `tests/test_cjk_glossary.py` — 19 invariants (script detection, alt-pool,
      cache resume).
- [x] Gates passed: proper-noun substitution, kana saturation, occurrence
      coverage (`coverage.json`).

Still open in 2a — see [`plan.md`](plan.md#phases--gates): user sign-off on
`tag_glossary_review.md`, and the `natural` prose register (implemented, not
run).

## Phase 2b — loop + gates (2026-08-15/16)

Measured verdicts: [`report.md`](report.md).

- [x] `scripts/distill_cjk/` — corpus cache, ext-table ladder, four objectives,
      G2 driver (`make exp-cjk-cache` → `exp-distill-cjk`).
- [x] `tests/test_cjk_distill.py` — G1 EN bit-exactness.
- [x] `scripts/distill_cjk/build_query_bank.py` — real cross-attn probe queries
      (DiT forwards at 2–3 σ) → `bench/cjk_distill/assets/query_bank.safetensors`.
      `attn_bank.build_bank` refuses random directions without an explicit flag.
- [x] `attn_bank.fit_centers` — the readout's common offset, projected out.
- [x] `load_pairs` splits **by image**, not by pair (no sibling leakage; the
      near metric is populated).
- [x] Gates G0b / G1 / G0 / G2 all passed. Settled: `param=global`, `loss=span`.

## Phase 2a — D2 (2026-08-16)

- [x] `datasets/commentary.py` — native-JA danbooru artist commentary via the
      gelcrawl route (`curl_cffi` + SpoofDPI). 434,800 raw →
      `assets/commentary_ja.jsonl`: **73,015** unique JA records (5.2 M chars,
      4,775 unique kanji), 3,347 with a human EN translation. Per-line promo
      stripping; zh/ko kept in the raw cache for when ja-only scoping lifts.
- [x] `commentary.py --mt` — the JA→EN teacher side (Hy-MT2-7B, greedy).
      Names pinned JA→EN off the D5 lexicon (fires on ~13% of records; the 1.8B
      renders 十時愛梨 "Jūji Aira" and hallucinates 虹ヶ咲 → "Neko no Hikari",
      which is why the 7B is worth 4× the wall clock). Length-bucketed batching
      with per-bucket `max_new_tokens` **and** batch size — batch 32 in the
      ≤128-char bucket OOMs the 7B at a 13 GiB weight budget. Resumable through
      `mt.py`'s prompt-keyed cache; `--from-cache` harvests a stopped pass on
      CPU. Partial run: **5,721 of 69,668** translated, 3.6% rejected by the
      output gate (`untranslated_cjk` / `empty` / `runaway`).
- [x] `build_pairs.py --commentary` — the D2 `commentary` register (9,068 pairs
      = 5,721 MT + 3,347 human). Span-less by construction. Measured in
      [`report.md`](report.md#d2--what-the-commentary-corpus-buys-2026-08-16).
- [x] `datasets/manga_text.py` — danbooru text-detection corpus, piloted and
      **rejected as corpus material** (register duplicates D4; OCR noise arrives
      MT-laundered and undetectable). Kept for geometry: mask validation and
      Phase 4. See [`datasets/README.md`](datasets/README.md).

## Reusable for Phase 2b onward

| Piece | Where |
|---|---|
| Teacher-side encoding | `run_bench.py::SplitTokenizeStrategy` |
| Adapter tensors without a DiT load | `build_ext.py::read_tensor` pattern — extend to `net.llm_adapter.*` |
| Bespoke distill-loop template | `scripts/distill_mod/` → `scripts/distill_cjk/` |
| Adapter injection point | `library/anima/models.py::LLMAdapter` (~:2599) |
| Tag vocabulary / entity list | `caption_index.json` (`make caption-index`) |
| OCR + text detection (Phase 4) | MIT backend of `make mask` (`scripts/tasks/masking.py`) |
