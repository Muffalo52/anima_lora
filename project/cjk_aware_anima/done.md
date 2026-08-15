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

## Reusable for Phase 2b onward

| Piece | Where |
|---|---|
| Teacher-side encoding | `run_bench.py::SplitTokenizeStrategy` |
| Adapter tensors without a DiT load | `build_ext.py::read_tensor` pattern — extend to `net.llm_adapter.*` |
| Bespoke distill-loop template | `scripts/distill_mod/` → `scripts/distill_cjk/` |
| Adapter injection point | `library/anima/models.py::LLMAdapter` (~:2599) |
| Tag vocabulary / entity list | `caption_index.json` (`make caption-index`) |
| OCR + text detection (Phase 4) | MIT backend of `make mask` (`scripts/tasks/masking.py`) |
