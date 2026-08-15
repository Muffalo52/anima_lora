# CJK-aware Anima — plan (Phase 2 onward)

*Line home: [`motivation.md`](motivation.md) (why) ·
[`done.md`](done.md) (what is measured and built) ·
[`report.md`](report.md) (Phase 2b measured verdicts).*

Status: Phase 2a (corpus) built and measured; **Phase 2b built and run** —
gates G0/G0b/G1 passed, G2 partial (see [`report.md`](report.md)). 2c onward
is unstarted. Scope is **Japanese only**; zh/ko are "rerun the same pipeline
with a different corpus" once ja passes.

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

- **L_flat** — cosine on the full padded `[512, d]` output. Control only. Its
  first rationale was wrong: pads are **zeroed** after the adapter
  (`inference/text.py:229`), so they carry no Qwen signal — `ja_native`'s 0.02
  is 3 informative positions plus a norm-ratio penalty.
- **L_attn (primary)** — the DiT's cross-attn applies **no RoPE to the
  context** (`models.py:385`), so it consumes the output permutation-
  invariantly, while the `512 − N` zero pads are unmasked sink mass whose
  weight depends on N: the object is a *set plus a length*. Match
  `Attn(Q_probe, K(out), V(out))` for a fixed probe-query bank: K/V read
  straight out of the DiT safetensors, sink folded into the softmax
  denominator analytically. **The queries must be real cached image-token
  queries** (a few DiT forwards at 2–3 σ) — substituting random directions
  degenerates the readout to a near-mean over the sequence and makes both the
  loss and the metric nearly vacuous, measured in [`report.md`](report.md).
- **L_span** — D1 captions are *composed* tag-by-tag, so EN↔JA alignment is
  free and exact; supervises each ext row from its own tag instead of a
  sequence average, and is the only loss that can carry a per-wording trust
  weight.
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

## Data sources not yet in the mix

Built and delivered (D0 wiki, D1 captions, D5 Wikidata, D6 quoted-text, D7
LoveHina) are in [`done.md`](done.md). Still available to widen the corpus:

| ID | Source | Direction | Domain fit | Est. size | Notes |
|---|---|---|---|---|---|
| D2 | Danbooru/pixiv artist commentary (native JA on the same images) | JA→EN via local LLM | exact, native phrasing | 10⁴–10⁵ | **BLOCKED**: `danbooru.donmai.us` unreachable from this machine (network-level). Any route must go through an HF-hosted dump. Risk 3 wants it at ≥ ~20% of the mix. |
| D3 | STAIR Captions (~820k JA on COCO, aligned to EN COCO captions); YJ Captions (~131k) | pre-aligned | caption-domain, photographic | ~1 M | Real parallel captions, zero MT — the right source for sentence register (particles, verb endings). **Check licenses before mixing** (STAIR is CC BY 4.0 last checked; verify). |
| D4 | JESC (~2.8 M subtitle pairs) | pre-aligned | casual dialogue | subsample | Matches the speech-bubble / quoted-text register. Subsample; don't let it dominate. |

Ja-only scoping: the corpus is ja-only, but the ext table's zh/ko rows stay
physically present (inert without zh/ko input text). The distilled sidecar
ships per-row provenance (`trained` / `zero-shot`) in the mapping JSON.
**Never-visited rows stay at zero-shot init but get flagged**; if eval shows
they poison nearby generations, demote them to the `<unk>` fallback in the
encoder mapping rather than shipping noise rows (a JSON edit).

## Phases & gates

- **Phase 2a — corpus. BUILT** (`post_image_dataset/cjk_distill/`; measured in
  [`done.md`](done.md)). Two items left:
  - **OPEN — user sign-off. Hard blocker on 2c.** 34% of tag *occurrences* are
    `mt_unverified` (`colored inner hair` → 色付きの陰毛, "colored pubic hair" —
    its own back-translation says so), and the composed caption is the
    *student's* input, so a bad wording trains that tag's ext rows toward the
    wrong meaning. `assets/tag_glossary_review.md` (200 rows ⇒ ~64% of
    occurrences) + `spotcheck.md`; fixes → `datasets/tag_overrides.json`. Until
    then `--trust provenance` demotes those spans instead of dropping pairs.
  - **D6 demoted to eval-only** (39% of pairs): `quote_preserved` puts the raw
    JA string on the *teacher's* stock-spiece side (→ `<unk>`) and
    `quote_translated` replaces it with `[TEXT]`, so both teach the ext rows to
    be vacuous exactly where verbatim identity matters. Glyph identity is
    Phase 4's job. Default `--train_registers tags,tags_alt`. **Owed
    instrument**: D6's two registers share their JA text verbatim, so they
    cannot measure glyph contrast at all — that needs pairs on the *same*
    template with *different* quoted strings (one extra field out of
    `build_pairs.py`, then a cache rebuild).
  - **The corpus is 9,922 pairs, not the 10⁵–10⁶ the cost model assumed** — D1
    is exhausted at the 3,008 local captions. Composition is CPU-only, so scale
    is bounded by EN caption text, not GPU: widen from an HF-hosted tag dump
    before 2c, but after G0/G2 (they may redirect the phase).
  - **Not run: the `natural` prose register.** Implemented and validated on 12
    captions, then deliberately not run — ~3 h of GPU to produce *MT-invented*
    Japanese, when the sentence-register coverage it buys is better bought from
    D3's human-written JA. Decide before 2c.
- **Phase 2b — loop + unit gates. BUILT** — `scripts/distill_cjk/`
  (`make exp-cjk-cache` → `exp-distill-cjk`), `tests/test_cjk_distill.py`.
  Corpus caches once (Qwen hidden + frozen-teacher output, trimmed to non-pad);
  teacher and student are the *same* frozen adapter differing only in which
  rows their ids reach. Trainable surface is a ladder: **2-i-a `global`**
  (low-rank + diag + gain correction shared by every ext row — so the 95% of
  rows the corpus never visits still move, and the build-time ×1.66 rescale
  becomes a learned gain) → **2-i-b `global_row`** (+ per-row residuals above a
  visit floor; 933 of 3002 visited rows are seen 1-4×) → 2-ii. Provenance ships
  in three tiers: `tuned` / `mapped` / `zero-shot`. Gates, in order:
  - **G0b `--mode oracle`** — student ids := teacher ids ⇒ loss ≡ 0. **PASSED**
    (worst 1-cos 2.9e-4 = bf16 floor), which also certifies the trimming
    invariant the cache rests on.
  - **G0 `--mode capacity`** — overfit 32 pairs. Settles 2-i vs 2-ii for
    minutes of GPU; run before any corpus work.
  - **G1** — EN bit-exactness: pure-EN ids ≡ stock spiece, and the split
    embedding returns stock rows bitwise before *and* after the ext params
    move. **PASSED** (pytest).
  - **G2** — loss × parameterization cross-tab, every arm scored on every
    metric so no arm wins on its own objective.
  - **G3 / G4** — teacher ceiling per register (is 0.6 even the right 2c
    number?); corpus health: token-count ratio, occurrence-weighted provenance.
  Headline metric is **recovery**, not raw cosine — `(cos(student,en) −
  cos(native,en)) / (cos(teacher,en) − cos(native,en))` over a held-out slice
  (each pair gives the whole triangle); the shared Qwen context puts an
  untrained student well above zero, so read everything against step 0.
  **Discrimination is stratified, and the two halves read in opposite
  directions.** `far` (different images) is the pathology gate — it is what
  the probe's 0.2 was measured on, between two maximally distant prompts.
  `near` (same image, `tags` vs `tags_alt`) is *supposed* to be high: two
  captions that differ only in wording should land on nearly the same
  conditioning, and a single threshold over arbitrary pairs punishes that.
  What `near` is good for is its distance from **1.0** — that residue is the
  only evidence that wording, and downstream the individual glyphs an OCR
  caption carries, reaches conditioning at all. Exactly 1.0 would mean
  per-character identity is invisible, i.e. the failure this whole line exists
  to prevent. Measured: zero-shot sits at far 0.10 / near 0.71.
- **Phase 2c — train + eval.** Acceptance harness is **the existing bench
  unchanged**: `run_bench.py --ext --languages ja` pointed at the distilled
  sidecar. Gates:
  - `ja_ext` cos_vs_en ≥ **0.6** on both content prompts (teacher sits at
    0.69/0.77 — get within noise of it; stretch: match it),
  - discrimination **far** ≤ **0.2** (the near half is not gated — see 2b),
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
   loss-correctness issue — see
   [`datasets/README.md`](datasets/README.md#wikidata_lexiconpy--proper-nouns).
5. **Row-norm / manifold**: zero-shot rows needed a ×1.66 norm rescale;
   training may drift norms — log row-norm distribution vs the original
   table as a diagnostic.
6. **Discrimination as a trap**: optimizing cos-to-teacher could collapse
   diversity if the corpus is template-heavy — the discrimination gate (≤0.2)
   and register variants guard this.
