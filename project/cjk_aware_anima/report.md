# CJK-aware Anima — Phase 2b report

Measured verdicts from the distillation loop's unit gates (2026-08-15, G2
re-run 2026-08-16). Run envelopes: `bench/cjk_distill/results/`. Code:
`scripts/distill_cjk/`.

*Line home: [`motivation.md`](motivation.md) (why) · [`done.md`](done.md)
(what exists) · [`plan.md`](plan.md) (design, phases, gates).*

## What ran

| Gate | Question | Verdict |
|---|---|---|
| **G0b** `--mode oracle` | student ids := teacher ids ⇒ loss ≡ 0? | **PASS** — worst `1-cos` 2.9e-4 (bf16 floor). Also certifies the trimming invariant the cache rests on: a non-pad adapter output does not depend on how many pads follow it. |
| **G1** pytest | EN bit-identical? | **PASS** — pure-EN text tokenizes to stock spiece ids exactly; the split embedding returns stock rows bitwise, before *and* after the ext parameters move. |
| **G0** `--mode capacity` | can ext rows *express* the teacher at all? | **PASS** — 32 pairs, loss 0.574 → **0.0244**, monotone. No escalation to 2-ii (adapter LoRA) needed. |
| **G2** `g2.py` | which loss × which parameterization? | **PASS** (2026-08-16) — `span` at `param=global`. The 2026-08-15 attempt was withdrawn; three instrumentation defects had to be fixed first, below. |

## The three instrumentation defects

The first G2 answered its design questions but its headline metrics were not
trustworthy, and the diagnosis recorded at the time was itself wrong. All three
faults are fixed; the numbers in the next section are from the corrected loop.

**1. The probe queries were random.** `attn_bank` synthesized them with
`torch.randn`. Queries are `q_proj(image tokens)` — they only exist during a
forward and cannot be read out of a checkpoint. `build_query_bank.py` now taps
the real thing: 32 DiT forwards on cached latents and cached post-adapter
contexts at σ ∈ {0.3, 0.6, 0.9}, hooking each sampled block's
`cross_attn.q_norm`, banking 768 real token queries per block into
`bench/cjk_distill/assets/query_bank.safetensors`. `build_bank` refuses to run
without it (`--allow_random_queries` reproduces the withdrawn run). Two traps
worth knowing: `load_anima_model`'s `device=` does not reach the runtime
buffers, so a directly-loaded DiT keeps a CPU mod-guidance schedule and dies in
`_run_blocks` (the shared harness's `.to(device)` is load-bearing); and a suffix
match on `blocks.0.cross_attn.q_norm` hits **`LLMAdapter`'s own block 0** first,
which is a text→text attention that does not even run on this path.

**2. The recorded root cause was wrong, and real queries alone made things
worse.** The withdrawn report blamed near-uniform attention ("a random query
attends almost uniformly, so the readout degenerates to a near-mean over the
sequence"). Measured: attention is **sharp** under both banks — 3–9 effective
tokens of ~99 real ones, with 80–87% of the softmax mass sitting on the zero-pad
sink. The actual defect is that every readout carries a large **common offset**
(`‖mean‖/‖vec‖` = 0.73 with random queries, **1.02** with real ones), and an
*uncentered* cosine over vectors sharing an offset that big saturates at 1 for
everything:

| step-0 metric | random q | real q, raw | real q, centered |
|---|---|---|---|
| `cos_native_vs_en_attn` (the floor) | 0.287 | 0.997 | **−0.031** |
| `cos_teacher_vs_en_attn` (the ceiling) | 0.916 | 0.999 | 0.841 |
| far discrimination, readout space | 0.374 | 0.999 | 0.269 |

The offset is real conditioning, but it is *shared* — teacher and student both
have it, so matching it is free and carries no information. `fit_centers` fits it
once on the frozen teacher outputs (arm-independent, batch-independent) and
`readout()` projects it out. This repaired `L_attn` as an **objective** as well
as a metric: uncentered, two unrelated prompts read 0.997 alike, so the loss had
almost no dynamic range about wording.

**3. The holdout was split by pair, not by image.** Every image contributes
several pairs that share tag content and differ only in wording (`tags` /
`tags_alt`, plus D6's two quote registers). A per-pair shuffle therefore
(a) **leaked** each held-out pair's sibling register into training ~91% of the
time — "held-out" was measuring generalization to new *wording of trained
content*, not to new content — and (b) left **exactly one** near pair in the
256-record eval slice. `discrimination_near` was a single-sample statistic; that
is the provenance of the **0.71 zero-shot near figure** previously quoted in
`plan.md`. Split by image it is **0.411 over 72 pairs**. `load_pairs` now groups
by image and logs the near-pair count so this cannot silently regress.

## G2 — the measured cross-tab

6 arms, staged (loss chosen at `param=global`, then parameterization at the
winning loss), 1500 steps × batch 32, 5,728 training pairs (`tags` +
`tags_alt`; D6 excluded — its teacher is degraded by construction).
Envelope: `bench/cjk_distill/results/20260816-1152-g2/`.

| param | loss | recovery_attn | disc far | disc near | held span | held attn | held flat | held pool |
|---|---|---|---|---|---|---|---|---|
| _(zero-shot)_ | — | 0.516 | 0.111 | 0.411 | 0.654 | 0.623 | 0.899 | 0.445 |
| global | `flat` | 0.758 | **0.304** | **0.910** | 0.590 | 0.392 | 0.753 | 0.279 |
| global | `span` | 0.974 | 0.085 | 0.394 | **0.120** | 0.123 | 0.859 | 0.163 |
| global | `attn` | 0.967 | 0.088 | 0.380 | 0.334 | **0.082** | 0.876 | 0.172 |
| global | `attn+span` | 0.962 | 0.094 | 0.392 | 0.173 | 0.076 | 0.869 | 0.165 |
| global_row | `span` | **0.975** | 0.089 | 0.392 | 0.107 | 0.120 | 0.858 | 0.162 |
| row | `span` | 0.923 | 0.101 | 0.408 | 0.338 | 0.318 | 0.879 | 0.280 |

- **`flat` is disqualified**, and now on a near metric that can actually see it:
  far 0.111 → **0.304** and near 0.411 → **0.910**. It buys recovery by pushing
  every prompt's conditioning toward one direction — risk 6 in `plan.md`. Both
  halves of the stratified discrimination catch it. (In the readout space its
  near reads exactly 1.000, so the *flat*-space discrimination is the collapse
  guard that matters.)
- **`span` wins, and the cross-tab is what shows it.** Each objective wins on
  its own held-out term, as expected — the question is what it costs on the
  others. `span` scores 0.123 on the attn term against `attn`'s own best 0.082;
  `attn` scores 0.334 on the span term against `span`'s own best 0.120. Span
  transfers, attn does not. `span` also wins `recovery_attn` (0.974 vs 0.967) —
  on attn's home turf. Adding attn to span (`attn+span`) does not help
  (0.962). **Ship `span`.**
- **The global correction does the work; per-row residuals do not.** `global`
  0.974, `global_row` 0.975, **`row`-only 0.923** — per-row freedom adds 0.001,
  removing the shared map costs 0.051. This reconfirms the 2-i-a hypothesis with
  trustworthy metrics: the zero-shot table's error is systematic (the anchor map
  was fit on non-CJK anchors), not 58,968 independent per-row errors — so the
  95% of rows the corpus never visits still move. `global_row` is nominally the
  winner but is inside the noise of `global` at 1,887 extra tunable rows;
  **prefer `global`** unless a later corpus makes the residuals earn their keep.
- **Discrimination stays healthy on every honest arm**: far ends at 0.085–0.101,
  *below* the 0.111 zero-shot baseline and far below the 0.2 gate. Near ends at
  0.380–0.408, i.e. wording still reaches conditioning.

**Do not read `recovery_attn` ≈ 0.97 as "97% done."** The readout is a heavy
compression (64 queries × 3 blocks) and is permutation-invariant by design, so a
student that carries the right content under a different segmentation scores
high there and low in flat space — which is exactly what happens: flat recovery
is **0.066** and `cos_student_vs_en` is 0.096 against the teacher's 0.777. The
Phase-2c gate is the existing bench's `cos_vs_en ≥ 0.6` on rendered prompts, and
nothing here says that gate is close. What G2 settled is the *design* — which
loss, which parameterization — not the distance to 2c.

## Next

1. **Corpus.** Every honest arm plateaus early and the train-vs-held gap is the
   binding constraint: 5,728 pairs from 3,008 local captions, 2,661 ext rows
   visited of 58,968. **D2 is in the mix as of 2026-08-16** — first slice
   measured below.
2. **2c** on the widened corpus with `param=global`, `loss=span`.
3. The **owed D6 instrument** (same template, different quoted strings) is still
   owed — near-discrimination now has 72 pairs, but none of them vary a *quoted
   string*, so glyph contrast remains unmeasured.

## D2 — what the commentary corpus buys (2026-08-16)

First measured slice of D2: **9,068 pairs** (5,721 JA→EN by Hy-MT2-7B greedy +
3,347 free human translations), from a partial MT pass stopped at ~6k cached
rows out of 69,668 candidates. Envelopes:
`bench/cjk_distill/results/20260816-1400-g2` (control) and
`…-1409-g2` (+D2). Both share one cache (18,090 train / 900 holdout) and one
seed; the only free variable is `--train_registers`.

**Coverage — D2 more than doubles the reachable table.** Ext rows visited by
the corpus: **3,002 → 6,394** of 58,968 (5.09% → 10.84%) for +9,068 pairs. Most
of the new rows are thin (1–4 visits: 933 → 2,463), but the 5–49 band also
doubles (1,106 → 2,411), so this is not only a tail.

| arm | train regs | pairs | recovery_attn | far | near | **cos(s,en) commentary** | cos(s,en) tags | held span | held attn |
|---|---|---|---|---|---|---|---|---|---|
| _(zero-shot)_ | — | — | 0.511 | 0.087 | 0.480 | 0.081 | 0.058 | 0.642 | 0.417 |
| `span` | tags,tags_alt | 5,730 | 0.886 | 0.068 | 0.449 | 0.097 | **0.100** | **0.128** | 0.171 |
| `attn+span` | tags,tags_alt | 5,730 | 0.868 | 0.065 | 0.442 | 0.096 | 0.096 | 0.166 | 0.146 |
| `span` | +commentary | 14,356 | 0.910 | 0.069 | 0.450 | 0.098 | 0.100 | 0.134 | 0.176 |
| `attn+span` | +commentary | 14,356 | **0.953** | 0.069 | 0.450 | **0.109** | 0.097 | 0.190 | **0.115** |

- **D2 is structurally inert under the settled `loss=span`.** Prose has no
  tag-by-tag alignment, so D2 pairs carry no `spans` and contribute *zero*
  gradient to the span term — their only effect there is to dilute the batch
  (~39% span-carrying rows instead of 100%). The `span` +D2 row moves
  commentary 0.097 → 0.098, i.e. nothing. **A corpus addition is not a mix
  question until the objective can consume it.**
- **Under a sequence-level term it works, in its own register.** `attn+span`
  +D2 lifts commentary **0.096 → 0.109 (+13%)** and leaves tags flat
  (0.096 → 0.097). It buys prose conditioning; it does **not** transfer to the
  tag register, which is what the 2c gate is scored on.
- **Read `recovery_attn` 0.953 with the holdout in mind.** The 900-pair holdout
  is ~48% commentary, so the D2-trained arm is partly being graded on its own
  domain. The per-register decomposition above is the honest column, and it is
  why the headline flip (`attn+span` 0.953 > `span` 0.910, reversing G2's
  verdict) is **not** grounds to re-open G2: on `tags`, `span` still wins.
- **Discrimination is unharmed**: far 0.065–0.069 across every arm, below the
  0.111 zero-shot baseline and far under the 0.2 gate. Near stays ~0.45.
- The 2c gate is untouched by this: `cos_vs_en` is 0.10 against a 0.6 target.

**What this settles.** D2 is worth finishing (the coverage doubling is real and
the register gain is real), but shipping it requires an objective change, not
just more data — either keep a sequence-level term in the mix for span-less
registers, or find an alignment for prose. Both are Phase-2c decisions.
