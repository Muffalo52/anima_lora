# CJK-aware Anima — Phase 2b report

Measured verdicts from the distillation loop's unit gates (2026-08-15).
Run envelopes: `bench/cjk_distill/results/`. Code: `scripts/distill_cjk/`.

*Line home: [`motivation.md`](motivation.md) (why) · [`done.md`](done.md)
(what exists) · [`plan.md`](plan.md) (design, phases, gates).*

## What ran

| Gate | Question | Verdict |
|---|---|---|
| **G0b** `--mode oracle` | student ids := teacher ids ⇒ loss ≡ 0? | **PASS** — worst `1-cos` 2.9e-4 (bf16 floor). Also certifies the trimming invariant the cache rests on: a non-pad adapter output does not depend on how many pads follow it. |
| **G1** pytest | EN bit-identical? | **PASS** — pure-EN text tokenizes to stock spiece ids exactly; the split embedding returns stock rows bitwise, before *and* after the ext parameters move. |
| **G0** `--mode capacity` | can ext rows *express* the teacher at all? | **PASS** — 32 pairs, loss 0.574 → **0.0244**, monotone. No escalation to 2-ii (adapter LoRA) needed. |
| **G2** `g2.py` | which loss × which parameterization? | **PARTIAL** — see below. The design questions were answered; the headline metrics were not trustworthy and are withdrawn. |

## G2 — what survives

6 arms, staged (loss chosen at `param=global`, then parameterization at the
winning loss), 1500 steps × batch 32, 5,711 training pairs (`tags` +
`tags_alt`; D6 excluded — its teacher is degraded by construction).

- **`flat` is disqualified.** Highest flat recovery of any arm (0.203) while
  far-discrimination went 0.096 → **0.321** and near-discrimination 0.711 →
  **0.970**. It buys its score by pushing every prompt's conditioning toward
  one direction — the collapse mode `plan.md` lists as risk 6. Both halves of
  the stratified discrimination caught it.
- **The global correction does the work; per-row residuals do not.**
  `global` 0.079, `global_row` 0.080, **`row`-only 0.062** — per-row freedom
  adds ~0.001, and removing the shared map *costs* 0.017. This confirms the
  2-i-a hypothesis: the zero-shot table's error is systematic (the anchor map
  was fit on non-CJK anchors), not 58,968 independent per-row errors. It also
  means the 95% of rows the corpus never visits still move.
- **Every honest arm plateaus by step 500** (0.072 @500 → 0.075 @1500). Not
  undertrained.
- **Train-vs-held gap is real**: `attn+span` 0.062 train vs 0.154 held (2.5×),
  `attn` alone 0.019 vs 0.098 (5×). Consistent with the corpus-scale problem —
  5,711 pairs out of 3,008 local captions, 2,661 ext rows visited.

## G2 — what is withdrawn

**Both recovery metrics were broken, in opposite directions.** No arm ranking
between `span`, `attn` and `attn+span` should be quoted from this run.

- **`recovery` (flat space) understates.** It is built on the position-wise
  cosine, where the teacher gets a free ride: the teacher's T5 side *is* the
  EN reference's token sequence (only the Qwen side differs), while the
  student's segmentation differs by construction. The student is charged for a
  difference the DiT cannot observe — the same objection that demoted `L_flat`
  from an objective to a control, which we failed to apply to the metric.
- **`recovery_attn` (readout space) overstates, and so does the `attn`
  objective.** It reported 0.51 zero-shot → 0.96 after 30 steps. The control
  kills it: in that space, unrelated prompts sit at **0.374 → 0.738** (the
  floor rises *during training*), and two different captions of the same image
  land at **exactly 1.000**. The space is blind to wording — and therefore to
  glyph identity, the axis this whole line exists to serve.

**Root cause**: the probe bank used **random query directions**. `plan.md`
specifies real cached image-token queries at 2–3 σ levels; only the K/V
projections and the q/k RMSNorm gains were taken from the checkpoint. A random
query attends almost uniformly, so the readout degenerates to a near-mean over
the sequence and washes out exactly the fine structure that matters. This
contaminates `L_attn` as a *training objective*, not just as a metric.

## Next

1. **Build the real query bank** — tap cross-attn queries from a few DiT
   forwards at 2–3 σ levels, cache a few hundred, and rebuild `attn_bank`.
   Both the metric and the objective depend on it.
2. **Re-run G2** on the corrected bank. Only then is the loss ranking real.
3. Then the corpus decision (widen D1 from an HF-hosted tag dump) with a
   trustworthy signal to measure the widening against.

Trustworthy in the meantime: held-out **span** loss (0.645 → 0.116) and
**flat-space far-discrimination** as a collapse guard.
