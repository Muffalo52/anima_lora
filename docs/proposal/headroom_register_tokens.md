# Headroom (register) tokens — give the DiT non-emitting scratch so it stops paying in canvas

**One-liner:** Append a few learnable, non-decoded **register tokens** to the DiT's
self-attention stream so the model has an off-canvas place to park the global
state / attention-sink it currently manufactures *in pixels* — the leading
hypothesis for unprompted white/black borders. Diagnose first (does the base even
seek headroom?), then ask whether the relaxation helps, then whether a few-hundred-
step budget can get the base to *adopt* it.

Status: **PROPOSED — not started.** Design-only; no code, no bench yet. The Phase-0
gate is a pure observation over the existing base DiT and reuses machinery we
already have.

- Planned bench: `bench/headroom/sink_probe.py` (new; Phase-0 — sigma-resolved,
  self-attn-split per-token activation-norm map over the patch grid; is there a
  relocatable sink, and does it sit on borders / low-information regions?).
- Planned network: a register-token option in `networks/` (new; Phase-1 — K learned
  tokens concatenated into the self-attn sequence, rope-exempt, stripped before
  unpatchify/decode). **Not a LoRA** — an architectural capacity add; the trainable
  surface is the register embeddings + a small attention-projection LoRA so the
  frozen base can learn to read/write them.
- Premise sources: `docs/proposal/tag_headroom_commitment_sigma.md` (the σ≈0.8 bulk-
  commitment finding this builds on, and its confound-grid discipline),
  `bench/cross_attn_drive/` (the commitment-σ / knockout probes and `how_to_observe.md`
  this extends from "which *tag* commits when" to "where does the *sink* live"),
  the ViT-registers result (*Vision Transformers Need Registers*, Darcet et al.) as
  the external mechanism, CLAUDE.md § "Text encoder padding" (padding-as-attention-
  sink — the same phenomenon one modality over) and § "The DiT operates on 5D
  latents" / "Free-fit native-shape bucketing" (the boundary + compile invariants any
  token-count change must obey).
- Priors this design must obey (be honest up front): the **per-σ reweight graveyard**
  ([[project_sigma_reshape_no_win]]) — global per-σ guidance reweights are a no-win
  line here, so this must be a *capacity add*, not a reweight; and the **base-owned
  ceiling** ([[project_x0_contradiction_bench]]) — x̂₀-wander/complexity is ~90% base-
  owned and caption-level fixes don't move it, so if borders are pure data prior a
  frozen-base+LoRA register add may not reach them. RQ1 exists to tell these apart
  *before* any training.

## The claim being made precise

Softmax attention must sum to 1: every query spends its full attention mass
somewhere. When nothing is a good match, the mass piles into a **sink** — a token
that absorbs leftover attention. ViTs were shown to *manufacture* sinks by hijacking
the lowest-information patches (uniform background), turning them into a high-norm
global scratchpad and degrading those patches locally (*Vision Transformers Need
Registers*). Dedicated register tokens — extra learnable sequence slots, discarded
at output — gave the model an off-canvas scratchpad; the artifacts moved into the
registers and the patch tokens recovered.

In an **encoder** the artifact lives in the internal features and is invisible in any
output. In a **generative DiT** the sink-bearing tokens are *decoded to pixels*, so
the artifact leaks into the image. The precise hypothesis:

> Unprompted uniform borders are the sink, rendered. A border is the lowest-frequency,
> least locally-informative region on the canvas — exactly the profile of a token the
> model would sacrifice as a spatial self-attention sink. Because low frequencies
> commit early (§ below), the sink is baked into the layout at high σ and is
> unremovable by low-σ refinement — which is why it survives to the final image.

Two facts make this concrete rather than analogy in *this* codebase:

1. **We already measured the "commits early" half.** `tag_headroom_commitment_sigma`
   + the `cross_attn_drive` probes established that **bulk content locks by σ≈0.8**,
   with only a thin tail of localized text-driven features committing through
   σ∈[0.6,0.8]. A border is bulk-frequency structure → it commits in the early band,
   consistent with "set at high σ, frozen by low σ."
2. **We already ship a load-bearing sink.** CLAUDE.md's text-encoder invariant —
   *zero-padded positions act as attention sinks in cross-attention softmax; trimming
   them produces black images* — is the identical mechanism on the text axis. Empty,
   information-free positions turn out to be computationally load-bearing. The border
   hypothesis is that same mechanism on the **image (self-attention)** axis.

The intended intervention is **at commit-time**: registers are present at high σ, so
when self-attn is establishing the low-frequency layout and looking for a sink, it
finds off-canvas slots and never allocates a border. It is deliberately *not* a low-σ
fix (there is no SNR budget left there — the graveyard prior) and deliberately *not*
a per-σ guidance reweight ([[project_sigma_reshape_no_win]]) — it is added capacity.

## Why the penalty asymmetry should make the model adopt them

ViT registers were **never supervised** — no loss says "store global state here."
They got used because they were the cheaper place to dump under the *existing*
objective: patch tokens carry a local-fidelity penalty (masked-modeling / local
losses want each to retain its own content), registers carry none. Gradient descent
migrated the scratchpad to the tax-free slots.

The same asymmetry holds for a DiT and is the reason to believe this transfers rather
than being wishful:

- **Emitted image tokens are penalized on pixels** — the flow-matching loss lands on
  exactly those tokens; sacrificing one to a uniform border costs loss on that region.
- **Registers are not decoded** — no pixel target, no penalty.

So the gradient pressure to offload the self-attn sink off-canvas exists for free.
The open question is not *whether the pressure exists* but *whether a short adapter
budget on a base that never had registers can act on it* — which is RQ3, and the
honest risk of the whole program (DINOv2 adoption emerged only with scale + long
training).

## The three research questions, as gated phases

### RQ1 — does the base seek headroom? (Phase 0, bench-only, no training)

Falsify the premise before building anything. `bench/headroom/sink_probe.py`: hook
the block forward (the **DAVE block-forward hook**, `library/inference/corrections/dave.py`,
already taps the block stream) and dump **per-token activation / attention-mass norms
across the denoising trajectory**, un-flattened back onto the patch grid. Two extra
axes make it discriminating:

- **Sigma-resolved.** Do high-norm sink tokens appear **at high σ (≈0.8–0.95),
  co-timed with the layout commit** measured in `tag_headroom_commitment_sigma`? A
  sink that only appears late (or never as a norm outlier, just as painted pixels)
  is *not* the mechanism.
- **Self-attn-split.** The sink signature must be in **self-attention** (image→image
  entropy collapsing onto a few tokens), not cross-attention (image→text). Reuse the
  `attn_contribution.py` / `attn_evolution.py` readouts from `cross_attn_drive`.

Correlate sink-token location with (a) low-information / border regions of the final
image and (b) the σ≈0.8 commit band.

**Kill criterion:** no high-norm sink tokens, **or** the sink is not co-located with
borders / low-information regions, **or** it lives in cross-attn, **or** it onsets
late. Any of these ⇒ borders are a data-prior / VAE-edge problem, not a relocatable
sink; the register line closes and the honest answer is "wants crop-conditioning or
border augmentation, not headroom." **Pass:** a high-σ, self-attn, border-co-located
high-norm sink exists — the base is manufacturing a scratchpad in pixels and there is
something to relocate.

### RQ2 — does the relaxation improve results? (Phase 1, minimal train + eval)

Only if RQ1 passes. Add K registers to the self-attn stream — and note the DiT
sweet spot is **not** the DINOv2 K=4: the DSR ablation (see § Related research) puts
it near **36 registers inserted at block ~8** (early-to-middle depth), non-monotonic,
with K=4 only a floor and K=100 a regression. So treat both **K and insertion depth**
as the ablation axes (K∈{4,16,36,64}, start-block∈{0,8,16}), not K alone. Train the register embeddings + a small
attention-projection LoRA (QKV) on the normal FM objective so the frozen base can
learn to attend to them. Evaluate two things, both against a matched no-register run:

- **Border-artifact rate (primary, near-detector-free).** Fraction of border pixels
  that are near-uniform white/black on a prompt set that requests **no** border,
  measured directly on decoded pixels — plus its internal correlate, the RQ1 sink-norm
  on border tokens, which should *fall* if the register absorbed it.
- **No general regression.** CMMD (distribution-level, not per-image — the metric
  trap) plus a blind eyeball grid at real 28-step SDE / target CFG for saturation,
  pose, and composition. The point is a *relaxation*, so the bar is "borders down,
  nothing else worse," validated by eye, not a scalar alone.

**Gate:** measurable border reduction with the sink-norm moving into the registers,
and no quality/diversity regression on the grid. A "borders down but outputs blander"
result is a **fail** — it would mean the register removed a load-bearing computation
rather than relocating it.

### RQ3 — can the base *adopt* it in a few hundred steps? (Phase 2, the hard gate)

The real risk, measured as a **training-time trajectory**, not an endpoint. Log, over
steps: register-token norm / attention-mass **rising** while border-token sink-norm
**falls**. That crossover *is* adoption. Because RQ2's training surface is tiny
(K·D register params + attention LoRA), the honest hope is a few hundred steps; the
honest fear is that a frozen base whose attention never knew registers existed can't
relocate a baked-in sink on that budget.

**Gate:** the adoption crossover appears within the target budget (≈a few hundred
steps) and is stable (registers don't collapse back to dead tokens, borders don't
return). **If it doesn't:** the fallback is a fuller finetune — unfreeze attention QKV
so the sink can actually move, or train registers into the base over a long schedule.
That is a *different, more expensive experiment* and must be reported as such, not
smuggled in. A negative here is a real, publishable result: "registers help but only
above the LoRA budget," which directly informs whether headroom belongs in the base
or in an adapter.

## Integration wrinkles specific to this repo (do not hand-wave)

- **Compile budget / free-fit coupling.** Registers add a constant K to every
  sequence, so `_native_flatten`'s token-count key shifts by K and each tier's
  `EDGE_TOKEN_BANDS` seq range must be widened by K. `train.py::_derive_token_budget`
  and `compile_blocks(n_token_families=…)` derive the dynamo budget from the buckets
  the caches populate — a constant offset is benign but must be threaded through, or
  `compile_dynamic_seq` will graph-miss on every forward. K is fixed (not per-image),
  so this stays one graph per tier.
- **Rope exemption.** Registers are non-spatial; they must be **excluded from RoPE**
  (like a CLS token), not assigned patch positions, in `networks/attention_dispatch.py`.
  Getting this wrong gives them a spurious spatial location and defeats the point.
- **Strip before decode.** Registers concat at the sequence level *after* patch-flatten
  and must be dropped before unpatchify → VAE decode. Mind the 5D↔4D boundary
  (CLAUDE.md § dim-2 singleton): the strip happens in the flattened `(B,1,seq,1,D)`
  domain, before the grid is restored.
- **Harness ordering.** Build via `library/runtime/harness.py::build_anima`
  (load → apply → compile) — the register concat is part of "apply," so compile must
  trace it, same rule as any adapter.

## Risks / priors (be honest)

- **Base-owned ceiling** ([[project_x0_contradiction_bench]]). If borders are dominated
  by the data prior (letterboxed/matted training art) rather than the sink mechanism,
  a frozen-base register add won't move them — the sink is real but small next to the
  prior. RQ1's border-vs-sink correlation is the guard; a weak correlation predicts a
  weak RQ2 and we say so.
- **Per-σ reweight graveyard** ([[project_sigma_reshape_no_win]]). This must not
  degenerate into "boost/suppress something in a σ band." It is a capacity add; if the
  only way to make it work turns out to be a σ-gated register gain, it inherits that
  no-win result and should stop.
- **Adoption may need scale** (the DINOv2 caveat, and RQ3 itself). The whole
  mechanism emerged with long training on a from-scratch structure; the LoRA-budget
  bet may simply lose. This is why RQ3 is a gate, not an afterthought.
- **Registers could learn to do nothing.** A frozen base can leave the new tokens as
  inert appendages the FM loss never routes into. RQ3's rising-register-norm signature
  is exactly the check that adoption happened rather than the loss ignoring the new
  capacity.
- **Removing a load-bearing sink can hurt.** If the border sink is doing real global
  bookkeeping and the register fails to absorb it cleanly, outputs get *worse*, not
  border-free. RQ2's "no-regression" gate and the sink-norm-moves-into-registers
  correlate are the guard against shipping a bland-but-borderless model.

## Success criterion

A sigma-resolved, self-attn-split demonstration that (1) the base manufactures a
border-located attention sink at high σ (RQ1), (2) K non-decoded register tokens
absorb that sink and measurably cut the unprompted-border rate with no quality or
diversity regression (RQ2), and (3) the base adopts them — register norm up, border
sink-norm down — within a few-hundred-step budget (RQ3). Any phase's gate failing is
an honest, useful negative: RQ1-fail redirects to a data-prior fix, RQ2-fail says the
sink wasn't the border's cause, RQ3-fail says headroom belongs in the base, not an
adapter.

## Related research

**Taming Outlier Tokens in Diffusion Transformers** (Wu, Wang, Fu, L.-C. Chen, Gan,
Wei; Rice + Apple; arXiv:2605.05206, May 2026) is direct prior art for the general
mechanism and should be read before starting. It introduces **Dual-Stage Registers
(DSR)** — register tokens in *both* a ViT-based encoder and the diffusion transformer.
The DiT half ("diffusion registers": a few trainable, non-decoded tokens learned
jointly with the generator, stripped at inference) is the RQ1→RQ2 intervention,
already validated across SiT, JiT, RAE, VAE-latent and pixel-space DiTs. It **de-risks
our premise** rather than threatening it — several load-bearing assumptions here are
now measured facts, not hopes:

- **RQ1 premise confirmed.** Outlier/attention-sink tokens *do* arise inside DiT
  generators, not just ViT encoders — the phenomenon we set out to prove exists.
- **RQ2 direction confirmed, with hyperparameters.** Trained diffusion registers
  consistently improve generation (e.g. RAE-SigLIP2 FID 5.89→5.33, VAE-SiT
  16.05→14.47, GenEval 0.426→0.466; ~4× faster convergence). Sweet spot **~36
  registers at block ~8**, non-monotonic in both count and depth — folded into RQ2
  above.
- **"Strip the sink" is a dead end.** Masking high-norm tokens does *not* help
  (their Tab. 1): the outlier is a *symptom of corrupted local patch semantics*, not
  the cause. This upgrades our "removing a load-bearing sink can hurt" risk from a
  hedge to a measured result — registers must *absorb/restore*, never remove.

**What DSR does not do — and where this proposal still adds signal:**

- **No bridge to a real, named failure mode.** DSR observes outliers as an *awkward
  pattern in latent/attention norm maps* and optimizes distributional scores (FID,
  GenEval). It never connects the sink to a *specific, visible, nameable* generation
  defect. Our whole angle — **is the sink the unprompted white/black border?** — and
  the border-rate metric is untouched by them. That the pattern is real in latent
  space is exactly the grounding that makes the border-correlation worth testing:
  they proved the mechanism exists; we ask what it *costs the user*.
- **No frozen-base cheap-adoption regime (RQ3).** Every DiT register in DSR is trained
  *jointly with the generator from scratch* (ImageNet 80–800 epochs; 10k steps @
  batch 2048 for T2I). Retrofitting a **frozen pretrained DiT** (Anima's setting) with
  registers on a **few-hundred-step adapter budget** is unaddressed — RQ3 is the
  genuinely open, Anima-relevant question, and DSR's from-scratch "4× faster" says
  nothing about it.

**One finding that pushes back on our temporal story (fold into the RQ1 probe).** DSR
reports DiT outliers concentrate in **intermediate layers** and *strengthen as noise
decreases* — strongest at low noise (t≈0.1, late denoising), weakest at high noise.
That is the **opposite σ direction** to our "sink commits at high σ / early, in the
σ≈0.8 layout-commit band" hypothesis. Either the norm-outlier and the layout-commit
are distinct phenomena, or the early-commit border story is wrong. So the RQ1 probe
must sweep **layer-depth × σ**, not just high σ, and be honestly prepared to find a
late-σ / intermediate-layer signature — in which case the border-commit-early
mechanism needs revising even if registers still help.

## References

- Wu, Wang, Fu, Chen, Gan, Wei, *Taming Outlier Tokens in Diffusion Transformers*
  (arXiv:2605.05206, 2026) — Dual-Stage Registers; the DiT-generator prior art above
  (register count/depth sweet spot, masking-fails result, intermediate-layer / low-
  noise outlier localization).
- `docs/proposal/tag_headroom_commitment_sigma.md` — the σ≈0.8 bulk-commitment finding
  and confound-grid discipline this builds on.
- `bench/cross_attn_drive/` — `how_to_observe.md`, `attn_contribution.py`,
  `attn_evolution.py`, `knockout_diffmap.py`; the commitment-σ / attention-readout
  machinery the Phase-0 sink probe extends.
- `library/inference/corrections/dave.py` — the block-forward hook the norm probe
  reuses as scaffold.
- `library/runtime/harness.py::build_anima` — load→apply→compile harness the register
  network must build on.
- CLAUDE.md § "Text encoder padding" (padding-as-attention-sink, the same mechanism on
  the text axis), § "The DiT operates on 5D latents", § "Free-fit native-shape
  bucketing" (the compile/token-count invariants a fixed +K must respect).
- *Vision Transformers Need Registers* (Darcet, Oquab, Mairal, Bojanowski) — the
  external mechanism: emergent high-norm artifact tokens in low-information patches,
  resolved by unsupervised, discarded-at-output register tokens.
</content>
</invoke>
