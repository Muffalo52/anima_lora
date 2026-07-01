# Headroom (register) tokens — Phase 2: real-budget adoption sweep on the frozen base

**One-liner:** The cheap-adapter register retrofit failed (RQ3 negative at K=16 /
400 steps / rank-8 QKV-LoRA), but the failure was measured at a **single untested
ablation point** and training turns out to cost ~2 it/s on the eager bench harness
— so the DSR sweet-spot configuration (K≈36, mid-block insertion, unfrozen QKV,
~4000 steps) is a **~35–60 min-per-arm sweep, not a "different, more expensive
experiment."** This doc specifies that sweep: adoption-gated, drift-controlled,
framing-checked.

Status: **rewritten 2026-07-02** after Phase-0 / Phase-0.5 / RQ2-proxy results
(full detail: `bench/headroom/README.md`; original border-hypothesis design in git
history of this file). Memory: [[project_headroom_registers_rq1_falsified]],
[[project_headroom_registers_rq3_negative]].

## Settled facts this design stands on

Everything below is measured on Anima, not hoped:

1. **The border line is CLOSED (RQ1 falsified).** Unprompted borders are a
   text-controllable data prior (induce 5/5 / erase 5/5 by prompt at fixed noise);
   the sink is present identically with and without a border. Registers are not a
   border fix; don't re-propose that bridge.
2. **The DSR sink is real on Anima.** A sparse (~0.2% of tokens) ~14–24×
   high-norm ‖x‖ outlier (magnitude is prompt-dependent — keep eval prompt sets
   fixed), mid-layer, high-σ, self-attn-written. `bench/headroom/sink_probe.py`.
3. **The sink is load-bearing and global.** Norm-clamping ~7 tokens re-plans the
   whole image with no quality win (DSR Tab. 1 replicated). Registers must
   *absorb*, never remove. `sink_intervention.py`.
4. **No adoption at the cheap point.** K=16 registers + rank-8 QKV-LoRA on all 28
   blocks, 400 steps: registers reach only ~1.1× median norm (sink needs ~14–24×);
   patch sink barely moves (arm B −10%). `train_registers.py` / `register_eval.py`.
5. **Fixed-zero registers are a LESION, not a control.** 16 zero-content tokens
   share one bias-derived key = a uniform attention-mass drain; the LoRA trains
   around it and **framing/crop collapses** (3/3 broken headless crops on the
   repro prompt; 0/3 for LoRA-only and learnable-register arms). Two lessons:
   (a) causal evidence the self-attn mass economy owns **global layout** —
   consistent with the sink being layout bookkeeping, and with DSR's Theory 2
   (registers must carry content); (b) **never ship or control with fixed-zero
   registers.** The honest drift control is **arm L** = LoRA-only, K=0 (supported:
   `--num_registers 0`).
6. **Everything else observed so far is generic LoRA drift.** Matched-seed
   re-plan magnitude (~0.13–0.23 L1) and the clamp-sensitivity drop are identical
   for LoRA-only and register arms. The only cleanly register-specific effect at
   the cheap point is a small sink-ratio drop (armB 12.9 vs armL 14.2),
   downstream-inert. Don't cite trained-arm visual change as register evidence.
7. **Pixel metrics are blind to framing failures.** Point 5 was caught by eye,
   not by pixel L1 / Laplacian medians. Every Phase-2 eval includes an explicit
   matched-seed **framing/coherence montage** as a mandatory gate artifact.

## The open question (and why it's now cheap)

DSR's sweet spot — **~36 registers inserted at block ~8** (non-monotonic in both
K and depth; K=4 a floor, K=100 a regression), trained **jointly with unfrozen
attention** — is untested on a frozen pretrained base. Our negative sits at
(K=16, insert-at-entry, rank-8 LoRA, 400 steps) — plausibly under-provisioned on
*every* axis at once, and the reachability argument (a rank-8 LoRA cannot rewire
where softmax mass flows against a baked-in ~20× attractor) points at unfrozen
QKV as the axis most likely to matter.

The bench harness trains at ~2 it/s (eager native-flatten, grad-ckpt,
`use_reentrant=False`), so **4000 steps ≈ 35–60 min per arm**; a six-arm sweep is
~4–6 GPU-hours. That converts RQ3-at-the-sweet-spot from "deferred" to "run it."

## Phase-2 sweep design

### Arms (each 4000 steps, seed 0, same data recipe as Phase-0.5)

| arm | K | insertion | attention surface | what it isolates |
|---|---|---|---|---|
| **L4k** | 0 | — | QKV-LoRA r8, all blocks | **drift control at matched budget** — every comparison is vs this, not vs base |
| **B4k** | 16 | entry | QKV-LoRA r8, all blocks | pure steps axis (10× the Phase-0.5 arm B) |
| **K36-b8** | 36 | block 8 | QKV-LoRA r8, blocks ≥8 | DSR sweet spot at LoRA reachability |
| **K36-b8-r32** | 36 | block 8 | QKV-LoRA r32, blocks ≥8 | rank/reachability axis |
| **K36-b8-QKV** | 36 | block 8 | **unfrozen qkv_proj**, mid blocks (~8–18) | max reachability — the arm the negative most likely mispriced |
| **K36-b8-aux** *(optional)* | 36 | block 8 | QKV-LoRA r8 | assisted relocation: mid-block patch-outlier-norm penalty (see below) |

Notes:
- **Register lr decoupled from LoRA lr.** Registers are fresh embeddings competing
  with a ~20× attractor; give them their own (higher) lr group — e.g. registers
  1e-2, LoRA 1e-4~1e-3. lr 1e-3 was fine for 400 steps; at 4000 steps watch for
  drift/divergence (checkpoint + montage every ~500 steps; keep the best by gate,
  not the last — cf. the turbo non-monotonic-checkpoint lesson).
- **K36-b8-QKV sizing:** qkv_proj is ~12.6M params/block; mid-only (11 blocks)
  ≈ 139M trainable → fp32 Adam states ~1.7GB. Fits alongside the frozen bf16 DiT
  with grad-ckpt. Do NOT unfreeze all 28 blocks in this pass.
- **The aux arm changes the question.** An explicit relocation pressure (penalize
  top-0.2% patch ‖x‖ at mid blocks, never penalize register norm) abandons
  "adoption for free" — it asks instead "does *forced* relocation preserve
  framing/quality?" Run it only if the unforced arms fail the adoption gate but
  the QKV arm shows partial movement; report it as forced, not as adoption.

### Metrics and gates (in order; each gates the next)

1. **Adoption gate (primary, metric-free, training-time + eval).** The relocation
   crossover: register max-‖x‖/median rising toward sink magnitude while
   patch top-0.2%/median falls. Log both ratios in the train history (currently
   only mean norms are logged — add `patch_sink_ratio`/`reg_ratio` per log step),
   and probe at **multiple depths** at eval (blocks ~8/14/20, not just 14 —
   insertion at 8 changes where a register-sink could form).
   **Pass:** any arm reaches reg_ratio ≥ ~5× median with patch sink ratio down
   ≥ 30% vs **L4k** (not vs base). **Fail all arms ⇒ the frozen-base register
   line closes for real** — "headroom belongs in the base" confirmed at the DSR
   sweet spot and 10× budget; that is the publishable negative and the line ends.
2. **Framing/coherence montage (mandatory guard, by eye).** Matched-seed grid
   (repro + control + the register_eval prompt trio, ≥3 seeds) for every arm vs
   L4k. Any systematic framing collapse ⇒ that arm is lesioned regardless of its
   ratios (the arm-A lesson). This is the artifact the pixel metrics can't replace.
3. **Benefit proxies (only for arms passing 1+2).** The RQ2 proxies vs L4k:
   (a) clamp-sensitivity (`sink_intervention` on the trained arm — relocation
   predicts a *register-specific* drop beyond L4k's drift level); (b) ex-sink
   per-patch Laplacian recovery (secondary — known noisy under re-planning);
   (c) **CMMD** distribution-level tiebreak (the existing paired PE-Core MMD²
   infra) + blind eyeball at real 28-step / CFG-4 settings.
   **Pass:** a register-specific sensitivity drop and no regression on the
   montage/CMMD. "Relocated but nothing improves" is itself a clean result:
   *the sink is adoptable but not quality-limiting on Anima* — distinguishing
   Anima from DSR's ImageNet DiTs.

### Discipline

- Every comparison is against **L4k** at matched steps; base is only the sink
  reference. Never against arm A (lesion) or the 400-step arms (budget mismatch).
- Fixed prompt set across all evals (sink magnitude is prompt-dependent).
- Generation is seed-deterministic — reuse prior images where the condition is
  unchanged; regenerate only new arms (`register_rq2_proxies.py --only`).
- Keep the sink probes' bit-exactness habit: eager, no compile, matched seeds.

## Implementation deltas (small, all in `bench/headroom/`)

1. **Insert-at-block-b.** Registers currently concat at `_run_blocks` entry.
   For b>0: concat (+ rope-row extension) via a `forward_pre_hook(with_kwargs)`
   on block b, keep the existing strip-at-return. Blocks <b run at seq, ≥b at
   seq+K — eager doesn't care; nothing AR-snaps.
2. **Unfrozen-QKV arm.** Adapter variant whose trainable surface is the target
   blocks' `qkv_proj` weights (fp32 master copies, restore-on-remove), no LoRA.
3. **Dual lr groups** (register vs attention surface) in `train_registers.py`.
4. **Richer train history**: log `patch_sink_ratio` / `reg_ratio` (already
   computed in the adapter) per log step, so the crossover is visible mid-run.
5. **Checkpoint every ~500 steps** + a tiny fixed-seed montage render per
   checkpoint (rank by gate, not by step count).
6. *(aux arm only)* top-0.2% patch-norm penalty at mid blocks, weight swept
   coarsely; never applied to registers.

K=0 (arm L) support already landed in `register_adapter.py`. The Phase-0.5
harness invariants carry over unchanged: eager native-flatten, rope-exempt
identity rows, strip before unpatchify, grad-ckpt `use_reentrant=False`
([[project_unsloth_reentrant_drops_grad]]).

## Risks / priors (be honest)

- **Adoption may need pretraining regardless.** The DINOv2/DSR caveat survives
  every budget increase: the sink attractor was carved by the full pretraining
  run, and 4k adapter steps may still lose. That outcome is the *point* of the
  sweep — it converts "not at this cheap config" into "not on a frozen base,
  period," with the sweet-spot config actually tested.
- **No per-image quality reward exists** (the Null-TTA wall). The benefit gate
  leans on register-specific *sensitivity* deltas, the montage, and CMMD — all
  imperfect. A quality claim stronger than "no regression + relocation happened"
  is not available and should not be made.
- **Longer schedules can drift.** 4000 steps of lr-1e-3 LoRA on 500 images can
  overfit/deform the model independent of registers; L4k absorbs this in every
  comparison, and per-checkpoint montages catch it early.
- **The aux arm can degenerate into norm-whack-a-mole.** Suppressing the outlier
  without giving the computation somewhere to go is exactly the clamp experiment
  (whole-image re-plans, no win). If the aux arm's registers don't light up while
  patch norms fall, stop it.
- **Per-σ reweight graveyard still applies** ([[project_sigma_reshape_no_win]]):
  if an arm only "works" via a σ-gated register gain at inference, it inherits
  that no-win result.

## Kill / success criteria

- **Kill (line closes):** no arm — including unfrozen-QKV at K36/b8/4k — passes
  the adoption gate. Write the negative into `docs/findings/`, keep the harness
  (it's a general non-decoded-token bench rig), stop proposing frozen-base
  registers on Anima.
- **Success (Phase 3 exists):** an arm passes adoption + framing + shows a
  register-specific sensitivity delta. Phase 3 = longer/bigger training of that
  config, CMMD-powered A/B, and a decision on whether the mechanism merits a
  base-model change (where DSR says it ultimately belongs).

## Related research

**Taming Outlier Tokens in Diffusion Transformers** (Wu et al., arXiv:2605.05206
— Dual-Stage Registers): the DiT-generator prior art. Confirms outlier sinks in
DiT generators, register benefit (FID 16.05→14.47 VAE-SiT, ~4× faster
convergence), the **~36-registers-at-block-8** non-monotonic sweet spot, and that
masking the outlier doesn't help (symptom, not cause). What DSR does *not* cover
— and where this sweep still adds signal — is the **frozen-pretrained-base
retrofit regime**: every DSR register is trained jointly from scratch. Their
"strengthens at low noise, intermediate layers" σ-profile matched our probe
mid-layer finding; our sink is strongest at high σ — sweep evals should keep the
σ-resolved readout rather than assuming either profile.

*Vision Transformers Need Registers* (Darcet et al.): the original emergent-sink
/ register mechanism on ViT encoders.

## References

- `bench/headroom/README.md` — Phase-0/0.5/proxy results this rewrite compresses;
  scripts: `sink_probe.py`, `border_toggle.py`, `sink_intervention.py`,
  `register_adapter.py`, `train_registers.py`, `register_eval.py`,
  `register_rq2_proxies.py`.
- Runs: `results/20260701-*` (RQ1/RQ3), `results/20260702-0659-rq2_proxies/`,
  `results/20260702-0715-rq3_armL/` + `20260702-0725-rq2_armL_only/` (arm L).
- Memory: [[project_headroom_registers_rq1_falsified]],
  [[project_headroom_registers_rq3_negative]].
- CLAUDE.md § "Text encoder padding" (padding-as-attention-sink, same mechanism
  on the text axis) — kept as motivating context.
- Original border-hypothesis design + RQ1/RQ2/RQ3 phase gates: git history of
  this file (pre-2026-07-02).
