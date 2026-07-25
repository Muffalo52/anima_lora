# directedit_ec — implementation

EasyControl's extended self-attention as a **learned, gated preservation
prior** on the DirectEdit trajectory — the trained generalization of
V-injection. Everything below is shipped and zero-training; the winning
recipe needs only the stock inpaint adapter.

## The recipe (Phase 1a/1b winner)

```bash
python scripts/edit.py <image> --edit "<caption + edit>" \
  --easycontrol_weight output/ckpt/methods/anima_inpaint.safetensors \
  --easycontrol_mask mask.png \
  --mask mask.png            # SAME file for both flags
# b_offset 0 — no per-image tuning
```

The hole must be punched in **both** preservation mechanisms:

1. **EC cond hole** (`--easycontrol_mask`): gray-fills the masked region of
   the cond *image* pre-VAE (`scripts/edit.py:887–907`) — matching the
   inpaint adapter's training distribution (never zero the latent). Outside
   the hole the prior clamps (trained behavior); inside it generates freely.
2. **Δz anchor mask** (`--mask`): drops the anchor residual inside the edit
   region — paper Eq. 12's anchor-side half, wired in
   `library/inference/editing/directedit.py::edit_forward` (`mask` param,
   ~line 340; applied at ~line 427). Without it the global anchor pulls the
   hole content back to the source (Phase-1a's 1/3 failure mode).

## Component map

| Piece | Where | Notes |
|---|---|---|
| Δz-anchored inversion + edit pass | `library/inference/editing/directedit.py` (`invert`, `edit_forward`) | anchor = inversion residuals; exact recon at ψ_tar==ψ_src |
| Eq. 12 anchor-side mask | `edit_forward(mask=…)` | latent-resolution, 1 = edit region; logs % of latent dropped |
| EC cond KV cache + gate | `networks/methods/easycontrol.py::_target_only_with_cached_cond_kv` (~1208) | LSE-extended attention; cond K/V prefilled once |
| `b_cond` offset dial | same function — **live logit bias**, NOT baked into the KV cache | additive offset post-`load_weights`; each −1 ≈ e× less cond mass; inpaint useful range −1..−2 (superseded by the mask recipe at offset 0). **Band location is per-checkpoint** — the subject descriptor engages at +6..+8, so always sweep wide on a new adapter |
| CLI wiring | `scripts/edit.py` `--easycontrol_weight / --easycontrol_b_offset / --easycontrol_mask / --mask`; `inference.py --easycontrol_b_offset` | `--easycontrol_mask` requires `--easycontrol_weight` |
| Bench harness | `project/directedit_ec/bench/run_bench.py` | phases: smoke / 0 / 0b / 1a / 1b / 2; `EDITS_1B` dict = per-image edit specs + hole boxes |
| Retrieval probe | `project/directedit_ec/bench/run_subject_probe.py` | DirectEdit-free: cond = image A, prompt = caption of image B, vs a no-EC control — the only arm that isolates the adapter from the composition |
| Phase-2 subject descriptor | `configs/easycontrol/subject.toml` + `easycontrol_adapters/tools/subject_pairs.py` | cross-image same-character pairs off `caption_index.json`; staging + cond tree are pure symlinks over the shared LoRA cache (no encode); `EASYADAPTER=subject` |

## Mechanism invariants (learned the hard way)

- **EC and V-injection cannot stack.** The EC-patched `Block.forward` routes
  attention through `_extended_target_attention`, bypassing the
  `Attention.forward` that `_v_injection_scope` patches. Replace, not compose.
- **`cond_scale` is near-binary on the inpaint prior** (0.25/0.5 ≈ no-EC,
  1.0 = total clamp) — scaled-down cond-LoRA deltas move cond K/V off the
  distribution the learned gate retrieves. The dial is `b_cond`, and with the
  mask recipe you don't need the dial at all.
- **`b_cond` does not train — treat `b_cond_init` as a fixed hyperparameter.**
  Both shipped checkpoints saved it at exactly their init (inpaint −6, subject
  −8, all 28 blocks). It is in the optimizer with an analytical gradient, so
  this is a near-stationary point, not broken wiring. Consequence: the init
  alone sets cond attention mass ≈ (S_c/S_t)·e^b for the entire run, and
  `cond_res_scale` multiplies into it — `-8` with `cond_res_scale=0.5` gives
  8.4e-5 (29.5× below inpaint's 2.5e-3), which is too closed for the cond path
  to receive useful gradient. Pick the init so the mass is workable *before*
  training; see `bench/report.md#phase-2`.
- **The aligned-cond copy path is architectural, not learned.** Extended
  self-attention over cond K/V reproduces the cond wherever it is spatially
  aligned with the target, and degrades the output where it is not — this
  happens at high `b_cond` even for an adapter whose cond stream barely
  trained. So "EC preserves the source" in a bench where cond == source is
  **not** evidence that the adapter learned anything; isolate with
  `run_subject_probe.py` (non-aligned cond) before crediting training.
- The cond KV cache must be active through **both** inversion and edit passes
  for the exact-recon composition with the anchor.
- Compile is disabled under EC in `edit.py` (matches the inference engine's
  eager EC path); compile-compat under EC is untested.
- Cost profile vs V-injection: one KV prefill instead of a parallel src
  forward per injected step; hyperparameter surface = one scalar (or zero
  with the mask recipe) vs `t_inj` steps × block set.

## Known limits

- Position-locked prior: geometry/pose edits degenerate (full-frame box ⇒
  unanchored generation — pose lands, nothing kept). Phase 2's target.
- Inpaint-prior style artifact: flat saturated regeneration inside the hole
  on simple flat-background images (all EC arms, mask-independent).
- Mask source is a manual box today; the cfgdelta subject localizer
  (foveation line's reusable artifact) is the planned automatic upgrade.
