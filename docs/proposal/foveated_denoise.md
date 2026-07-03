# Deferred-foveated merge — SHIP proposal

> **This replaced the research proposal on 2026-07-03.** The research arc (P0→P3) is
> COMPLETE: full digest in `bench/foveated/report.md`, per-phase detail and kill
> criteria in `bench/foveated/plan.md`, the original research proposal in
> `_archive/proposals/foveated_denoise_research.md`. This doc is the plan for wiring
> the one surviving deliverable into production.

## What ships

**Deferred-foveated merge** — training-free inference acceleration, image-identity
preserving (same image where it matters; periphery = intentional soft blur):

- Above σ_c: baseline full-grid sampling, unchanged. Composition/identity are decided
  identically to no-foveation.
- During the pre-crossing steps, accumulate two free signals: **cfgdelta**
  (per-cell |v_cond − v_uncond|, prompt-aware) and **x0var** (x̂₀ Laplacian energy at
  the crossing).
- At the σ_c crossing, build the **combo** mask: normalized signal sum → threshold →
  morphological open → close → dilate-1 → re-solve threshold so the final fraction
  hits target. Compact, subject-following, per (prompt, seed).
- Below σ_c: run the DiT block stack on the reduced sequence — fovea tokens 1:1,
  each 2×2-token periphery group averaged to one token (4096→~2100 at frac 0.35);
  merged rope = renormalized elementwise mean of the members' (cos, sin) rows (exact
  mean-position rope for symmetric groups); broadcast before `final_layer`.
- Final readout: periphery read from the merged representation (avg-pool → bicubic
  up), fovea untouched.

**Measured** (bench, bare DiT, eager, 1024², 28 steps, CFG 4): fwd ×2.15, **e2e
×1.37**, fovea visually baseline-identical, subject RMSE ties-or-beats a static
center rect on every prompt tested (−29 % on 3-subject hard prompts).

## Knobs and defaults

| knob | default | notes |
|---|---|---|
| `fovea_sigma_c` | 0.75 | standalone knee (P0/P1b). 0 disables. |
| `fovea_frac` | 0.35 | knob floor **0.25** (3s ladder knee); ≤0.15 falsified on multi-subject. |
| mask source | `combo` | needs CFG for cfgdelta; CFG=1 fallback → x0var-only, or static center rect. |
| periphery pool | 2×2 tokens, fixed | `FoveatedTokenMerge(merge_edge=4)` exists (exact-rope carries) but is unbenched — speed escalation only, not a launch knob. |

## Wiring plan

> **Status 2026-07-03: items 1–5 SHIPPED** — `networks/foveated.py` (runner +
> mask pipeline), `--fovea_sigma_c` / `--fovea_frac` / `--fovea_mask_source`
> on `inference.py`, `tests/test_foveated_merge.py`, bench probes re-import
> the promoted home. Item 6 (ComfyUI) is the remaining work — plan below.

1. ~~**Promote the mechanism out of the bench**~~ DONE: `FoveatedTokenMerge` +
   the mask pipeline (`score_to_cells`, morphology, `rect_cells`,
   `cells_to_mask4`) live in `networks/foveated.py`; `foveated_denoise`
   self-registers with `library.inference.generation` (SPD pattern). One
   refinement over the plan: the merged forward runs through the model's own
   `forward` via a `token_merger` kwarg on `forward_mini_train_dit`
   (merge/rope-swap at the `_run_blocks` boundary, riding the fake-5D
   native-flatten layout) — no duplicated external forward path, so the
   "custom-runner ≡ `anima()`" invariant is structural. The bench imports the
   promoted home (bespoke-loop mirroring lesson).
2. ~~**CLI surface**~~ DONE: `--fovea_sigma_c` (0 = off) / `--fovea_frac` /
   `--fovea_mask_source {combo,cfgdelta,x0var,rect}` in
   `library/inference/args.py`; dispatched from `generate_body` on
   `fovea_sigma_c > 0`, mutually exclusive with `--spectrum`/`--spd`.
   `GenerationRequest` reaches them via `extra_argv` (routes through
   `inference.parse_args`).
3. **Compile interaction** — scoped to eager for v1 as planned: a compiled
   model (`_native_flatten`) gets a one-time warning (one extra token count
   per mask, outside the tier's `dynamic_seq` band → expect a recompile per
   generation). Validation before default-on with `torch_compile` is still
   open.
4. ~~**Plug-in posture**~~ DONE: DCW / SMC-CFG / FSG / CFG++ warn-and-ignore
   (SPD posture); stochastic samplers fall back to Euler with a warning.
   Spectrum composition stays closed — mutual exclusion raises at dispatch
   with a "P3 closed, don't re-propose" message.
5. ~~**Tier 1.5 obligations**~~ DONE: `tests/test_foveated_merge.py` — all-fovea
   mask ≡ identity (bit-exact through the shipped `token_merger` path),
   group-shared periphery velocity, exact mean-position rope, mask-fraction +
   compactness contracts, CPU end-to-end runner smoke. Bench regression =
   re-run `probe_mask_sources` / `probe_fraction_stretch` (they now import the
   shipped code, so the p2b/p3s numbers certify the production pipeline).
6. **ComfyUI node** — see the integration plan below.

## Invariants for the implementer (hard-won, do not rediscover)

- **Never rewrite the latent.** The latent stays full-res end-to-end; merging is
  what the *compute* sees. Group-constant renormalized noise is off-manifold garbage
  even with correct per-pixel variance (P0 run 1).
- Mask lives on the pooled cell grid — no group may straddle the fovea boundary;
  every 2×2-token cell uniformly fovea or periphery.
- Masks must be **compact blobs** (morph open/close/dilate) — scattered fovea cells
  lose fidelity because their whole attention neighborhood is merged (P2, +47 %).
- The **final bicubic readout is part of the quality contract** — skipping it leaves
  never-denoised HF detail in the periphery at decode; nearest-neighbor reads as
  mosaic.
- 5D/4D discipline: pooling ops are 4D; `squeeze(2)`/`unsqueeze(2)` explicitly
  (CLAUDE.md dim-2 invariant).
- σ_c is a knob but **0.75 is load-bearing**: foveating inside the authority window
  makes a *different image* (that's the "deferred" in the name).
- Verdicts are eyeball-first on full-res montages; RMSE certifies change, not
  improvement — and calibrate on **hard prompts** (multi-subject, texture-heavy),
  not the centered default; the composed line survived four phases on a collapsed
  baseline because nobody rendered channel6 (P3 lesson).

## Explicit non-goals (closed by the bench — don't re-propose)

- **Composed foveated-Spectrum** (spatial refresh allocation): no headroom at sane
  schedules, baseline collapse at aggressive ones (P3).
- **Partial recompute** (fovea queries vs cached K/V): quality ceiling ties plain
  Spectrum at both operating points. If ever revisited: catch-up-every-2 re-anchor
  is mandatory (frozen-periphery fits degrade at sane cadence; catchup-2 restores
  exact neutrality).
- **Fovea-region SEA triggers / asymmetric cadence**: trigger is region-independent
  (P2t); the timing win ships as `schedule="sea"`.
- **True grid coarsening** (paper mechanism, post-trained adapter): 1b already
  delivers the token reduction training-free.

## ComfyUI integration (item 6 — the remaining wiring)

Ship as an option in the **Spectrum-KSampler repo**
(`ComfyUI-Spectrum-KSampler`, which already hosts the SPEED and FSG ports —
same "sampler-level runner" family), not a standalone node: the mechanism
needs a custom sampling loop anyway, and that repo owns the KSampler-compat
plumbing. Vendor the promoted math verbatim from `networks/foveated.py`
(`FoveatedTokenMerge`, `score_to_cells`, morphology, readout) — the module is
numpy/torch-only with no repo-internal imports beyond the runner half, so the
mechanism classes vendor cleanly; keep edits upstream-first and re-vendor
(never fork the math — the `spd_core.py` precedent).

**How the pieces map onto comfy:**

- **The merge is a runner, not a weight patch.** Comfy's cosmos backbone uses
  split q/k/v while our training DiT fuses them — irrelevant here: the merge
  sits at the block-stack boundary (tokens + rope), touching no projection
  weights. It ports as a `model_options` wrapper the same way the EasyControl
  KSampler node patches the block loop.
- **Forward hook point**: wrap the cosmos model's block loop (the equivalent
  of our `_run_blocks` boundary — after patch-embed + rope build, before
  `final_layer`): merge tokens, swap in `merge_rope`'s reduced (cos, sin),
  run blocks on the reduced sequence, broadcast back. On comfy's predict2
  cosmos the rope tensors are per-block arguments — reduce once per mask and
  cache, exactly as `merge_rope` does.
- **Signal accumulation**: cfgdelta needs both CFG branches. In comfy the
  clean tap is a `sampler_post_cfg_function` (it receives `cond_denoised` /
  `uncond_denoised` per step) — accumulate the per-cell |Δ| there; x̂₀
  Laplacian comes from the same callback's denoised output at the crossing.
  No extra forwards, mirroring the CLI runner.
- **σ_c crossing + mask build**: the custom sampler owns the σ schedule
  (comfy sigmas are the flow σ directly for cosmos), so the crossing check,
  `score_to_cells`, and merger construction happen inside the sampler
  function; below σ_c it sets the block-loop wrapper into the model options
  for the remaining steps.
- **Final readout** (part of the quality contract): apply the avg-pool →
  bicubic periphery blend to the final latent inside the sampler, before the
  latent returns for VAE decode.
- **Knobs**: same three — `sigma_c` (default 0.75), `fovea_frac` (0.35,
  UI-min 0.25), `mask_source` (combo/cfgdelta/x0var/rect). CFG=1 workflows
  auto-fall back to x0var, as in the CLI.

**Gotchas carried from the node fleet** (don't rediscover):

- **AnimaBlockCompile rebuilds the DiT and strands hooks**
  ([[project_blockcompile_rebuilds_dit_strands_hooks]]): bind the block-loop
  wrapper at *sample time* against `executor.class_obj.diffusion_model`, not
  at node-construction time, or a downstream BlockCompile node silently drops
  the foveation.
- **Euler-only**, like the CLI runner: the node exposes its own sampler entry
  (or validates the selected one), not a compose with `er_sde_cns` etc.
- **Mutually exclusive with the Spectrum caching path** in the same repo —
  the P3 closure applies identically; the node UI should make them an
  either/or choice, not stackable toggles.
- Register-token checkpoints (kept-live `num_registers` LoRAs) extend the
  sequence and rope at inference — merging + register injection is
  unvalidated; refuse or warn when both are active.

**Definition of done**: parity render against the CLI (`--fovea_sigma_c 0.75`)
at matched seed/schedule — fovea pixel-close, same mask (dump `cells` from
both paths), e2e speedup within noise of ×1.37 at 1024²/28-step/CFG 4; the
usual hard-prompt eyeball (channel6-class multi-subject, not just the centered
default — the P3 lesson).
