# AdaLN LoRA — training → ComfyUI-compatible shipping

Status: **extraction path shipped & verified; training path plumbed but unbenched.**
Any train-side default change is bench-gated (Tier 1.5/2 per `../../CONTRIBUTING.md`) — this
doc is the map, not the verdict.

## What adaln is in this architecture

Anima inherits Cosmos-Predict2's per-block modulation: each `Block` has three
branch-modulation MLPs (`adaln_modulation_{self_attn,cross_attn,mlp}`) producing
shift/scale/gate from the **timestep embedding only** — no text, no spatial
conditioning. Every Anima checkpoint uses the **AdaLN-LoRA bottleneck form**
(`use_adaln_lora=True, adaln_lora_dim=256`):

```
SiLU → Linear 2048→256 (".1", down) → Linear 256→6144 (".2", up)
```

The name "AdaLN-LoRA" is NVIDIA's — it's a *pretraining architecture* choice
(low-rank bottleneck on the modulation MLPs), not an adapter. The param math is
why: full-rank adaln would be 2048×6144 ×3 branches ×28 blocks ≈ **1.06B params**;
the 256-bottleneck is ≈176M. It is not optional per checkpoint — the weights
(`blocks.{b}.adaln_modulation_{br}.1/.2.weight`) exist in base and turbo alike.

### The `use_adaln_lora=False` "mystery" in diffusion-pipe

Not a decision by the Anima author. `models/cosmos_predict2_modeling.py` is
vendored NVIDIA reference code (SPDX NVIDIA header) and `False` is just the
upstream class-signature default preserving the vanilla-DiT form. diffusion-pipe
**hardcodes** `use_adaln_lora=True, adaln_lora_dim=256` for every checkpoint it
loads (`models/cosmos_predict2.py::get_dit_config`, ~line 125). ComfyUI's
detection does the same (`comfy/model_detection.py:801`).

## Who trains adaln today — we are the outlier

| Trainer | adaln in LoRA target set? | Evidence |
|---|---|---|
| **diffusion-pipe** (Anima author's trainer) | **YES** — targets *every* `nn.Linear` under `Block`/`TransformerBlock`, no exclusions and no config filter | `models/base.py::configure_adapter` (~line 219) |
| **sd-scripts** (official kohya, upstream Anima support) | **NO by default** — appends `.*(_modulation\|_norm\|_embedder\|final_layer).*`; opt-in via `include_patterns` (include beats exclude, same mechanism as ours) | `sd-scripts/networks/lora_anima.py:254` |
| **this repo** | **NO** — `_DEFAULT_EXCLUDE` blocks `adaln_up_` | `networks/lora_anima/config.py:168` |

Our exclusion is **inherited verbatim from kohya's `lora_anima.py`** (ours extends
the same regex with the runtime rename names) — a conservative
"attention+MLP only" convention bundled with norms/embedders, specific to the
Anima implementation (kohya's Flux LoRA has no such exclude machinery at all).
Notably kohya also ships `sd-scripts/networks/convert_anima_lora_to_comfy.py`, which
**already handles adaln module names** — so an sd-scripts user who opts in gets
a ComfyUI-shippable adaln LoRA today. That converter is prior art for our
missing trained-checkpoint export step.

diffusion-pipe also saves in "ComfyUI format" (`diffusion_model.`-prefixed PEFT
keys, `models/cosmos_predict2.py::save_adapter`), so **diffusion-pipe-trained
Anima LoRAs all carry adaln and ComfyUI already applies it**; sd-scripts-trained
ones default to no adaln, like ours. Two sharpenings on the diffusion-pipe side:

- Ecosystem LoRAs carry **both** `.1` and `.2` adaln LoRAs (both are Linears
  inside `Block`), plus the **LLM adapter** (`TransformerBlock` is in the target
  list) — strictly more surface than our target set at equal rank. Keep this in
  mind for any expressivity comparison against third-party LoRAs.
- **Cross-tool render mismatch**: loading an ecosystem LoRA in-repo via
  `--lora_weight` drops its adaln + LLM-adapter keys (different naming, excluded
  targets) — the same file renders differently here vs in ComfyUI. First thing
  to check when a "repo inference ≠ Comfy" report involves a third-party LoRA.

## ComfyUI support (verified 2026-07-15)

ComfyUI needs no special-casing: `comfy/lora.py::model_lora_keys_unet` (generic
branch, ~line 191) maps **every weight in the model's state dict** to a
`lora_unet_<path-with-underscores>` key. Base Anima and turbo are the same model
class, so the same keys apply to both. Verified end-to-end by feeding our
adaln-inclusive extraction through ComfyUI's own `comfy.lora.load_lora` against
a key map built from the real checkpoint: **all keys consumed — 364 patch
targets = 280 attn/MLP + 84 adaln**, aimed at
`diffusion_model.blocks.{b}.adaln_modulation_{br}.2.weight`.

## Key-naming contract (the one real gotcha)

The same Linear has two names, and ComfyUI only knows one of them:

| Surface | Module path | LoRA key |
|---|---|---|
| Checkpoint / ComfyUI | `blocks.{b}.adaln_modulation_{br}.2` | `lora_unet_blocks_{b}_adaln_modulation_{br}_2` |
| This repo, runtime (post `_dit_rename_hook`, `library/anima/weights.py`) | `blocks.{b}.adaln_up_{br}` | `lora_unet_blocks_{b}_adaln_up_{br}` |

Runtime-named adaln keys are **silently dropped** by ComfyUI. Anything shipped
must use the comfy layout; anything consumed in-repo (warm-start, `--lora_weight`)
wants the runtime layout.

## Path 1 — extraction (SHIPPED)

`scripts/extract_delta_lora.py` extracts adaln up-projections from a full-model
delta with `--include_adaln`, and `--adaln_layout comfy` renames to the ComfyUI
layout at save time (default `runtime` keeps warm-start compat; layout recorded
in `ss_delta_extract` metadata).

```bash
python scripts/extract_delta_lora.py \
    --tuned models/diffusion_models/anima_turboV10.net.safetensors \
    --rank 96 --act_scales models/extracted/act_scales_base_4step.safetensors \
    --include_adaln --adaln_layout comfy \
    --out models/extracted/anima_turboV10_delta_r96_asvd_adaln.safetensors
```

Shipped artifact: `models/extracted/anima_turboV10_delta_r96_asvd_adaln.safetensors`.

Delta facts (official turbo vs base) that motivated this:

- `.2` up-projs are the **largest movers** in the whole delta (rel-Δ 0.008–0.03);
  `.1` down-projs barely moved (0.0002–0.006) — extractor deliberately takes `.2` only.
- `.2` in-dim is 256, so r=96 captures ~99% energy (recon rel-err ≈ 0.10) —
  adaln is essentially lossless at this rank; residual infidelity lives in attn/MLP.
- Adding adaln lifted mean captured energy **0.803 → 0.860** vs the adaln-less
  r96 ASVD file (min stays 0.423, a non-adaln module).

## Path 2 — training (PLUMBED, NOT BENCHED)

No new machinery needed to *train* adaln: `include_patterns` beats the default
exclude (`networks/lora_anima/network.py` ~line 245: `if excluded and not
included: continue`), and runtime `adaln_up_{br}` is a plain `nn.Linear`.

```toml
# method TOML
include_patterns = [".*adaln_up_.*"]
```

Turbo knobs (2026-07-15): `network.train_adaln` wires the include on student+fake;
`network.fake_adaln = false` builds an adaln-less critic (VRAM lever);
`network.adaln_rank = N` builds the adaln modules at their own rank via the
factory's `network_reg_dims` regex→dim override — measured on the r96 extract,
the adaln ΔW keeps 0.980/0.991/0.995 of its energy at r32/48/64 (in-dim is 256),
so sub-attn rank there is near-free. The reg_dims path keeps the NETWORK alpha
(hotter alpha/rank scale on adaln by rank/adaln_rank; warm start folds scale so
init is exact). The extractor mirrors it as `--adaln_rank`. Guards in
`tests/test_turbo_adaln.py`.

**Compile interaction (fixed 2026-07-15)**: training adaln under
`compile_dynamic_seq` initially crashed at the first grad-bearing forward with a
`ConstraintViolationError` on the marked seq range. The adaln LoRA makes
shift/scale/gate require grad, so the backward gains a seq-axis reduction whose
inductor mix-order-reduction fusion records a `Ge(seq, 4096)` guard that
FxGraphCache replays into the ShapeEnv — contradicting the strict `mark_dynamic`
bound. Fixed by disabling `triton.mix_order_reduction` whenever dynamic-seq
marks are active — full mechanism in
[`../optimizations/for_compile.md`](../optimizations/for_compile.md) §2.6 and
[[project_inductor_mix_order_reduction_guard]]. Applies to ANY LoRA on a
broadcast-consumed Linear, not just adaln.

**Missing piece for shipping trained adaln**: the trainer saves runtime-layout
keys. A comfy-layout rename at export (same mapping as the extractor's
`--adaln_layout comfy` pass) is needed — either a small post-export utility or a
save-time option. Until then, trained adaln LoRAs work in-repo but not in ComfyUI.

### What adaln can and can't learn

t-conditioned **global** modulation only: per-σ tone/contrast/gating shifts.
No text or spatial pathway. Plausible fits:

- **Turbo distillation** — proven load-bearing: the σ→behavior remap is exactly
  what few-step distillation changes, and the official turbo moved adaln hardest
  ([[project_official_turbo_v10_eval]] already lists adaln targets as a lever).
- **Style LoRAs** — speculative: a home for *untagged global style* (grade,
  palette bias), given cross-attn only learns labeled tags
  ([[project_lora_crossattn_learns_labeled_only]]). Could equally be inert or a
  new coupling hazard. Bench decides.

### Interaction with modulation guidance (MOD=1)

The adaln modulation MLPs are a shared pipe with **three tenants**
(`../inference/mod-guidance.md`):

1. **Stock**: t-embedding → per-σ shift/scale/gate.
2. **Mod-guidance**: adds `schedule[ℓ]·δ` (pooled-text steering delta) to the
   t-embedding *upstream of each block's modulation MLPs* — the same MLPs
   transduce the delta into coefficient steering. It also installs
   `pooled_text_proj`, making the t-embedding itself pooled-text-aware.
3. **AdaLN LoRA** (this doc): changes the MLP weights themselves.

Consequences:

- **An adaln LoRA reshapes mod-guidance's steering.** Guided coefficients become
  `(W+ΔW)(t_emb + w·δ) = stock + ΔW·t_emb + w·ΔW·δ` — the `ΔW·δ` term re-rotates
  the calibrated steering direction. Mod-guidance schedules and the external
  Spectrum node's scalar defaults were tuned with stock `W`; "MOD=1 composes
  with any checkpoint" silently becomes "…any checkpoint that doesn't touch
  adaln." Any bench arm that ships adaln to users must include a MOD=1 A/B.
  (Turbo inference runs CFG 1.0 / 4-step without mod-guidance, so the turbo
  student arm is unaffected.)
- **The "adaln is text-blind" claim is stock-model-only.** On a mod-distilled
  model (`pooled_text_proj` installed and loaded — note the gotcha in
  [[project_sea_delta_generalizes_guidance]]: it loads in `load_dit_model`),
  the t-embedding carries max-pooled text, so an adaln LoRA *trained on such a
  model* would learn a weakly pooled-text-conditioned modulation response
  (global tags, not token-level). Untested hypothesis; changes what bench arm 2
  could measure.
- **Independent evidence convergence**: mod-guidance's effectiveness (ICLR'26
  result) and the official turbo's adaln movement both say the modulation
  pathway is the high-leverage lever for *global* behavior — and neither says
  it carries content. Consistent with keeping expectations modest for style
  LoRAs.

#### Compatibility design options (escalate only on evidence)

Magnitude bound first: the steering distortion is `‖ΔW‖/‖W‖` per block — ≤1–3%
even for the turbo extract, smaller for trained style LoRAs. diffusion-pipe
LoRAs have been silently exercising this coupling in ComfyUI without visible
mod-guidance breakage, so treat this as a *measured* risk, not an assumed one.

0. **Do nothing + verify** (default): the MOD=1 A/B in the bench plan is the
   gate. If drift is invisible at rendered level, stop here.
1. **Ship adaln as a separate file** — worth doing regardless of the coupling:
   the extractor's module specs already make an adaln-only emit trivial, ComfyUI
   chains LoRA loaders natively, and it buys a free per-part strength knob,
   free A/Bs, and a zero-cost opt-out for MOD users. Downside is two-file UX.
2. **Magnitude renorm** (cheap patch): rescale the baked δ per block so the
   transduced steering norm matches the stock-`W` calibration. Fixes magnitude,
   not rotation — but rotation is the ≤3% part.
3. **Frozen steering path** (structural fix): compute the guidance transduction
   through *stock* modulation weights — bake per-block coefficient-space deltas
   `δc(ℓ,br) = M⁰(ē+δ) − M⁰(ē)` offline from the base checkpoint (σ-flat, same
   approximation the current bake already accepts) and add them *post*-MLP.
   Guidance becomes checkpoint-independent by construction; distributable like
   the CNS γ auto-download. Only worth building if option 0 shows real drift —
   note ComfyUI merges LoRA into weights at patch time, so stock `W` must come
   from an offline bake, not the live model.
4. **Per-LoRA recalibration**: retrain/re-bake mod-guidance with the LoRA
   merged. Correct but doesn't scale to an ecosystem; not recommended.

### Existing checkpoints

Every LoRA trained in this repo to date has **zero adaln keys** (verified on
atlas / colorize / soup checkpoints) — the pathway was never trained, nothing is
mis-shipped, and there is nothing to retrofit. Trained models learned to
compensate through attn/MLP; see caveat below.

## Bench plan (gate before any default change)

1. **Turbo student arm** (highest expected value): student targets `adaln_up_`
   via include_patterns, warm-started from a *runtime-layout* adaln-inclusive
   extraction; vs current student. Rank by **rendered 4-step grid**
   ([[project_turbo_lr_instability_threshold]] — never by fm_mse), CMMD
   within-run only ([[project_cmmd_val_signal]]).
2. **Style-LoRA arm**: same artist dataset ± adaln targeting, rendered A/B.
3. **Zero-training stacking probe** (cheapest first signal): in ComfyUI, chain
   our published turbo student + an adaln-only comfy-layout extract of the
   official delta. Caveat: the student trained *without* the adaln shift and
   compensated around it — stacking is off-distribution; a win here is a strong
   green light, a loss is not conclusive against arm 1.

## Checklist

- [x] Extractor: `--include_adaln` + `--adaln_layout comfy` (rename at save)
- [x] Shipped comfy-layout extraction artifact (r96 ASVD + adaln)
- [x] Verified against ComfyUI's own `load_lora` (all 364 targets matched)
- [ ] Comfy-layout export step for *trained* checkpoints
- [ ] Bench arm 3 (stacking probe — no training)
- [ ] Bench arm 1 (turbo student + adaln, warm-started)
- [ ] Bench arm 2 (style LoRA ± adaln)
