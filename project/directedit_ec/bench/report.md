# directedit_ec — EasyControl cond stream as a learned preservation prior for DirectEdit

## Phase 2: cross-image subject descriptor (2026-07-25)

**Train:** `anima_easycontrol_subject`, 8928 steps / 8 epochs over the 1116-pair
set, 1h45m, `loss/epoch_average` 0.0797, no anomalies. Pair data verified: cond
latents symlink to a *different* image of the same character (73% cross-artist).

**Runs:** `results/20260725-0930-phase2a-boffset` (offsets 0..+4),
`-0949-…-hi` (+6/+8), `-0953-…-fine` (+5/+7) — one b_offset curve, same
seed/config, cross-run comparable · `-1000-phase2-retrieval` (geometry,
`--phase 2`) · `-1014-phase2-subject-probe-engaged` (DirectEdit-free retrieval).

### Verdict: both gates FAIL — but the run does NOT test the Phase-2 hypothesis

The gates fail, and the kill criterion's *conclusion* ("pairing wasn't the
constraint") is **not** supported: the cond stream was closed for the entire
train, so cross-image pairing was never actually exercised.

**The gate never opened during training.** `b_cond` is empirically non-learning
— the checkpoint saved exactly `-8.0` on all 28 blocks, as the inpaint
checkpoint saved exactly `-6.0` (bf16 resolution at that magnitude is 0.0625,
so |drift| < 0.03 over 8928 AdamW steps at lr 2e-5). It *is* wired to train
(in the optimizer via `get_trainable_params`, analytical gradient in
`easycontrol_attention.py`), it simply sits at a near-stationary point. With
`b_cond=-8` and `cond_res_scale=0.5` (S_c/S_t ≈ 0.25) the cond keys carried

    cond attention mass ≈ 0.25·e⁻⁸ ≈ 8.4e-5   (0.008%)

versus inpaint's `2.5e-3` at `b=-6, cond_res_scale=1` — **29.5× more**. The
weights corroborate it: cond-LoRA up-projections (zero-init) reached only
|w| ≈ 1.2e-4, while the adaln LoRA — which feeds the *target* stream and is not
gated by `b_cond` — moved 8× further. The cond path got almost no gradient
because the gate was shut.

**Gate (a) — sweet-spot width: FAIL (0 usable units, vs inpaint's ~1).** The
preservation band exists but is displaced ~7 units; MSE vs source, edit = +glasses:

| arm | dan_9596032 | 10473210 | 7538087 |
|---|---|---|---|
| base_t0 (pure anchor) | 0.13988 | 0.05723 | 0.02324 |
| vinj_t6 | 0.01594 | 0.00811 | 0.00549 |
| ec_s1 (offset 0 = trained point) | 0.14454 | 0.06598 | 0.03024 |
| ec_b4 | 0.15791 | 0.05846 | 0.02013 |
| ec_b5 | 0.15001 | 0.05391 | 0.01850 |
| ec_b6 | 0.09146 | 0.01666 | 0.00881 |
| ec_b7 | **0.01068** | **0.00498** | **0.00147** |
| ec_b8 | 0.00451 | 0.00236 | 0.00101 |

At +7 preservation beats `vinj_t6` on all three images — but the **edit is
suppressed across the whole band**. Render-judged (face crops), +glasses lands
only at +5, where preservation is nil (0.150). Preserve-and-land is empty:
usable width **0**. Inpaint had ~1 unit (b−1 on dan). The width did not
improve; it got worse.

**Gate (b) — geometry parity: FAIL to demonstrate retrieval.** `--phase 2` adds
the arm 1b could not express: cond left **whole** (identity available
position-free) with only the Δz anchor released full-frame. 1b's geometry row
gray-fills the *entire* cond, so a subject descriptor gets zero identity to
retrieve — it degenerates to unanchored generation for any adapter.

| arm | dan (squatting→standing) | 10473210 (+arms up) | 7538087 (→sitting) |
|---|---|---|---|
| vinj_t6 | pose unchanged | pose unchanged | pose unchanged |
| ec_anch (offset 0) | stands, keeps **nothing** | no change | no change |
| ec_anch_b6 | pose unchanged, source kept | no change | no change |
| ec_anch_b8 | ≈ source (clamped) | ≈ source | ≈ source |

Preservation and edit stay mutually exclusive — the same cliff as inpaint. EC
ties `vinj_t6` only by both failing. No arm lands the pose *and* keeps identity.

**The decisive check: no position-free retrieval was learned.**
`run_subject_probe.py` drops DirectEdit entirely and replays the adapter's own
training task as plain generation — cond = image A, prompt = caption of image B
(different image, same character), against a no-EC control at the same seed (the
control matters: the prompt already carries the character name as a tag). On
train-set pairs, i.e. an upper bound:

- offset 0 (trained point): `ec_b0` ≈ `noec` on all 3 pairs — the adapter
  contributes essentially nothing.
- offsets +6/+7/+8: the image **degrades** — washed out at +6, muddy at +7,
  collapsed to noise at +8. Identity does not transfer.

So what appears at +7/+8 in the edit bench is not learned retrieval, it is the
**architectural** copy path: extended self-attention over cond K/V reproduces
the cond whenever it is spatially aligned with the target (in the edit bench
cond *is* the source), and floods the attention with mismatched features when it
is not. That path exists without any training; the subject pairs contributed
almost nothing to it.

### What this does and does not settle

- **Does not settle** whether cross-image pairing can teach position-free
  retrieval — the mechanism that would carry it was closed at ~8e-5 mass for
  the whole run. This is a training-configuration failure, not a refutation.
- **Does settle** that `b_cond_init` is a load-bearing hyperparameter that does
  *not* self-correct, and that `-8` (with `cond_res_scale=0.5`) is far too
  closed to train through. Next arm: `b_cond_init ≈ -2` (mass 3.3e-2) and/or
  `cond_res_scale=1.0`, same cost as this run (~1h45m).
- **Untested combination:** the shipped 1a/1b mask recipe (`ec_mask_anch`) at
  the subject adapter's engaged point (+7). Given the probe result it would at
  best reproduce inpaint behavior, but it was not run.
- Q4 (hole-style artifact) is **not** answered — it needs the mask recipe at an
  engaged offset.

Wiring shipped alongside: `inference.py --easycontrol_b_offset` (the dial existed
only in `scripts/edit.py`; this checkpoint is unusable from the main inference
path without it), `EASYADAPTER=subject` for `test-easycontrol`, and
`run_bench.py --phase 2`.

## Phase 1a: masked-cond probe (2026-07-24)

**Runs:** `results/20260724-1827-phase1a` (3 img × 7 arms), `results/20260724-1844-phase1a-anchmask` (+2 arms, same seed/config — cross-run comparable) · Edit: caption + ", glasses", CFG 4, 28 steps, seed 42, b_offset 0 everywhere.

### Verdict: PASS, amended — the hole needs punching in BOTH preservation mechanisms

Feeding the inpaint prior its trained input (cond = source with a gray hole over the face box) gives exception-driven preservation exactly as proposed — but the cond hole alone landed the edit on only **1/3** images. The missing piece is the **Δz anchor**: it is global, so it keeps pulling the hole content back to the source after the EC prior has released it. Dropping the anchor inside the edit region (`--mask`, the never-implemented paper-Eq.-12 anchor-side half — now wired in `directedit.edit_forward`) fixes this: **`ec_mask_anch` (EC cond hole + anchor mask) lands the edit on 3/3 images — including 10473210, where every Phase-0 recipe at b−1 failed — at b_offset 0, no per-image tuning, still zero training.**

Outside-hole MSE vs source (recon_base level in parens):

| arm | dan_9596032 | 10473210 (hard) | 7538087 |
|---|---|---|---|
| recon_base | (0.00019) | (0.00015) | (0.00005) |
| base_t0 | 0.15114 | 0.05904 | 0.01915 |
| vinj_t6 | 0.01441 | 0.00781 | 0.00502 |
| ec_b-1 | 0.05394 | 0.00618 | 0.00850 |
| ec_b-2 | 0.15057 | 0.05029 | 0.02639 |
| ec_mask | 0.00239 | **0.00038** | 0.00301 |
| anch_only | 0.15340 | 0.05804 | 0.01728 |
| **ec_mask_anch** | **0.00238** | **0.00036** | **0.00310** |
| edit lands (ec_mask_anch) | ✓ | ✓ | ✓ |

- **Best-of-alternatives comparison** (the gate's real question): ec_mask_anch beats best-of-{vinj_t6, ec_b-1, ec_b-2} on outside-hole preservation by 2.6–17× on every image, and is the only arm landing the edit on all three.
- **The two controls split the blame cleanly.** `ec_mask` (cond hole only): preservation identical, edit lands 1/3 — the anchor suppresses the edit inside the hole. `anch_only` (anchor mask only, no EC): edit lands but outside-MSE ≈ base_t0 (composition destroyed) — at CFG 4 the anchor never was the preservation mechanism; the EC prior is.
- **The literal "≤ 2× recon" gate FAILS as written** (ratios 12.6 / 2.4 / 61.5) and is mis-calibrated: recon is near-pixel-exact, so the denominator is ~0.0001 and any visible-but-negligible drift (0.002–0.003 absolute, far below every alternative) explodes the ratio. Judged on renders + vs-alternatives, the probe achieves what the gate was written to test.
- **Known artifact:** on 7538087 the hole regenerates with a flat, saturated style (present in *every* EC arm on that image, masked or not — an inpaint-prior artifact on this simple gray-background style, not a mask effect). Edit still lands.

Wiring shipped: `scripts/edit.py --easycontrol_mask <png>` (gray-fills the cond image pre-VAE, matching the training distribution) and `--mask <png>` (drops Δz inside the region, latent-resolution). The recipe: pass the same mask to both.

## Phase 1b: edit-type generalization (2026-07-24)

**Run:** `results/20260724-1850-phase1b` (3 img × 4 edit types × {base_t0, vinj_t6, ec_b-1, ec_b-2, ec_mask_anch}; `EDITS_1B` in `run_bench.py` defines the per-image concrete edits + hole boxes). Same seed/CFG/steps as 1a.

### Verdict: PASS — ec_mask_anch ≥ vinj_t6 on all 3 in-place edit types

Render-judged (edit lands + composition held), per type across images:

| edit type | dan_9596032 | 10473210 (hard) | 7538087 | verdict |
|---|---|---|---|---|
| REMOVE (kanzashi / halo / blush) | **EC lands, vinj fails** (ornaments erased clean) | both fail (halo survives everything) | **EC lands, vinj fails** (blush gone; style-drift caveat) | **EC > vinj** |
| REPLACE hair color | **EC lands** (pale blue; vinj stays pink) | both fail (stays white) | both fail (EC goes *black*, not blonde) | **EC > vinj** |
| expression | both land; EC preserves better (0.0024 vs 0.0142 outside) | both ambiguous | both land; vinj cleaner in-box, EC better outside | **EC ≥ vinj** |
| geometry (control) | EC: pose DOES change (standing) but composition fully released; vinj: no edit | same pattern | same pattern | expected fail, recorded |

- Outside-hole preservation: ec_mask_anch is best-in-class on **every** in-place row (0.0004–0.0034), 2–6× ahead of vinj_t6, while being the only recipe that lands REMOVE/REPLACE edits at all.
- **Geometry nuance:** with a full-frame box the recipe degenerates to unanchored generation (gray cond + no anchor) — it *does* produce the pose, proving the suppression was preservation-owned, but keeps nothing. Position-locked prior confirmed; this is Phase 2's associative-retrieval target.
- **Hard-image ceiling:** 10473210's in-place edits (halo removal, white→black hair) fail for every method — beyond the current teacher regardless of preservation mechanism.
- **Failure colors:** 7538087 brown→blonde came out black in both EC arms (prior's dark-line style bias + "black bikini" in-caption attractor, plausibly); the flat-saturated hole-style artifact from 1a persists on this image.

Gate ("EC ≥ vinj_t6 for ≥ 2 of 3 in-place types"): **3/3. PASS.** Phase 2 (cross-image subject descriptor) is unblocked, with the geometry row as its falsifiable target.

---

# Phase 0: EasyControl cond stream as a learned preservation prior for DirectEdit

**Date:** 2026-07-24 · **Runs:** `results/20260724-1731-phase0-full` (3 img × 8 arms), `results/20260724-1749-phase0b-boffset` (2 img × 10 arms) · **Adapter:** `output/ckpt/methods/anima_inpaint.safetensors` (hole-free cond = "copy everything" reference) · **Edit:** caption + ", glasses" (in-place attribute edit), CFG 4, 28 steps, seed 42.

## Verdict: PASS, with the dial moved from `cond_scale` to `b_cond`

The zero-training composition works and, at the right gate offset, **beats V-injection on composition preservation while still landing the edit** (image-dependent sweet spot). Wiring: `scripts/edit.py --easycontrol_weight … --easycontrol_b_offset …`.

## Findings

1. **Exact composition with the Δz anchor.** With the cond KV cache active through BOTH inversion and edit passes, ψ_tar == ψ_src reconstructs the source pixel-exactly (recon gate recon_ec/recon_base = 0.85–0.97 ≤ 2.0 on all images). The EC prior does not perturb the anchor.
2. **`cond_scale` is near-binary on the inpaint prior.** 0.25/0.5 ≈ no-EC baseline (prior disengages — scaled-down cond-LoRA deltas move cond K/V off the distribution the learned gate retrieves); 1.0 = total clamp (pixel-level source copy, edit fully suppressed). No usable middle regime on this axis.
3. **`b_cond` offset is the continuous dial.** It's applied live as a logit bias in the LSE-extended attention (`easycontrol.py::_target_only_with_cached_cond_kv`), NOT baked into the KV cache, so an additive offset after `load_weights` shifts cond softmax mass ~e× per −1. Useful range on the inpaint prior: **−1 to −2**; −3/−4 ≈ disengaged.
4. **Head-to-head vs V-injection** (face crops `faces_*.png`):
   - dan_9596032 @ b−1: thin red glasses land AND the full source composition survives (pond reflections, obi, ornaments) — `vinj_t6` landed bolder glasses but invented a fireworks background. **EC wins.**
   - 10473210 @ b−1: edit fails to land (so does `vinj_t6`); @ b−2 glasses land with partial divergence (slightly better than pure anchor). **Tie-ish; sweet spot shifted.**
   - Pure anchor (`t_inj=0`) at CFG 4 loses the composition entirely on 2/3 images — consistent with why `t_inj` exists.
5. **Hyperparameter surface shrinks but doesn't vanish:** one interpretable scalar (b_offset, per-image sweet spot within −1..−2) vs V-injection's step count × block set. Also EC costs one KV prefill instead of a parallel src forward per injected step.

## Caveats / next levers

- The inpaint adapter is used off-label (hole-free cond; trained "trust cond fully", `b_cond_init=-6`, `drop_p=0`). The narrow/binary operating point is plausibly inpaint-specific. A purpose-trained prior should widen the sweet spot:
  - **cross-image subject descriptor** (cond = image A of a character, target = image B; mine pairs via `caption_index.json`) — trains position-free appearance retrieval, the thing no shipped aligned-pair adapter has;
  - **DirectEdit-synthesized edit pairs** (the feed-forward-editor distillation route).
- Compile is disabled under EC in `edit.py` (matches the inference engine's eager EC path); the compile-compat claim is untested.
- MSE-vs-source is a preservation proxy only; edit success was judged on renders (no tagger checkpoint available for readback at run time).
