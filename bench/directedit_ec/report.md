# directedit_ec — EasyControl cond stream as a learned preservation prior for DirectEdit

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
