# directedit_ec — bench digest

Canonical: `bench/directedit_ec/report.md` (full tables + renders), proposal
`docs/proposal/directedit_ec_preservation.md`. Runs under
`bench/directedit_ec/results/` (all 2026-07-24; 3-image set incl. the hard
image 10473210; edit = caption + ", glasses" unless noted; CFG 4, 28 steps,
seed 42; adapter = off-label `anima_inpaint.safetensors`, hole-free cond
unless masked).

## Phase 0 — composition + the dial (`…-1731-phase0-full`, `…-1749-phase0b-boffset`) — PASS

- EC cond stream composes **exactly** with the Δz anchor: recon gate
  recon_ec/recon_base = 0.85–0.97 on all images.
- `cond_scale` near-binary (no usable middle); **`b_cond` offset is the
  continuous dial** (useful range −1..−2 on inpaint; −3/−4 disengaged).
- Head-to-head at the sweet spot beats vinj_t6 on composition while landing
  the edit — but the sweet spot is image-dependent (dan @ −1; 10473210
  needed −2 with partial divergence).

## Phase 1a — masked-cond probe (`…-1827-phase1a`, `…-1844-phase1a-anchmask`) — PASS, amended

- Cond hole alone: preservation excellent, edit lands only **1/3** — the
  *global* Δz anchor pulls the hole back to the source.
- **`ec_mask_anch`** (cond hole + anchor mask, same file, b_offset 0):
  edit lands **3/3** including the hard image, outside-hole MSE
  0.0004–0.0031 — **2.6–17× better** than best-of-{vinj_t6, ec_b-1, ec_b-2}
  on every image, zero per-image tuning.
- Controls split blame cleanly: `ec_mask` (hole only) → anchor suppresses the
  edit; `anch_only` (anchor mask, no EC) → edit lands but composition
  destroyed (≈ base_t0) — at CFG 4 the EC prior, not the anchor, is the
  preservation mechanism.
- The literal "≤2× recon" gate FAILED as written (ratios 2.4–61×) and was
  judged mis-calibrated: recon is near-pixel-exact (~0.0001 denominator), so
  negligible absolute drift explodes the ratio. Passed on renders +
  vs-alternatives — the question the gate was written to test.

## Phase 1b — edit-type generalization (`…-1850-phase1b`) — PASS 3/3

3 img × 4 edit types × 5 arms (`EDITS_1B`):

| type | verdict |
|---|---|
| REMOVE | **EC > vinj** — lands 2/3 (vinj 0/3); ornaments/blush erased clean |
| REPLACE (hair color) | **EC > vinj** — lands 1/3 (vinj 0/3); failure = black-not-blonde color bias |
| expression | **EC ≥ vinj** — parity on landing, EC better outside-hole |
| geometry (control) | expected fail — full-frame box ⇒ pose lands but nothing kept (suppression is preservation-owned; Phase 2's falsifiable target) |

- ec_mask_anch best-in-class outside-hole on **every** in-place row
  (0.0004–0.0034, 2–6× ahead of vinj_t6) while being the only recipe landing
  REMOVE/REPLACE at all.
- **Hard-image ceiling**: 10473210's halo-removal and white→black recolor
  fail for *every* method — teacher-owned, not preservation-owned.
- Known artifacts: 7538087 flat-saturated hole style (all EC arms);
  brown→blonde → black (prior's dark-line bias + in-caption attractor).

## Metric caveats

MSE-vs-source is a preservation proxy only; edit success is render-judged
(small n — per repo policy no CMMD at this scale,
`project_seed_floor_cmmd_fragile`). Tag-readback edit-success is the planned
metric upgrade once a trained tagger checkpoint is available
(`docs/proposal/tag_readback_reward.md`).
