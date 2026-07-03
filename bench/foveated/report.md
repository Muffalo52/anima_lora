# Foveated denoise bench — final report (2026-07-03)

One-day arc, P0 → P3 complete, all runs in `results/`. Full per-phase detail and kill
criteria live in `plan.md`; this is the digest. The ship-side follow-up is
`docs/proposal/foveated_denoise.md` (ship proposal — production wiring).

## TL;DR

**Ships:** the **standalone deferred-foveated merge** — below σ_c=0.75, run the DiT
block stack on a reduced sequence (fovea tokens 1:1, periphery 2×2-token groups
merged; exact mean-position rope), mask derived at the crossing from free endogenous
signals (`combo` = cfgdelta + x0var). **×1.37 e2e** at 1024²/28-step/CFG, fovea
visually baseline-identical, periphery = intentional-looking soft blur. Fraction
0.35 default, knob floor 0.25.

**Closed:** the composed **foveated-Spectrum** line (spend Spectrum's refresh budget
spatially) — killed at both operating points; **partial recompute** (fovea queries
vs cached K/V) — quality ceiling ties plain Spectrum, do not build; **fovea-region
SEA triggers** — region-independent, the timing win already ships as
`schedule="sea"`; **true grid coarsening** (the paper's mechanism) — 1b already
delivers the token reduction training-free.

## Method in one paragraph

*Foveated Diffusion* (arXiv:2603.23491) merges periphery tokens at **all** steps and
needs a post-trained LoRA because mixed resolution breaks early structure
negotiation. The inversion that fits Anima (no gaze, freely-viewed art): keep the
full grid through the high-σ authority window — composition decided identically to
baseline — and only foveate **below** a cut σ_c, after structure locks
("**deferred foveation**"). The σ-timing is backed by repo findings: text drive is
front-loaded (σ>0.85), low bands lock by σ≈0.75, detail is written by self-attn+MLP
growing into low σ. Result: training-free, no RoPE surgery, image *identity*
preserved (unlike SPD, which makes a different image).

## Phase history

| phase | run | outcome |
|---|---|---|
| P0 velocity-foveation emulation | `20260703-1347/1354/1359-p0*` | PASS σ_c∈[0.5,0.75] after 2 falsified transition designs. Lessons: never rewrite the latent's noise (group-constant ε is off-manifold garbage); emulate merging as what the *compute* sees; bicubic readout (nearest = mosaic). |
| P0b Spectrum-compose smoke | `20260703-1701-p0b` | PASS @σ_c=0.5. Aggressive-Spectrum error is fovea-concentrated ×1.6–1.7 (the compose premise) — *but see P3: this only holds where the baseline is broken*. Composed knee 0.5 (bare-loop 0.75 shrinks under caching). |
| P1b real whole-stack merge | `20260703-1720-p1b` | PASS. fwd ×2.15 (492→228 ms) at 51 % tokens, **e2e ×1.37 @σ_c=0.75**; fovea RMSE 0.065/0.027 — *better* than the emulation. Merged rope = renormalized cos/sin mean (exact for symmetric groups). Startup invariants bit-exact. |
| P1a foveated Spectrum | `20260703-1744-p1a` | PASS as measured (fovsea −0.007 fovea at matched forwards; fovblend exactly neutral) — mechanism story later overturned by P2t. |
| P2 mask shapes | `20260703-1910-p2` | Placement dominates (miss = 2.3× subject error); moderate fragmentation free, **scatter falsified** (+47 % — isolated cells lose their attention neighborhood); periphery bokeh only clean on low-contrast content (moiré on hard edges). Masks must be compact + subject-following. |
| P2t trigger attribution | `20260703-1920-p2t` | **The 1a win is NOT foveal** — every trigger region (incl. anti-oracle background) gives the same schedule and numbers. It is generic SEA-vs-window timing; production `schedule="sea"` already ships it. |
| P2f blend masks | `20260703-1932-p2f` | fovblend neutrality is mask-shape-independent (emit is per-token, no attention coupling) — recompute region freely choosable. |
| P2b mask sources | `20260703-1948-p2b` | **`combo` (cfgdelta + x0var) is the v1 source** — never loses to static rect: tie on centered default, **−29 %** subject RMSE on 3-subject channel6 (95 % face cover), −6 % ootomo. cfgdelta is prompt-aware (ignores the texture trap); x0var over-selects busy texture (confirmed, morphology contains it). Zero extra models/forwards. |
| P3 partial recompute + mergefresh | `20260703-2019-p3` (aggressive), `20260703-2028-p3prod` (production) | See below — the composed line closes. |
| P3s fraction ladder | `20260703-2036-p3s` | No free stretch; knee 0.25, cliff ≤0.15; speed ceiling ~×1.6. |

## Phase 3 — why the composed line closed

Same arms at two Spectrum operating points, hard prompts included for the first time
on the composed stack:

1. **Aggressive point (window 3/flex 3/warmup 4 — the 0b/1a calibration point):
   plain `spec` collapses on hard prompts by itself** — channel6 renders corrupted
   red faces, inverted hair colors, dissolved twintails with zero foveation involved
   (subject RMSE 0.41 vs 0.20 on default). All earlier composed-line results were
   eyeballed on the default prompt only. Foveated arms can only partially *rescue*
   a broken baseline (`mergeall75` −8 % = rescue, not a win).
2. **Production point (window 2/flex 0.25/warmup 6): `spec` ≈ `full` even on
   channel6** — there is no error budget to reallocate. Merged refreshes save ~5 %
   wall-clock and visibly wash out periphery subject parts (a 0.35 mask cannot cover
   three full-body subjects).
3. **Partial recompute (3b) is dead at both points**: the emulated quality ceiling
   (`pr_all` — every below-σ_c step a fovea-anchored actual) ties `spec` exactly.
   Posterity fact for any future revisit: frozen-periphery anchoring *does* degrade
   at sane cadence (comp Δ+0.017–0.032) and a full re-anchor every 2nd step
   (`pr_catchup2`, = 1c) restores exact neutrality (Δ≤0.0002) — mandatory cadence if
   ever built. But there is no quality upside to pay for the surgery.
4. "fovsea/fov1a + combo" as a method **reduces to `schedule="sea"`** (P2t trigger
   region-independence + P2f blend neutrality) — already shipped in production and
   the ComfyUI node.

## Phase 3s — the fraction knob

`combo` mask, σ_c=0.75, real merge path. Subject RMSE (cover, e2e×):

| frac | default | channel6 | ootomo |
|---|---|---|---|
| rect@0.35 | **0.065** (100 %, ×1.37) | 0.196 (68 %, ×1.36) | 0.133 (93 %, ×1.35) |
| 0.35 | 0.066 (68 %, ×1.38) | **0.139** (95 %, ×1.36) | **0.125** (×1.37) |
| 0.25 | 0.085 (×1.41) | 0.166 (89 %, ×1.39) | 0.136 (×1.43) |
| 0.20 | 0.099 (×1.48) | 0.177 (86 %, ×1.44) | 0.152 (×1.47) |
| 0.15 | 0.113 (×1.54) | 0.193 (74 %, ×1.49) | 0.153 (×1.51) |
| 0.10 | 0.124 (×1.54) | 0.248 (58 %, ×1.54) | 0.183 (×1.55) |

Monotone decay — no free stretch. Knee 0.25 (still beats the static rect on hard
prompts); cliff ≤0.15 on multi-subject (masks start dropping faces). Eyeball at
0.20: faces crisp, bodies mush — on subject-dense prompts the periphery *is* the
characters. Speed is bounded ~×1.6 by 2×2 pooling (eff tokens = f + (1−f)/4); the
next speed lever is `merge_edge=4` (generalized in `FoveatedTokenMerge`, exact-rope
argument carries — unbenched), not a smaller fovea.

## Load-bearing lessons (beyond this bench)

- **Never rewrite the latent's noise.** Variance-correct group-constant ε is
  distribution-wrong (f²× per-mode LF power) — the DiT reads it as content.
- **Calibrate on hard prompts before believing a composed-stack result.** The
  aggressive-Spectrum collapse was invisible for four phases because every verdict
  eyeballed the centered default prompt. RMSE certified change, not improvement,
  exactly as the no-quality-metric invariant warns.
- **Emulation arms that look great can be cost mirages** — `pr_catchup2` renders
  pristine channel6 while being the *slowest* arm; its attractive look is spec's
  own quality plus full compute, and its hypothetical FFE never beats spec.
- **cfgdelta**: accumulated |v_cond − v_uncond| is a free, prompt-aware subject
  localizer — reusable wherever a "where is the prompt steering" map is needed.
- Attention isolation between fovea and periphery is strong (P0: fovea survived a
  garbage periphery); isolated fovea *cells* however lose fidelity (P2 scatter) —
  compactness is about the attention neighborhood, not aesthetics.

## Ship spec (carried into the ship proposal)

Mechanism: σ-gated whole-stack token merge (1b) + endogenous combo mask (2b) +
bicubic merged readout. Defaults: `sigma_c=0.75`, `fovea_frac=0.35` (floor 0.25),
2×2 pooling, morphology open→close→dilate-1, fraction re-solved after morphology.
CFG required for cfgdelta (fallback: x0var-only or static rect). Expected: ×1.37 e2e,
fovea baseline-identical. Composition with Spectrum: possible (`mergefresh` measured
safe) but not default — tiny saving, visible periphery cost at sane schedules.
