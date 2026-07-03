# Deferred foveation — spatial effort allocation in the detail phase

> **Naming (2026-07-03)**: the method is **deferred foveation** — the name encodes the
> load-bearing inversion (foveation locked out until σ ≤ σ_c, because foveating during
> the high-σ authority window produces a *different image*; deferring it preserves image
> identity and is what makes the line training-free). "Deferred-foveated merge" = the
> real token-merge mechanism (Phase 1b); the static rectangle is the *mask* being
> "fixed", a Phase-2-replaceable detail, not part of the method name. σ_c is a knob,
> not a constant: emulation knee [0.5, 0.75] bare-loop, ≈0.5 under aggressive Spectrum.

Status: **Phases 0, 0b, 1b, and 1a ALL PASS (2026-07-03, `bench/foveated/`).** 0b:
headroom confirmed (forecast error ×1.6–1.7 fovea-concentrated; compose knee σ_c≈0.5).
1b (**deferred-foveated merge**, owner-reordered first): real whole-stack token merge
ships **×1.37 e2e @ σ_c=0.75** (fwd ×2.15 at 51% tokens), fovea 0.065/0.027 — better
than the emulation; standalone knee 0.75. 1a (**foveated Spectrum**): fovea-cropped SEA
trigger beats plain aggressive Spectrum on fovea fidelity at *identical* forward count
(0.1909 vs 0.1983, visibly cleaner sign text; periphery also −10%), and the
periphery-rides-forecast emit is exactly neutral — partial-recompute foundation
validated. Remaining: production wiring, 1c periphery catch-up, asymmetric cadence /
partial recompute (Phase 3), mask sources (Phase 2). Full run history:
`bench/foveated/plan.md`. Origin: *Foveated Diffusion* (Chao,
Yariv, Xiao, Wetzstein — arXiv:2603.23491, same Wetzstein-lab lineage as SPD), inverted
to fit Anima's output contract.

## The idea, and why it isn't the paper

The paper allocates tokens non-uniformly in **space** (full density in a gaze-derived
foveal mask, 2×2-merged periphery) at **all** steps, and needs a post-trained LoRA +
mixed-res RoPE because naïve mixed-resolution denoising breaks *global structure
negotiation* (their Fig. 4/5: duplicated entities, scale mismatch). Its quality claim is
gaze-contingent — the periphery genuinely loses resolution, acceptable only when you know
where the user is looking. Anima's outputs are freely-viewed art; there is no gaze.

The inversion that fits here: **keep the full-res grid through the high-σ authority
window** — composition, content addressing, and identity are decided *identically to
baseline* — then below a cut σ_c stop spending detail effort outside a foveal mask.
The σ-timing is backed by repo findings:

- Cross-attn addressing burst lives at σ>0.9; text drive decays below σ≈0.85
  ([[project_crossattn_drive_frontloaded]], `docs/findings/crossattn_self_attn_dominance.md`)
  — content placement completes inside the full-effort window.
- Low bands lock by σ≈0.75; the top band resolves only by σ≈0.29 (`bench/spd/` Phase 1)
  — the periphery forfeits exactly the late top-band work, which is the *definition* of
  "less important there".
- Detail is written by self-attn+MLP, which grow into low σ — that's where the effort
  (and any saving) lives.

Because structure is locked before any mixed-effort computation starts, the paper's
naïve-failure mode is dodged by construction — no post-training, no RoPE surgery, in the
emulated form. Distinct from SPD (which runs *globally* low-res early and produces a
**genuinely different image**; this line's contract is *fidelity*: same image where it
matters). Guard check: the resolution-curriculum shelf entry
([[project_shelved_explorations]]) explicitly left "σ×res coupling in a genuinely
compute-bound setup" as the one coherent residual — inference is that setup; this is the
σ×res×**space** instance. The killed low-σ tag levers were attention-*reweighting*;
compute *placement* is a different lever and not covered by that guard.

## Phase 0 — quality contract, training-free ✅ PASS (2026-07-03)

`bench/foveated/probe_velocity_foveation.py` — emulates masked token merging at the
sampler boundary with zero attention surgery: below σ_c the DiT evaluates on a composite
where the periphery is `pool(z)` (the merged-token view real merged attention would see),
the periphery velocity is group-pooled (all tokens in a 2×2-token group share one
update), the latent itself is never rewritten, and the periphery is read from the merged
representation once at decode (bicubic up — the paper's smooth Up(·)). Full run history
and kill criteria in `bench/foveated/plan.md`; the load-bearing lessons:

1. **Never rewrite the latent.** The "principled" transition re-noising (pooled x̂₀ +
   variance-renormalized pooled ε) matches per-pixel variance but not distribution —
   group-constant ε is off-manifold block noise (f²× per-mode LF power) and the DiT reads
   it as content; the periphery never converges. Emulate merging as *what the compute
   sees*, not what the state stores.
2. **`tome_eval` beats `frozen`**: letting the model keep seeing σ_c-scale HF noise in
   the periphery (`frozen`) produces speckle; the pooled composite view is both the
   faithful emulation and the cleaner one.
3. **Readout matters**: nearest broadcast reads as mosaic; bicubic reads as anime
   DOF/bokeh — a *natural* look for these images.

Result (bare DiT, 1024², 28 steps, CFG 4, flow_shift 3, 2 seeds): fovea near-identical at
σ_c=0.5 (RMSE 0.032 vs baseline), composition-identical with minor drift at 0.75 (0.089),
real detail drift at 0.9 (0.20). Periphery = clean soft blur, no grain/seams/tone shift.
Quality knee **σ_c ∈ [0.5, 0.75]** — the same single-late knee SPD found. Inverse-mask
control destroys the subject symmetrically (probe has teeth); fovea/periphery isolation
through attention is strong.

## The arithmetic that reshapes the plan

Phase-1-style real merging (fovea 35%, 2×2-token groups → ~51% effective tokens post-σ_c)
saves ~×1.1 at σ_c=0.5, ~×1.25–1.3 at 0.75 — **standalone speed cannot be the point**
(Spectrum ships ×3.75). Composing *with* Spectrum is structurally clean but marginally
valuable as speed: Spectrum is step-level whole-DiT caching (Chebyshev per-token feature
forecasting; cached steps skip all 28 blocks), so it reduces the *count* of actual
forwards while merging reduces *cost per forward* — orthogonal, multiplicative, and
unlike SPD the token layout never changes so the forecaster's per-token time series
survives σ_c intact (no SPEED-style naive-reset; just force one actual forward at the
crossing, machinery Spectrum already has). But at ×3.75 only ~4–5 actual forwards remain
below σ_c → merging buys ×3.75→~×4.2. Real, not compelling.

**The reframing that survives: spend Spectrum's error budget spatially.** The bokeh
periphery is exactly the regime where feature forecasting is strongest (smooth,
low-frequency, slowly-evolving), and the fovea is where forecast error hurts. So the
prize is not "merge ∘ Spectrum" but **foveated Spectrum**: periphery rides the forecast
nearly permanently; the refresh budget concentrates on the fovea. At matched wall-clock
the claim becomes *"aggressive-Spectrum + foveation beats aggressive-Spectrum alone on
subject fidelity"* — a quality-allocation claim, immune to the speed arithmetic, and the
honest version of "concentrate denoise where it's needed".

## Phase 0b — Spectrum compose smoke ✅ PASS at σ_c=0.5 (2026-07-03)

`bench/foveated/probe_spectrum_compose.py`; wiring = `spectrum_denoise(foveation=...)`
hook points (composite input on actual forwards, output-side pooling of every emitted
v — forecast steps included, post-unpatchify so forecaster state is untouched, final
readout pool, plus the forced actual forward at the σ_c crossing; DCW/SMC warned and
ignored while active). Run `20260703-1701-p0b`: aggressive Spectrum (window 3 / flex 3 /
warmup 4 → 10/28 forwards, ×2.8), 2 seeds:

- **Headroom CONFIRMED** — the readout that de-risks the whole reframe: plain aggressive
  Spectrum's forecast error concentrates in the fovea (fovea/periph RMSE ratio
  1.62–1.74; errmap bright on sign lettering / face / hands, near-black sky). Its
  visible damage is *subject* damage (garbled sign text, smear artifacts on hands)
  while the periphery survives — exactly the allocation asymmetry foveated Spectrum
  wants to exploit.
- **Composition stable and free at σ_c=0.5**: fovea RMSE +0.002 over plain spec
  (visually identical drift), same forward count, periphery clean bokeh, no
  instability through the σ_c feature discontinuity.
- **σ_c=0.75 fails under aggressive caching** (+0.029 fovea RMSE, visibly worse text):
  the bare-loop knee [0.5, 0.75] shrinks to **≈0.5** when most post-crossing steps are
  forecast. Phase 1a operates at σ_c=0.5.
- Scope: 0b's wiring doesn't reallocate refresh budget yet, so foveated arms only
  *matched* spec's fovea — the "beats aggressive Spectrum on subject fidelity at
  matched compute" claim is Phase 1a's to demonstrate, now with its premise measured.

## Phase 1 — pick the mechanism — NEXT GATE (0b passed → default branch is 1a)

Two branches; 0b's outcome picked the default:

- **1a. Foveated Spectrum decisions** (if 0b shows the quality-allocation win): make the
  refresh decision spatially aware. Cheapest form first — global refresh cadence
  unchanged, but *periphery* features keep riding the forecast even on refresh steps
  (only fovea features update from the actual forward: mask-blend actual vs forecast at
  the final_layer hook). No attention surgery at all. Then, if worth more: per-region SEA
  accumulators driving asymmetric cadences.
- **1b. Real ToMe merge inside blocks** (if 0b shows composition is fragile and
  standalone fidelity-preserving speedup is wanted): merge 2×2 periphery token groups
  before each block's attn/MLP, broadcast after; latent stays full-res end-to-end. RoPE
  for a merged group = its mean position (approximation is the known ToMe cost).
  Measure actual wall-clock at the P0 knee. Bench + invariant test per CONTRIBUTING
  Tier 1.5.

## Phase 2 — mask sources (gated on Phase 1 shipping anything)

P0's movable rectangle is a stand-in, and it showed the contract issue: a rectangular
sharpness boundary is invisible crossing bokeh-able background but would read as an
artifact crossing the subject. Real masks, both training-free and endogenous:

- **FreeText Stage-1 localizer** — I2T cross-attn "where is concept X" (concentration
  2–3.6× uniform, L6–L17, mid-t; the one reusable win from the FreeText line) read
  during the full-res steps, union over subject tags.
- **Local x̂₀ variance** across early steps — "which regions are still undecided" (x̂₀
  wander tracks scene complexity and is base-owned, [[project_x0_contradiction_bench]]).
- Snap the union to pool groups, dilate one group, keep a `--fovea_frac` floor.

Gate: masks must cover the subject on a 10–20 prompt sweep (montage eyeball) without
exceeding ~50% area (or the saving/allocation evaporates).

## Phase 3 — heavy variants (gated on Phase 1 + demand)

Only if the line earns it:

- **Partial recompute** (foveated-Spectrum endgame): on refresh steps run fovea queries
  only against cached/forecast K/V per block; MLP on fovea tokens only. Real savings on
  refresh steps (~65% of block cost at 35% fovea) without changing token layout.
- **True grid coarsening** (the paper's mechanism, time-gated): actual sequence-length
  reduction post-σ_c — phase-aligned mixed-res RoPE, likely a post-trained adapter
  (SPD-Case-B-style analytic targets, no teacher), low-res VAE decode + blend (their
  known boundary color-artifact risk). Biggest speed, most surgery; competes with 1b.

## Invariants / gotchas for whoever builds this

- On-manifold lesson from P0 run 1: never rewrite the latent's noise; group-constant
  renormalized noise is off-manifold garbage even with correct per-pixel variance.
- Mask must be built on the pooled grid (no group straddles the boundary) — see
  `build_fovea_mask`.
- Force an actual forward at the σ_c crossing when composing with Spectrum (feature
  discontinuity vs the Chebyshev fit).
- Final readout is part of the contract: periphery is *read from the merged
  representation* (bicubic up), otherwise never-denoised HF noise/detail survives to
  decode.
- 5D/4D boundary discipline: all pooling ops are 4D `(B,C,H,W)` — `squeeze(2)` /
  `unsqueeze(2)` explicitly (see CLAUDE.md dim-2 invariant).
- DCW/SMC compose with Spectrum at the same boundary and should compose here, but are
  unvalidated against pooled velocities — warn-and-ignore until checked (mirror SPD's
  posture).
- No metric we have tracks Anima sample quality ([[project_shelved_explorations]],
  Null-TTA lesson): every gate above is a full-res montage eyeball; RMSE/Laplacian
  numbers certify *change*, not *improvement*.
