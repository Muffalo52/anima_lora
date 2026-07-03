# Foveated denoise — spatial effort allocation in the detail phase

Status: **Phase 0 PASS (2026-07-03, `bench/foveated/`); Phase 0b (Spectrum compose
smoke) is the next gate; Phases 1–3 proposed.** Origin: *Foveated Diffusion* (Chao,
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

## Phase 0b — Spectrum compose smoke — NEXT GATE

Wire the P0 velocity-foveation into `spectrum_denoise` (both live at the sampler
boundary; composite input on actual forwards, output-side pooling of every emitted v —
forecast steps included, it's post-unpatchify so forecaster state is untouched, final
readout pool, plus the forced actual forward at σ_c).

**Hypothesis to falsify:** at aggressive Spectrum settings (e.g. `flex_window 3.0`,
~×5 forward-count reduction) + foveation σ_c=0.75, the fovea is visually closer to the
full-compute baseline than plain aggressive Spectrum's fovea region is, at equal or lower
actual-forward count.

- Judge: same-seed montage eyeball (fovea crops), fovea RMSE as change-certifier only.
- KILL the composition (not the line) if foveation adds visible fovea damage on top of
  Spectrum's, or if forecast error through the σ_c discontinuity destabilizes cached
  steps despite the forced forward.
- Also read out: does the σ_c crossing inflate SEA/forecast residuals in the *fovea*
  (leak through shared AdaLN/attention)?

Cheap: ~half-day wiring + one bench run. PASS → Phase 1 chooses its branch with data.

## Phase 1 — pick the mechanism (gated on 0b)

Two branches; 0b's outcome decides the default:

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
