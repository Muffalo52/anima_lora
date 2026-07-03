# Foveated denoise — selective effort in the detail phase

> **BENCH COMPLETE (P0→P3, 2026-07-03).** Digest: **`bench/foveated/report.md`**.
> Ship-side follow-up: **`docs/proposal/foveated_denoise.md`** (now the SHIP proposal
> for production wiring; the original research proposal is archived at
> `_archive/proposals/foveated_denoise_research.md`). This file is the full run
> history with per-phase kill criteria and numbers.

Premise (inverse of SPD, spatial instead of temporal): keep the **full-res grid through
the high-σ authority window** (composition decided identically to baseline), then below a
cut σ_c stop spending detail effort on the *periphery* — the region outside a foveal mask
— by sharing one update per 2×2-token group. Inspired by *Foveated Diffusion* (Chao,
Yariv, Xiao, Wetzstein — arXiv:2603.23491), but time-gated so the paper's naïve-mixed-res
structural failures (duplicated entities, scale mismatch — their Fig. 4/5, all *early
structure negotiation* failures) are dodged by construction: mixed effort only starts
after structure locks.

Why the σ-timing should work (repo findings):
- Cross-attn addressing burst lives at σ>0.9; text drive decays below σ≈0.85
  (`docs/findings/crossattn_self_attn_dominance.md`) — content placement is inside the
  full-effort window.
- Low frequency bands lock by σ≈0.75, top band only by σ≈0.29 (`bench/spd/` Phase 1) —
  the periphery forfeits exactly the late top-band work.
- Detail is written by self-attn+MLP which *grow* into low σ — that's where the effort
  (and therefore the saving) is.

Gated — each phase is cheap relative to the next and can kill the idea.

## Phase 0 — quality contract, training-free (velocity-foveation probe)

**Hypothesis to falsify:** sharing one velocity update per 2×2-token group in the
periphery below σ_c (a) leaves the fovea near-identical to the same-seed baseline and
(b) degrades the periphery only to an acceptable "soft half-res detail" — no grain, no
seams, no tone shift, no structural damage.

**Script:** `probe_velocity_foveation.py` — emulates block-level token merging (build B)
at the sampler boundary with zero attention surgery: below σ_c the periphery velocity is
avg-pooled per token group and broadcast back, and at the transition the periphery latent
is rebuilt once from its pooled x̂₀ / renormalized pooled ε split (so per-token noise
stats still match the global σ — the "brittle without re-noising" trap the paper warns
about). `transition_op` controls (`pool_plain` = walk into the σ-stats trap, `none` =
frozen-HF-noise demo) exist to show the probe has teeth.

This probe measures **quality only** — velocity foveation computes the full grid every
step, so there is no speedup; the saving arrives in Phase 1 if quality passes.

**Kill criteria** (verdict is a full-res eyeball call on `compare_seed*.png`; auto-metrics
only certify change / hard divergence — no metric we have tracks Anima sample quality):
- fovea visibly diverges from baseline (composition/identity change) at σ_c ≥ 0.75, or
- periphery shows grain / block seams / tone shift obvious at full res even at the most
  conservative σ_c.

PASS → Phase 1. FAIL at all σ_c → the whole line dies (a trained adapter could still
rescue it, but that forfeits the training-free selling point — probably not worth it
given Spectrum ×3.75 already ships).

### Result (2026-07-03, bare DiT, 1024², 28 steps euler, CFG 4, flow_shift 3, 2 seeds)

Three iterations, each falsifying its predecessor's transition handling:

| run | mode | outcome |
|---|---|---|
| `20260703-1347-p0` | `split_renorm` latent rebuild | **FALSIFIED** — variance-correct but distribution-wrong: group-constant ε is off-manifold block noise (f²× per-mode LF power); periphery = rainbow blocks that never converge. Two facts survive: fovea/periphery isolation is strong (fovea RMSE 0.035 at σ_c=0.5 *despite garbage periphery*), and the inverse-mask control mirrors perfectly (probe has teeth). |
| `20260703-1354-p0v2` | `tome_eval`/`frozen`, latent untouched, nearest final readout | Mechanism works. `tome_eval` (DiT evaluates the pooled-periphery composite — the true merged-token view) beats `frozen` (model keeps seeing σ_c-scale HF noise → sky speckle). Periphery clean but mosaic-blocky = readout artifact. |
| `20260703-1359-p0v3` | `tome_eval` + **bicubic** final readout, fovea raised to face | Periphery reads as anime DOF/bokeh blur — no grain, seams, or tone shift. Fovea: σ_c=0.5 near-identical (RMSE 0.032); σ_c=0.75 composition identical, minor detail drift (0.089); σ_c=0.9 composition intact but real detail drift (sign font/hands, 0.20). |

**Verdict: PASS at σ_c ∈ [0.5, 0.75]** — pending owner eyeball. Same knee SPD found
(single-late σ0.5). σ_c=0.9 fails criterion (a)'s spirit at the detail level.

**Phase-1 cost sketch** (fovea 35%, pool 4 ⇒ 4 tokens/group ⇒ ~51% effective tokens
post-σ_c): σ_c=0.5 foveates only the last 7/28 steps → ~×1.1; σ_c=0.75 → 14/28 steps →
~×1.25–1.3. Modest standalone — the case rests on composing with Spectrum (orthogonal:
block skipping vs token count) and on quality (fovea fidelity) rather than raw speed.
Also note the rectangle-shaped sharpness boundary is fine where it crosses bokeh-able
background but would look odd crossing a subject — Phase 2's subject-following mask
matters for the contract, not just for saliency.

## Phase 0b — Spectrum compose smoke (`probe_spectrum_compose.py`)

### Result (2026-07-03, run `20260703-1701-p0b` — bare DiT, 1024², 28 steps euler CFG 4,
aggressive Spectrum window 3 / flex 3 / warmup 4 ⇒ 10/28 actual forwards, 2 seeds)

Wiring: `spectrum_denoise(foveation=...)` hook points (composite eval-view on actual
forwards below σ_c, output-side pooling of every emitted v including forecasts, one
forced actual at the crossing, bicubic merged readout). Arms: `full` (all-actual),
`spec`, `spec_fov_sc{0.75,0.5}`.

1. **Headroom CONFIRMED — the reframe's premise holds.** Plain aggressive Spectrum's
   error vs full-compute concentrates exactly where the fovea is: fovea/periphery RMSE
   ratio **1.62–1.74**, and the errmap is stark (sign lettering / face / hand edges
   bright, sky near-black). Visually the damage is garbled sign text and pink smear
   artifacts on hands — subject damage, while the periphery stays essentially fine.
   Spatial refresh allocation has something real to buy.
2. **Compose smoke at σ_c=0.5: PASS.** Fovea RMSE 0.2000 vs spec's 0.1983 (+0.002 =
   noise; visually the same drift), same 10-forward count (the forced crossing
   coincided with a scheduled refresh), periphery = clean bokeh, no NaN/std blow-up —
   cached steps ride through the σ_c feature discontinuity fine.
3. **σ_c=0.75 FAILS under aggressive caching** (+0.029 fovea RMSE, visibly worse text,
   +1 forward). The bare-loop P0 knee [0.5, 0.75] shrinks to **≈0.5** when most
   post-crossing steps are forecast: the composite eval-view kinks the feature
   trajectory mid-fit where the forecaster carries many cached steps. Operating point
   for Phase 1: σ_c=0.5.
4. Honest scope note: this wiring does **not** yet reallocate refresh budget (that is
   Phase 1a), so foveated arms were never *better* than spec in the fovea — 0b only
   establishes the premise (1) and the composition's stability/cost (2). The
   quality-allocation win is Phase 1a's claim to demonstrate at matched wall-clock.

**Verdict: PASS (σ_c=0.5) — proceed to Phase 1a (foveated Spectrum decisions), pending
owner eyeball.**

## Phase 1 — real savings: masked token merge inside blocks (ToMe-style) — GATED on P0

Merge 2×2 periphery token groups before each block's attn/MLP, broadcast after; latent
stays full-res end-to-end (no VAE blending, no RoPE surgery — merged groups sit at their
mean position's RoPE only inside attention). Measure actual wall-clock vs quality at the
P0-surviving σ_c. Composability with Spectrum (block skipping — orthogonal) is the
interesting aggregate number.

### Phase 1b result — **deferred-foveated merge**, PASS at both σ_c (2026-07-03, run
`20260703-1720-p1b`, `probe_token_merge.py`)

Owner reordered 1b before 1a (real production wall-clock first); the method was named
**deferred foveation** here — the load-bearing idea is the σ-gate (foveating inside the
authority window makes a *different image*), not the mask shape.

Implementation: whole-stack merge (not per-block round-trips — P0's final pooled
readout discards within-group detail anyway, which is all per-block would preserve).
Merge once after patch embed (fovea 1:1, periphery 2×2-token cells averaged; 4096 →
2107 tokens at fovea 35%), run the model's own `_run_blocks` on the fake-5D
`(B,1,L_red,1,D)` layout, broadcast before `final_layer`. Merged rope = renormalized
elementwise mean of the 4 members' (cos,sin) rows — **exact** mean-position rope for a
symmetric 2×2 group (angles symmetric about center ⇒ mean vector keeps the center
angle; renorm restores unit norm). Startup invariants bit-exact (custom runner ≡
`anima()`, all-fovea merge ≡ identity, both max|Δ|=0.0).

Numbers (bare DiT, eager, 1024², 28 steps euler CFG 4, 2 seeds):
- **fwd ×2.15** (492→228 ms) at 51% tokens — beats the naive token-ratio estimate
  (attention's quadratic term).
- **e2e ×1.37 @ σ_c=0.75, ×1.16 @ σ_c=0.5** (proposal's sketch said ~×1.25–1.3 @0.75).
- Fovea RMSE vs baseline: **0.065 @0.75 / 0.027 @0.5** — both BETTER than the P0v3
  emulation (0.089 / 0.031): real merged attention hands fovea queries processed merged
  features, gentler than the emulation's pooled-latent view. Fovea crops visually
  baseline-identical at both σ_c (sign text crisp); periphery clean bokeh (lap ×0.23–0.30).

**The real-merge standalone knee is 0.75** (unlike the Spectrum-composed knee of 0.5
from 0b — different regimes, don't conflate). Standalone ship-shape: σ_c=0.75 → ×1.37.
Pending owner eyeball.

## Phase 1a result — foveated Spectrum refresh allocation, PASS (2026-07-03, run
`20260703-1744-p1a`, `probe_foveated_spectrum.py`)

Bench-local spectrum loop (production `spectrum_denoise` untouched), same aggressive
point as 0b (window 3 / flex 3 / warmup 4 → exactly 10/28 forwards in every arm).
Two zero-surgery mechanisms + compose, all at **exactly matched compute**:

| arm | mechanism | fovea RMSE vs full | periph RMSE |
|---|---|---|---|
| `spec` | plain window schedule | 0.1983 | 0.1180 |
| `fovblend` | emit = fovea-actual + periphery-forecast below σ_c=0.5 (fits stay full-anchored) | 0.1984 (+0.0001) | 0.1192 |
| `fovsea` | SEA trigger distance on the **fovea crop only**, δ matched to window refresh fraction | **0.1909 (−0.0073)** | **0.1060** |
| `fov1a` | both | 0.1910 | 0.1064 |

- **Gate PASS**: `fovsea` beats `spec` on fovea fidelity at identical forward count, and
  the win is *visible* (seed-41 sign legible vs spec's glassy mush; lighter hand smear,
  cleaner face) — RMSE understates it because damage is localized. Mechanism: the
  fovea-driven trigger fires the mid-run refreshes earlier ([6,11,18] vs [6,12,21]),
  shortening tail extrapolation where fovea error compounds. Periphery *also* improved
  (−10% rel) — reallocation, not a trade.
- **`fovblend` is exactly neutral** — the more load-bearing result: periphery riding
  the forecast on refresh steps costs nothing, which is the foundation the
  partial-recompute escalation (fovea-only queries on refresh steps) stands on.
- Caveats: 2 seeds / 1 prompt; effect modest in aggregate RMSE; blend arm tested with
  fits anchored to full actuals (partial recompute won't have periphery actuals — that
  degradation is untested; needs a no-periphery-update arm before building Phase 3).

Escalations now justified: (i) asymmetric cadence (fovea-δ + a slow periphery re-anchor
cadence → crank global aggressiveness at held fovea quality), (ii) Phase-3 partial
recompute (refresh = fovea queries vs cached K/V ≈ ⅓ forward cost → 2–3× more fovea
re-anchors per wall-clock).

> **RE-ATTRIBUTED by Phase 2t (below): the fovsea win is NOT foveal.** Every trigger
> region (including anti-oracle background) produces the same schedule and the same
> numbers — the win is generic SEA-vs-window timing at this aggressive point. The
> fovea-fidelity gate result stands; the *mechanism* story and escalation (i) do not
> (fovea-δ is meaningless when δ is region-independent). Escalation (ii) partial
> recompute is untouched — it rests on `fovblend` neutrality, not on trigger timing.

## Phase 2 — mask shape/placement controls, then mask sources — GATED on P0 (open)

P0/1b used a single oracle-placed center rect. Before building real mask *sources* (the
FreeText Stage-1 localizer — endogenous I2T cross-attn "where is concept X", read during
the full-res steps — ∪ high local-x̂₀-variance regions), Phase 2 first runs the shape and
placement **controls** those sources would have to beat. Cost is fraction-only (attention
is global — layout is free), so all arms run at matched fovea fraction on the real 1b
merge path (`FoveatedTokenMerge` takes any binary cell mask; merged-rope exactness is
per-2×2-group, mask-independent). `probe_mask_shapes.py`, arms:

| arm | tests |
|---|---|
| `rect` (oracle center) | reference — the 1b configuration |
| `rect_miss` (least-oracle-overlap random rect) | placement null — what a mis-aimed mask costs the subject. NB at frac 0.35 a rect can't fully miss the center subject (min overlap ~30% — corner placement); `subject_cover` is logged. |
| `multi3` (3 random rects, frac/3 each) | moderate fragmentation — the "multiple subjects" primitive |
| `scatter` (random cells) | max boundary perimeter — shape stress; also a "graded foveation" data point (32px blur cells everywhere) |

`rect` vs `rect_miss` isolates placement; `rect_miss` vs `multi3` vs `scatter` isolates
fragmentation with placement held random. Readouts: own-mask RMSE (mechanism holds per
shape?), fixed subject-box RMSE in every arm (placement value), own-periphery Laplacian,
boundary-ring Laplacian (seam/halo flag), matched-cost sanity. Gate: fragmentation ≈
free (own-mask RMSE ≈ oracle rect) AND placement matters (rect_miss subject damage ≫
rect) → derived masks are worth building and must beat the placement null. Known
contract caveat regardless of outcome: hard mask edges crossing a *subject* look odd
(P1 note) — a shipped mask source should be morphologically closed/dilated and
subject-following; scatter violates this by construction (that's its job).

### Result — gate PASS, with two shape constraints (2026-07-03, run
`20260703-1910-p2` — bare DiT, 1024², 28 steps euler CFG 4, σ_c=0.75, frac≈0.352,
2 seeds; startup invariants bit-exact)

- **Matched cost confirmed**: every shape ×1.37–1.39 e2e (same token count) — layout is
  free, fraction is the only cost knob, as designed.
- **Placement dominates**: subject RMSE 0.065 (`rect`) vs 0.151 (`rect_miss`, residual
  cover 30% — geometric floor at this frac). Visually the missed subject is destroyed
  (face gone, sign unreadable). Saliency has large value to capture; derived masks must
  beat this null and are worth building.
- **Moderate fragmentation ≈ free; maximal is NOT.** Own-mask RMSE at held-random
  placement: `rect_miss` 0.072 → `multi3` 0.077 (+7%, fine) → `scatter` 0.106 (+47%).
  Isolated fovea cells lose fidelity — their entire attention neighborhood is merged —
  and visually scatter is a *streaky smear*, not the hoped uniform "graded foveation"
  softness. Shipped masks must be compact blobs (morphologically close/dilate any
  derived saliency).
- **Periphery blur is only clean bokeh on low-contrast content.** High-contrast content
  landing in the periphery (the sign text in `rect_miss`, both seeds) shows moiré /
  checker aliasing from 4× pooling + bicubic on hard edges — 1b's clean-bokeh readout
  was partly a property of its low-contrast periphery (sky/grass). Second independent
  reason the mask must be subject-following.
- Boundary rings show no halo/sharpness spikes (ring Laplacian ratio 0.31–0.71 < 1
  everywhere); the multi3 rect edge crossing the sign mid-word confirms the P1
  half-sharp-subject contract note. `multi3` subject RMSE tracks coverage (0.157 @35%
  → 0.120 @58%), consistent with placement being the whole story.

**Next (Phase 2b — mask sources):** derive masks from x̂₀-variance (cheap, endogenous,
read during full-res steps) and/or the localizer; morphological close + dilate to
compact blobs; evaluate bracketed between `rect` (oracle) and `rect_miss` (null) on
subject RMSE at matched fraction.

### Phase 2b design (agreed 2026-07-03) — endogenous sources, hard dataset prompts

v1 is purely **endogenous** (zero extra models / forwards; SAM3/PE-Spatial saliency is
the escalation only if these fail). Two free signals, computed during the full-res
steps before the σ_c crossing:

- **`cfgdelta`** — accumulated per-cell |v_cond − v_uncond| over pre-crossing steps.
  Both branches exist under CFG anyway; the delta marks where the prompt is steering =
  a prompt-aware subject localizer (text drive is front-loaded, so the signal is
  complete before crossing).
- **`x0var`** — per-cell HF (Laplacian) energy of x̂₀ at the crossing (last 2–3 full
  steps averaged). Marks detail-critical / high-contrast cells — the moiré-risk set.
  Known risk: over-selects busy texture (trees, patterns); the bench measures exactly
  this.
- **`combo`** — normalized score sum.

Score → mask: threshold at target fraction → drop specks (<4 cells) → morph close →
dilate 1 cell (boundary margin, the subject-clipping contract) → re-solve threshold so
the FINAL fraction hits target. Built per (prompt, seed) at the crossing step — the
adaptivity is the value proposition vs the static rect. FoveatedTokenMerge construction
is cheap index ops, so mid-run build is free.

**Prompts — the design trap**: on the centered default prompt the static center rect is
near-optimal and derived masks can only tie. Hard prompts come from **real dataset
captions** (owner: pick difficult ones from post_image_dataset) — candidates mined for
multi-subject spread + pattern-heavy periphery: `channel6` (3girls, explicit
left/middle/right layout — center rect provably misses 2 of 3 faces) and
`ootomo 4424330` (3girls against checkered wall + step-and-repeat — stresses x0var's
texture over-selection). Baselines rendered first; subject boxes hand-annotated per
(prompt, seed).

Arms: `rect` (static center — status quo, not "oracle" on hard prompts), `rect_miss`
(null), `x0var`, `cfgdelta`, `combo`. Real 1b merge, σ_c=0.75, matched fraction 0.35,
default + hard prompts × 2 seeds. Gate: on the centered prompt derived ≈ rect (no
regression on easy mode); on hard prompts derived clearly beats static rect and
trivially beats rect_miss; overlays visually subject-following. Stretch (winner only):
fraction 0.20–0.25 matching rect-at-0.35 quality → mask intelligence converts to speed.
Scope: standalone merge line only (clean attribution); the winning source carries to
the composed stack later — P2f guarantees the blend side accepts any mask.

### Result — gate PASS, `combo` is the v1 source (2026-07-03, run
`20260703-1948-p2b`, `probe_mask_sources.py`; NB no cross-attn maps used — sources are
the two sampler-free signals; a tag-column `xattn` arm from `bench/cross_attn_drive`
machinery remains the v2 option)

Subject RMSE (aggregate, cover in parens):

| source | default (easy) | channel6 (3 subjects) | ootomo (texture trap) |
|---|---|---|---|
| `rect` static | **0.0647** (100%) | 0.1961 (68%) | 0.1329 (93%) |
| `rect_miss` | 0.1514 (30%) | 0.2537 (0%) | 0.2438 (0%) |
| `x0var` | 0.0725 (66%) | 0.1496 (91%) | 0.1322 (91%) |
| `cfgdelta` | 0.0836 (60%) | 0.1615 (89%) | **0.1132** (96%) |
| `combo` | 0.0664 (68%) | **0.1393** (95%) | 0.1250 (95%) |

- **Gate met on every axis**: on the centered default, `combo` ties static rect
  (+0.0017, at just 68% box-cover — it spends the saved budget on hands/props);
  on channel6 every derived source beats static rect, `combo` by **−29%** with 95%
  face coverage (masks visibly trace all three characters as compact blobs; all three
  face crops near-baseline, vs moiré mush for the rect's missed side faces); on ootomo
  `cfgdelta` beats rect −15%. Everything ≫ `rect_miss`. e2e ×1.33–1.40 maintained,
  achieved fraction 0.33–0.36, ring Laplacian <1 everywhere.
- **x0var's texture trap confirmed and quantified**: its score map lights up the
  checkered-wall grid (visible wall blobs in the mask); morphology limits the damage
  to a tie-with-rect instead of a win on that prompt. `cfgdelta` ignores the wall —
  prompt-awareness is what survives busy peripheries. `combo` inherits enough of both
  to be the only source that never loses to static rect on any prompt → **v1 pick**.
- Open follow-ups: (i) stretch leg — `combo` at fraction 0.20–0.25 vs rect@0.35
  (converting mask intelligence into speed); (ii) optional `xattn` source (per-tag
  cross-attn columns) if sharper semantics are wanted; (iii) production wiring of
  source+morphology into the merge path (with 1c).

## Phase 2t — trigger-region attribution: the 1a win is NOT foveal (2026-07-03, run
`20260703-1920-p2t`, `probe_trigger_masks.py`)

Follow-up question (owner): what happens if the 1a mechanism runs on the diverse P2
masks? Answer: it's the attribution check 1a never had — and it **overturns the 1a
mechanism story**. Setup: pure timing reallocation (no merge, no blend, full-res
everywhere), SEA trigger distance computed on five regions — oracle rect (= 1a's
`fovsea`, regression check), whole latent (`global`, = production `schedule="sea"`),
anti-oracle rect, multi-rect, random scatter (identical placements to the P2 run) —
each with its own auto-δ at the window schedule's refresh fraction.

- `sea_rect` reproduced 1a's fovsea exactly (subject RMSE 0.1909, refreshes [6,11,18])
  — validating both the 1a result and the `spectrum_arm` dist-fn refactor.
- **Every other region produced the same schedule and the same numbers** (0.1909–0.1910,
  ±1 fwd) — including the background-driven anti-oracle trigger. Calibrated δ spans just
  0.515–0.535 across all five regions: the *relative* SEA distance (l1rel) is spatially
  near-uniform, so WHERE the trigger looks doesn't move WHEN it fires.
- Conclusion: the 1a improvement is **generic SEA-vs-window timing** at this aggressive
  operating point (earlier mid-run refreshes → shorter tail extrapolation), not foveated
  refresh allocation. Production already ships this trigger (`schedule="sea"`) — the
  timing win needs **no mask and no new wiring**. Symmetrically, a bad mask can't hurt
  the trigger either. NB the SEA-schedule bench's earlier "reallocation REFUTED at
  matched compute" was a different (non-aggressive) operating point — SEA-vs-window is
  setting-dependent; region-independence is the robust fact here.
- Consequence for the roadmap: **masks matter only where P2 showed they do — the merge
  mask (and Phase-3 partial-recompute queries)**. Drop the fovea-δ / asymmetric-cadence
  trigger escalation; keep Phase 2b mask sources scoped to the merge line.

## Phase 2f — diverse masks on the blend mechanisms: fovblend neutrality is
mask-shape-INDEPENDENT (2026-07-03, run `20260703-1932-p2f`, `probe_blend_masks.py`)

Owner follow-up: run the diverse masks through `fovblend` / `fov1a`. This closes the
remaining 1a mechanism: fovblend's oracle-rect neutrality is the Phase-3
partial-recompute foundation, and the merge line's scatter falsification raised the
question of whether the compact-blob constraint carries over. Setup: blend σ_c=0.5
(composed knee), same four mask placements as P2/P2t; fov1a arms use the global SEA
trigger (per P2t region-independence). Neutrality measured as own-region / complement
RMSE deltas vs `spec`'s same-region values.

- **fovblend is neutral for every shape** — rect +0.0001/+0.0013 (own/comp), miss
  +0.0002/+0.0009, multi3 +0.0003/+0.0010, **scatter +0.0008/+0.0009** — all within the
  1a standard (≲0.003), both seeds, no visual artifacts (blend arms carry exactly
  spec's known damage signature and nothing else). The merge line's compact-blob
  constraint does NOT carry over: blending is per-token at emit (`final_layer` has no
  attention), so isolated cells cost nothing here. **Phase-3 recompute region is freely
  choosable at the emit level.**
- **fov1a = SEA timing win + neutral blend, for any mask**: all fov1a arms land at or
  slightly better than p2t's plain-SEA numbers (seed-40 fov1a_rect subject 0.1812 ≡
  p2t sea_rect exactly; subjects 0.179–0.201 vs spec 0.184–0.215). The negative deltas
  vs spec are the schedule effect, not the blend. Seed-41 fov1a arms took 11 fwd (one
  extra late refresh, same as p2t's global trigger) — matched ±1 standard.
- Caveat unchanged: fits still anchored to full actuals; the no-complement-update
  degradation remains Phase 3's own pre-gate.

**Net shape rules after P2/P2t/P2f:** merge mask — compact + subject-following
(placement 2.3×, scatter falsified, moiré on missed high-contrast content); trigger —
no mask at all (region-independent); blend/recompute region — any shape. Phase 2b mask
sources therefore target the merge mask only, with the blend side free to reuse
whatever mask (or none) falls out.

## Phase 3 — partial recompute + aggressive settings (designed 2026-07-03)

> The original Phase-3 sketch ("true grid coarsening — paper mechanism") is **retired**:
> 1b's whole-stack merge already delivers the token-count reduction (4096→2107 through
> all blocks, exact mean-rope, no adapter, no VAE blending); the paper mechanism's
> residual win is a low-res VAE decode at the cost of post-training, which forfeits the
> training-free selling point. Phase 3 is the partial-recompute endgame instead, plus
> the aggression sweep the 2b stretch leg left open.

Four legs, gated in order. Suggested execution: **3s and 3-pre first** (both cheap,
both reuse existing probes almost verbatim), then 3a, then 3b only if 3a leaves fovea
fidelity on the table.

**Eval protocol (all legs)** — same discipline as 2b: default + `channel6` + `ootomo`
prompts, 2 seeds, `combo` mask via the 2b source+morphology pipeline, subject-box RMSE
bracketed between oracle rect and `rect_miss`, ring Laplacian, verdict eyeball-first.
**Wall-clock matched in full-forward equivalents** — partial refreshes count ~⅓,
merged forwards ~0.47 — and the accounting must be explicit in the result envelope or
the "matched compute" claim is soft.

### 3-pre — anchoring degradation, zero surgery (the pre-gate 1a/P2f owed)

Partial recompute means the periphery never gets actual updates below σ_c — but every
blend arm so far ran with Chebyshev fits anchored to *full* actuals. Isolate exactly
that: reuse the `spectrum_arm` machinery (`probe_foveated_spectrum.py`), run full
forwards (compute unchanged), but below σ_c feed the fits **only fovea-token actuals**.

| arm | tests |
|---|---|
| `noperiph` | periphery fits frozen at crossing-time state, extrapolating the whole tail — the honest partial-recompute emulation |
| `catchup_k` (K∈{2,3}) | one full re-anchor every K refreshes — this **is** the open 1c periphery catch-up, folded in as the rescue arm |

Kill/shape: periphery degradation vs the merge line's bokeh standard + fovea
contamination check. `noperiph` fails but `catchup_2` passes → Phase 3 pays a small
re-anchor tax; both fail → 3b dies (3a survives — it re-anchors every refresh).

### 3a — `mergefresh`: the cheap competitor that sets the bar

Before any attention surgery, compose what already passed: below σ_c, Spectrum's
*refresh* forwards run through the 1b merged path (×2.15/forward → ~2× refresh cadence
at matched wall-clock). Zero new mechanism, and unlike partial recompute it re-anchors
the periphery every refresh (at pooled resolution), so it may dodge 3-pre's degradation
entirely. Claim at matched compute: beats plain aggressive `spec` on subject fidelity.
This is the arm most likely to ship.

### 3b — true partial recompute — GATED on 3a not sufficing

Fovea queries only (~35% rows) against per-block K/V where fovea rows are fresh and
periphery rows are **cached from the last full/merged forward**. Spectrum's forecast
exists only at the emit level, so periphery K/V inside blocks must come from a cache —
stale in σ (staleness partially proxied by 3-pre). Design constraints:

- Store periphery K/V **pooled** (2×2, riding the 1b merge layout): full-row caching
  across 28 blocks is ~1GB+ at 1024²; pooled drops it ~4× and matches what the merged
  path computes anyway.
- Cost ≈ ⅓ forward → ~3× refresh cadence at matched wall-clock. The claim is the one
  1a failed to deliver (P2t re-attribution): *fovea re-anchored 3× more often beats
  plain aggressive Spectrum on subject fidelity at matched compute* — and it must also
  beat `mergefresh`, not just `spec`, to justify the surgery.
- Recompute region is freely choosable per P2f (blend side accepts any shape); the
  *merge-layout* side of the cache still wants the compact combo mask.

### 3s — aggressive-settings sweep (standalone merge line, orthogonal — can run first)

All arms on the real 1b merge path with the `combo` source, hard prompts included:

The aggression knob is the **fovea fraction** — how much area runs full-res (owner
clarification 2026-07-03: the 2×2 periphery pooling stays fixed; area is what matters).
Fraction ladder on the real merge path, `combo` source, σ_c=0.75:

| arm | eff. tokens | question |
|---|---|---|
| `rect`@0.35 | 51% | the status-quo reference (1b/2b configuration) |
| `combo`@0.35 | 51% | 2b winner — carryover check |
| `combo`@{0.25, 0.20, 0.15, 0.10} | 44/40/36/33% | where does subject quality break as the fovea shrinks? Multi-subject channel6 is the stress case (3 subjects competing for the budget); the stretch gate is `combo`@low-frac ≈ `rect`@0.35 subject quality (mask intelligence → speed). |

Optional follow-ups if the ladder motivates them: 4×4 periphery pooling (merge-side
aggression — `FoveatedTokenMerge(merge_edge=4)` is already generalized), σ_c=0.85.

### Result — thin reward, knee at 0.25, cliff below 0.15 (2026-07-03, run
### `20260703-2036-p3s`, `probe_fraction_stretch.py`)

Subject RMSE (cover, e2e×) — `combo`@0.35 reproduces 2b exactly (regression ✓):

| frac | default | channel6 | ootomo |
|---|---|---|---|
| `rect`@0.35 | **0.0647** (100%, ×1.37) | 0.1961 (68%, ×1.36) | 0.1329 (93%, ×1.35) |
| 0.35 | 0.0664 (68%, ×1.38) | **0.1393** (95%, ×1.36) | **0.1250** (95%, ×1.37) |
| 0.25 | 0.0849 (58%, ×1.41) | 0.1655 (89%, ×1.39) | 0.1362 (90%, ×1.43) |
| 0.20 | 0.0986 (47%, ×1.48) | 0.1766 (86%, ×1.44) | 0.1517 (84%, ×1.47) |
| 0.15 | 0.1125 (38%, ×1.54) | 0.1926 (74%, ×1.49) | 0.1526 (82%, ×1.51) |
| 0.10 | 0.1243 (28%, ×1.54) | 0.2476 (58%, ×1.54) | 0.1832 (67%, ×1.55) |

- **No free stretch** — decay vs `combo`@0.35 is strictly monotone on every prompt;
  each fraction step costs ~0.01–0.03 subject RMSE. Vs the *static rect* the stretch
  partially works (combo@0.25 still −16% on channel6, tie on ootomo — mask
  intelligence spends the saved area well), but the 2b stretch hope "0.20–0.25
  matching rect-at-0.35 quality *on every prompt*" fails on default, where the rect
  is the oracle.
- **The speed reward is thin and bounded**: ×1.37 → ×1.48 at frac 0.20, ×1.55 floor —
  effective tokens f + (1−f)/4 cap the whole ladder at ~×1.6 even as f→0. The knob
  works but pays little; the next real speed lever is `merge_edge=4` pooling
  (eff → f + (1−f)/16), not a smaller fovea.
- **Cliff below 0.15 on multi-subject**: channel6 cover 95→86→74→58%; at 0.10 the
  mask starts dropping faces (subject 0.2476, +78% vs 0.35). Eyeball at 0.20: all
  three faces stay crisp (mechanism holds) but torsos/hands/bags blur to mush — on
  subject-dense prompts the periphery IS the characters, and the "anime DOF" contract
  stops reading as intentional. At 0.10 face edges start smearing too.
- **Ship setting**: default frac **0.35**; expose fraction as the aggressiveness knob
  with a practical floor of **0.25** (0.20 acceptable for centered single-subject
  content); ≤0.15 falsified for multi-subject.

### Phase-3 roll-up (2026-07-03)

3-pre: anchoring degrades at sane cadence, 1c catch-up (k=2) restores exact
neutrality — recorded, moot. 3a/3b: the composed foveated-Spectrum line is CLOSED
(no headroom at sane schedules, baseline collapse at aggressive ones; partial
recompute's ceiling ties spec at both points — do not build). 3s: fraction ladder
knee 0.25, cliff ≤0.15, speed ceiling ~×1.6. **What ships from the foveated bench:
the standalone deferred-foveated merge — 1b mechanism + 2b combo mask, σ_c=0.75,
frac 0.35 (knob floor 0.25), ×1.37 e2e — plus production Spectrum's existing
`schedule="sea"` as the separate quality-side win (P2t). Next step: production
wiring (2b follow-up iii).**

### Result — 3-pre PASS, 3b DEAD, mergefresh free discount; and the aggressive
### Spectrum point itself COLLAPSES on hard prompts (2026-07-03, run
### `20260703-2019-p3`, `probe_phase3.py`)

Aggressive point (window 3 / flex 3 / warmup 4 = the 0b/1a/2t setting), combo mask
built endogenously at the crossing, σ_c=0.5 (composed knee; `mergeall75` at 0.75),
default + channel6 + ootomo × 2 seeds. One design upgrade over the sketch above: the
3-pre arms (`pr_all`/`pr_catchup2`) run at partial-recompute's REAL cadence (every
below-σ_c step is a fovea-anchored actual, frozen-periphery fits), which makes
`pr_all`'s subject RMSE double as the **3b quality ceiling** at its emulated ~⅓ cost.
Regression check: `spec` on default = 0.1983 fovea RMSE ≡ 1a exactly.

- **3-pre gate PASS** — `pr_all` complement RMSE Δ+0.0017 vs `spec` (aggregate;
  neutrality standard ≲0.003), `pr_catchup2` Δ+0.0008. Periphery extrapolating from a
  crossing-frozen fit degrades nothing measurable even at every-step cadence; the 1c
  catch-up rescue is unnecessary. `fovblend` neutrality extends to the no-anchor regime.
- **3b is DEAD at this operating point** — the ceiling ties the cheap competitor:
  `pr_all` subject 0.3250 @ ~9.0 emulated FFE vs `mergeall` realized 0.3252 @ 9.2
  measured FFE. Zero headroom for fovea-query/cached-K/V attention surgery. Do not
  build it.
- **`mergefresh` is a free discount and the only 3a gate met**: subject RMSE ties spec
  on every prompt (Δ ≤ 0.002) at **FFE 8.3 vs 10.0** (e2e 4.4s vs 5.0s) — running
  post-crossing refreshes through the 1b merged stack costs nothing measurable.
  `mergeall` (every below-step a merged actual) also only *ties* spec at 9.2 FFE — the
  "beats spec at matched compute" gate is NOT met, because post-crossing effort is not
  where spec's error lives (see next point).
- **The load-bearing eyeball finding: plain aggressive Spectrum collapses on hard
  prompts.** channel6 `spec` (no foveation anywhere) renders inverted hair colors, red
  corrupted faces, dissolved twintails, striped bodies — subject RMSE 0.41 vs 0.20 on
  default. Every arm inherits this: the damage is written in the PRE-crossing
  σ∈[0.5,0.85] forecast steps, which is exactly why all σ_c=0.5 arms tie. `mergeall75`'s
  numeric "wins" (−8% channel6, −5% default) are a partial *rescue* — it replaces
  forecast steps 14–21 with real merged forwards, visibly less corrupted faces — NOT a
  quality gain over a sane baseline. RMSE certified change, not improvement, exactly as
  the no-quality-metric invariant warns. The whole 0b/1a/2t/2f line calibrated its
  operating point on the default prompt only, where spec's damage is moderate and
  localized.
- Consequence: quality-allocation claims at the aggressive point are malformed on hard
  content. Re-run at the production Spectrum point → next section.

### Result — production point: the composed foveated-Spectrum line CLOSES
### (2026-07-03, run `20260703-2028-p3prod` — window 2 / flex 0.25 / warmup 6 =
### `spectrum_denoise` defaults, ~16/28 forwards; same arms/prompts/σ_c)

- **Sanity restored, headroom gone.** `spec` at the production point renders even
  channel6 essentially indistinguishable from `full` (subject 0.096–0.14 vs the
  aggressive point's 0.20–0.41 collapse; eyeball: near-identical). The 0b premise —
  "forecast error concentrates in the fovea, reallocate refreshes spatially" — only
  exists at the aggressive point, where the baseline is unusable on hard prompts
  anyway. Where the baseline is usable there is nothing to reallocate.
- **3-pre honestly FAILS at sane cadence — and `catchup2` rescues it exactly.**
  `pr_all` complement RMSE degrades Δ+0.017/+0.032/+0.017 (default/channel6/ootomo)
  vs `spec` — far over the 0.003 standard; the aggressive-point "PASS" was masked by
  spec's own periphery damage. `pr_catchup2` (full re-anchor every 2nd below-step)
  restores exact neutrality (Δ ≤ 0.0002 everywhere). So frozen-periphery
  extrapolation does degrade over a real post-crossing tail; any partial-recompute
  build needs the 1c catch-up cadence. Recorded for posterity — moot given the next
  point.
- **3b stays dead at both operating points**: `pr_all`'s subject ceiling ties `spec`
  (0.0970 vs 0.0960 default; 0.1179 vs 0.1148 channel6) despite every-step fovea
  actuals below σ_c — full-res-fovea re-anchoring buys nothing a sane schedule
  doesn't already have.
- **Merge arms now cost quality**: `mergefresh`/`mergeall` subject +0.006…+0.020 vs
  spec at only −5–7% FFE (e2e 7.6s vs 7.9–8.2s), and the eyeball shows the real
  contract problem — at frac 0.35 on 3-subject prompts the periphery contains legs /
  hair tails, which the pooled readout washes to mush. Visible damage for negligible
  savings.

**Net Phase-3 composed-line verdict: CLOSED.** The foveated-Spectrum reframe (0b/1a)
dies by squeeze: aggressive schedule → baseline collapses on hard content (foveation
can only partially rescue, not fix); production schedule → no headroom, merge
discount negligible, periphery cost visible. Partial recompute (3b) is dead at both
points; do not build. What survives Phase 3: the **standalone deferred-merge line**
(1b mechanism + 2b combo mask) as its own ×1.37 speed knob — its aggressive-fraction
ladder is Phase 3s (next), and it remains the only inference-stack candidate from
this bench.
