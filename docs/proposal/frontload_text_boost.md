# Front-loaded text-drive boost — amplify the text pathway in the first 2–3 steps only

Status: **PROPOSED 2026-07-06 (Phase 0 not run).**

Two banked findings compose into a lever nobody has pulled, and it's a
one-afternoon falsification either way.

## Premise sources

- Cross-attn text drive is **front-loaded**: peaks at σ = 1, falls to a ~0.02
  floor below σ ≈ 0.85 (`docs/findings/crossattn_self_attn_dominance.md`,
  `bench/cross_attn_drive/`). The reweight lever only has headroom in the first
  ~2–3 steps of the 28-step schedule.
- LoRAs learn **labeled tags only** — influence Δ tracks training-label
  frequency; weak/rare tags underdrive precisely in the window where drive
  exists at all.
- **Guard compliance**: the shelved-explorations roll-up bans *low-σ* tag
  cross-attn levers — banked because there is no drive left down there. This
  proposal is the *opposite end* of the same finding (σ ≳ 0.85), which is
  exactly where that finding says headroom lives.
- Honest prior against arm (b): the MAP-evolution probes found self-attn + MLP
  dominate the block residual **at every σ** — the cross-attn branch is a small
  fraction of the update even at σ = 1. Boosting it may be inert. That is the
  falsifiable question, not a reason to skip the experiment.
- Composition note: FSG's shipped band is [0.59, 0.75] — no overlap with
  σ ≥ 0.85, so the two can't fight. SMC-CFG overlap checked in Phase 1.

## The idea — two arms, both inference-only

Sampler-boundary plug-ins in `library/inference/corrections/` (same shape as
SMC-CFG/CNS: compose with any checkpoint, no training). Standard 5D caveat
applies (latents at the sampler boundary are `(B, C, 1, H, W)`, dim-2 singleton).

- **Arm (a) — σ-gated CFG schedule** (`--cfg_hi 7 --cfg_hi_band 0.85`):
  guidance scale g(σ) = `cfg_hi` for σ ≥ band, the normal `--cfg` below. ~20
  lines; no such schedule exists today (checked 2026-07-06). This boosts the
  whole cond−uncond delta, so it drags style/layout along with tags — cheap but
  blunt.
- **Arm (b) — cross-attn branch gain** (`--xattn_boost 1.5 --xattn_boost_band
  0.85`): block-level hook (DAVE-style `Block.forward` hook, not an override —
  same invariant as the custom nodes) scaling the cross-attn residual
  contribution by λ for σ ≥ band. Surgical to the text pathway; this is the arm
  the front-loading finding actually predicts something about.

Known failure mode to watch for arm (a): high CFG at high σ is the burn/border
regime — the border-artifact reproducer prompt goes in the Phase-0 grid
deliberately.

## Phase 0 — grids, matched everything

Fixed prompt set: ~12 prompts built around known **weak tags** (selected with the
existing `bench/cross_attn_drive/tag_influence.py` machinery — used for
*selection only*, it is not a gate) + the border reproducer + 4 ordinary prompts
as regressions. Render baseline / arm (a) at 2 strengths / arm (b) at 2
strengths, matched seed and NFE, standard 28-step protocol. Judgment is
eyeball on side-by-side grids:

- **G1 (effect)**: at least one arm visibly improves weak-tag adherence on a
  majority of the weak-tag prompts.
- **G2 (no-harm)**: ordinary prompts unchanged; no burn/border regression on the
  reproducer at the strength that passes G1.

Secondary (free, non-gating): rerun `tag_influence.py` with the boost active —
the influence deltas should move if the mechanism story is right; a G1 pass with
flat influence numbers means the improvement came from somewhere else and the
mechanism claim gets rewritten before Phase 1.

## Phase 1 (only on a G1+G2 pass)

- Compose-flag plumbing so it stacks with `SPECTRUM=1`/`MOD=1` like every other
  correction; SMC-CFG interaction test.
- Turbo check: the 4-step student lives entirely at σ ∈ {1.0, 0.9, 0.75, 0.5} —
  the band covers half its steps, and turbo runs CFG 1.0, so only arm (b)
  applies there. Separate small grid.
- Node export decision (Spectrum-KSampler repo) after the repo-side version has
  survived real use.

## Kill criteria

- Neither arm moves weak-tag adherence at any tested strength → bank "the
  front-loaded drive window is not exploitable at inference" next to the low-σ
  guard (the two together close the σ axis for inference-side tag levers;
  whatever remains is training-side) and close.
- Arm (a) can't pass G2 at any strength that passes G1 → close (a) alone, (b)
  stands on its own result.
- Arm (b) inert while (a) works → the self-attn-dominance prior wins; bank that
  confirmation explicitly (it upgrades the MAP-evolution finding from
  "descriptive" to "causally tested at inference").
