# Foveated denoise — selective effort in the detail phase

> **Canonical roadmap: `docs/proposal/foveated_denoise.md`** (written after Phase 0
> passed). The Phase-1 cost sketch below reshaped the plan: standalone merge speed is
> marginal, so the proposal's next gate is Phase 0b (compose with Spectrum — spend its
> refresh budget spatially) and Phases 1–3 branch from that. The phases listed further
> down this file are the original pre-arithmetic sketch, kept for history.

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

## Phase 1 — real savings: masked token merge inside blocks (ToMe-style) — GATED on P0

Merge 2×2 periphery token groups before each block's attn/MLP, broadcast after; latent
stays full-res end-to-end (no VAE blending, no RoPE surgery — merged groups sit at their
mean position's RoPE only inside attention). Measure actual wall-clock vs quality at the
P0-surviving σ_c. Composability with Spectrum (block skipping — orthogonal) is the
interesting aggregate number.

## Phase 2 — mask sources — GATED on P0

P0 uses a static center rect. Real masks: the FreeText Stage-1 localizer (endogenous I2T
cross-attn "where is concept X", read during the full-res steps) ∪ high local-x̂₀-variance
regions. Only worth building once P0/P1 show the mechanism is sound.

## Phase 3 — true grid coarsening (paper mechanism, time-gated) — GATED on P1

Actual token-count reduction post-σ_c (phase-aligned mixed-res RoPE, probably a
post-trained adapter, low-res VAE decode + blend). Only if P1's savings are worth less
than this buys.
