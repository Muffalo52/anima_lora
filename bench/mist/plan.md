# MIST bench — Phase-0/1 plan

**Paper:** Peng et al., *MIST: Moment-Aligned Invariant Stability Transform for
Robust Flow Matching* (ICML 2026; repo-root `18927_MIST_*.pdf`).
**Core:** `library/inference/corrections/mist_core.py` (`MISTState.combine`, a pure-
compute sibling of `smc_cfg.py` — drop-in at the velocity-combine slot,
`generation.py:984`). **Not wired into `generation.py` until this bench clears.**

## The question

Should MIST **replace** SMC-CFG (and possibly FSG) in the Spectrum node? MIST and
SMC occupy the *same* combine slot (mutually exclusive). FSG is a *different* hook
(pre-step latent calibration) so MIST can replace OR stack with it. So this is a
head-to-head, not a MIST-vs-CFG demo.

## Arms (`render_compare.py`, same noise/prompt/seed/sampler)

`baseline` (plain CFG) · `smc` (incumbent) · `cfg++` (shipped reweight) ·
`fsg/cfg` (shipped combo) · `mist` (ST+IA) · `fsg+mist` (orthogonal stack).
`--ablate` adds `mist-IA` (IA only) and `mist-ST` (ST only) — reproduces the
paper's Tables 2/9 split.

## Priors to falsify (from stored Anima findings)

- **TD likely inert.** Anima's guidance delta already decays front-loaded
  (`crossattn_drive_frontloaded`: peaks σ=1 ~0.27 → ~0.02 floor below σ0.85), so
  the monotone-norm clamp rarely fires. Expect `mist-IA ≈ mist`.
- **IA may be redundant with CFG++.** Variance renorm overlaps with what CFG++
  already does; several CFG-space levers have come back inert/net-negative here.
- **DC overlaps with shipped DAVE** (per-block DC-mean subtraction).

## GO gate

A real win = `mist` beats BOTH `smc` and `cfg++` on prompt adherence **without**
the tone collapse (sat/contrast not crushed), AND `fsg+mist ≈ mist` (FSG
redundant → can drop it from the node). Tone numbers (sat/RMS-contrast) are
first-order but certify *change* not *quality* — the contact sheet is the gate.
If even `mist-IA` ⪅ `cfg++`, drop it.

MIST's headline regime is high w (paper sweeps w∈[2,20]); run `--guidance 7` and
`--guidance 10` too, not just Anima's usual low/moderate CFG.

## Next (if GO)

CMMD + PE/CLIP scoring (live val signal) on a prompt set; then wire `--mist` into
`generation.py` (sibling to `--smc_cfg`, mutually exclusive with it and `--cfgpp`)
+ `_combine_cfg` in `spectrum.py`; vendor `mist_core.py` into the Spectrum node
via `make vendor-sync`. Invariant test: all-toggles-off ≡ plain CFG (bit-exact);
4D/5D shape parity; dim-2 singleton preserved.
