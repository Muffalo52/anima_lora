# Front-loaded cross-attn boost (`--xattn_boost`)

Training-free weak-tag adherence lever: scale every block's **cross-attn
residual** by λ on the **conditional forward only**, gated to **σ ≥ band**
(default 0.85) — the plan-writing window where cross-attn text drive exists at
all (peaks at σ = 1, ~0.02 floor below σ ≈ 0.85 —
`docs/findings/crossattn_self_attn_dominance.md`). Amplify the text voice while
the plan is being written and the plan changes; self-attn + MLP then render the
corrected plan in the normal style.

Origin: `docs/proposal/frontload_text_boost.md` arm (b), Phase-0 G1+G2 PASS
(`bench/frontload_text_boost/report.md`). The σ-gated **CFG** arm (a) of the
same proposal is CLOSED — style collapse at every strength that does anything;
do not re-propose plain high-σ CFG-up.

## Usage

```bash
make test XATTN_BOOST=2 PROMPT="..."          # band defaults to 0.85
make test XATTN_BOOST=1.5 XATTN_BOOST_BAND=0.95   # tighter window (~4 steps)
python inference.py ... --xattn_boost 2.0 --xattn_boost_band 0.85
```

`1.0` = off (exact identity — pinned by `tests/test_xattn_gain.py`). Phase-0
strengths: 2.0 strictly dominated 1.5 on adherence (fixed 5/7
baseline-failed watch tags vs cfg_hi's 2/7-with-style-collapse), at a mild
global desaturation cost (−8% sat at 2.0, direction *opposite* to burn).
λ ∈ {2.5, 3} and bands other than 0.85 are unmapped.

## Mechanism

- Per-block non-persistent buffer `Block._xattn_gain`
  (`library/anima/models.py`), read inside the compiled `_forward` — retuned
  per step via `fill_()` with **no recompile** (same pattern as the
  mod-guidance buffers). Multiplies the cross-attn residual *after* the
  `gate_cross_attn` AdaLN gate, before the residual add.
- `library/inference/adapters.py::set_xattn_gain` writes all blocks; every
  denoise loop sets it **before the cond forward** and resets to 1.0 **before
  the uncond forward** (so the boost lands entirely in the guidance delta and
  none of it is spent amplifying the negative prompt), plus a `finally` reset
  so no gain leaks across generations.
- Threaded to loop runners via `SamplerSideChannels.xattn_boost` /
  `.xattn_boost_band` (`library/inference/sampler_context.py`).

## What it does / doesn't do

- **Moves relations & bindings between things the model knows** (multi-subject
  role↔attribute bindings, object relations, scene adherence). It does NOT
  conjure unknown concepts — a multiplicative lever on ~zero signal is ~zero
  (theremin/upside-down-umbrella class prompts stay failed). Same law as
  "LoRA cross-attn learns labeled tags only", observed at inference.
- **Amplifies ALL caption tags, wanted or not** — framing priors (`cropped`,
  an artist's habitual close-up crop) ride the boost, and with style/artist
  tokens in the caption the content win attenuates (the gain splits across
  tokens). The token-selective arm (c) of the proposal targets exactly this.
- Under **CFG 1.0** (turbo student) there is no uncond pass; the single
  forward is boosted.

## Compose matrix

| Stack | Status |
|---|---|
| `SPECTRUM=1` | Wired. Real cond forwards boosted; forecast steps extrapolate from boosted cond features (consistent — and warmup covers the earliest in-band steps with actual forwards). |
| `--spd` | Wired (gate reads the re-spaced per-step σ). |
| `--fovea_sigma_c` | Wired; band sits above any sane σ_c so boosted steps are full-grid. |
| `--smc_cfg` / `--cfgpp` | Compose mechanically (they change the cond/uncond combine; the boost changes the cond forward). Interaction grid: Phase 1. |
| `MOD=1` / `--dave` / `--cns` / FSG | Orthogonal pathways (AdaLN modulation / DC attenuation / noise recoloring / mid-σ latent calibration — FSG's band [0.59, 0.75] doesn't overlap σ ≥ 0.85). FSG's internal forwards run at identity. |
| Tiled diffusion | Wired (per-tile cond passes boosted). |

## Files

- `library/anima/models.py` — `_xattn_gain` buffer + `_forward` read
- `library/inference/adapters.py` — `set_xattn_gain`
- `library/inference/args.py` — `--xattn_boost`, `--xattn_boost_band`
- `library/inference/generation.py` — inline + tiled loop wiring
- `networks/spectrum.py`, `networks/spd.py`, `networks/foveated.py` — runners
- `tests/test_xattn_gain.py` — identity/scaling/reset invariants
- `bench/frontload_text_boost/` — Phase-0/1 grids + report
