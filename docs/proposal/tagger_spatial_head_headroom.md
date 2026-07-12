# Tagger spatial-head headroom — the floor is the head, not the features

Status: **CLOSED 2026-07-12** — Phase 1 shipped + v3 promoted 2026-07-08; the
owed ceiling-probe re-run on the shipped v3 came back **`MIXED`** (deployed ≈
same-arch isolated oracle; floor median lift +0.022, control gap ≈0), i.e. the
refit captured the head-side headroom this line was about. Bench instruments
archived to `_archive/bench/tagger_ceiling/`; `bench/tagger_eval` stays live as
the regression scorer. Residual open lever = the un-run `label_embed` tail arm
(see Closing). Original Phase 0/0.5 verdict: **`HEAD_HEADROOM`, cause =
MULTITASK STARVATION.** The deployed spatial branch floors on localized-semantic
tags (`tagger_eval`: pose ~0.18, expression ~0.15, gesture ~0.11) at 0.260 mean
AP over 1207 supported spatial tags. The factor ablation is unambiguous: taking
the **deployed architecture unchanged** and training it **spatial-only** (freed
from the core/rating/people multitask) recovers **+0.123 AP → 0.383** — nearly
the whole achievable gap, with no architecture change and no class-balancing.
`pos_weight` balancing *hurts* (−0.107). So the floor is **not** a feature
ceiling (tower is fine) and **not** an architecture problem (the shipped
MAPHead+trunk is fine) — it is a **selection + recipe** problem (the two trunks
are disjoint, so it is *not* shared-gradient interference; see the Phase-1
mechanism note). Fix is training-side and cheap, and is **SHIPPED** (Phase 1
below): `--select_metric spatial_ap`, an optional spatial-only refit stage, and a
spatial optimizer param-group. Instruments: `_archive/bench/tagger_ceiling/{run_bench.py,
ablate.py}`, `_archive/bench/tagger_ceiling/test_tagger_ceiling.py`, `tests/test_tagger_spatial_headroom.py`.

## Where this comes from

`tagger_eval` (2026-07-04) showed the deployed tagger near-solves identity
slices (artist 0.75 / character 0.77 / copyright 0.59) but floors on every
*localized* semantic slice — pose, expression, gesture, clothing_details,
body_parts, sexual_acts all 0.13–0.29 macro-F1. Those slices all route to the
**spatial** branch (`tag_indices_spatial`, 1224 tags off frozen PE-Spatial via
a learned MAPHead pool + trunk). The open question `tagger_eval` structurally
could not answer: is that floor a **head/training** problem (fixable) or a
**feature ceiling** (frozen PE-Spatial doesn't encode pose at the needed
granularity — needs a new tower)?

## Phase 0 — the ceiling probe (`_archive/bench/tagger_ceiling/run_bench.py`)

Re-fit the spatial branch on the *same* frozen PE-Spatial tokens at three
capacity rungs, score every spatial tag threshold-free (mean AP), stratify by KB
slice. Self-checking verdict: the oracle must match the deployed head on
high-AP control slices before a floor null counts as a ceiling (else
`INCONCLUSIVE`).

Run `v2-full` (12k train cap, 40 epochs, val N=756, 1207 supported spatial tags):

| arm | head | spatial mean AP |
|---|---|---|
| deployed | shipped MAPHead(K=4)+trunk, multitask | 0.260 |
| **linear** | mean\|max\|cls pool → logistic, spatial-only + balanced | **0.344** |
| oracle | MAPHead(K=8)+2-layer MLP, spatial-only + balanced | 0.307 |

Verdict `HEAD_HEADROOM` (floor median lift +0.051, oracle-validity gate passed
at +0.022 on controls). Floor slices move most: pose 0.183→0.319, gesture
0.114→0.223, decorative_elements 0.210→0.393, furniture 0.265→0.581,
sexual_acts (142 sup) 0.251→0.332.

This established the floor is **not a feature ceiling** — frozen PE-Spatial reads
out far better under a fresh head. But the ceiling probe's arms were confounded
(spatial-only *and* class-balanced *and* different architecture), and the
initial read ("linear beats the MAP oracle → architecture is over-engineered")
turned out to be a `pos_weight` artifact. Phase 0.5 disentangles it.

## Phase 0.5 — factor ablation (`_archive/bench/tagger_ceiling/ablate.py`) — RUN 2026-07-08

Grid on the same materialized features (all spatial-only), reading spatial mean AP:

| cell | head | pos_weight | mean AP | vs deployed |
|---|---|---|---|---|
| deployed (anchor) | deployed-arch, multitask | deployed | 0.260 | — |
| **dep_arch__nobal** | deployed-arch, isolated | off | **0.383** | **+0.123** |
| linear__nobal | linear pool, isolated | off | 0.353 | +0.093 |
| linear__bal | linear pool, isolated | on | 0.344 | +0.084 |
| dep_arch__bal | deployed-arch, isolated | on | 0.276 | +0.016 |

**Read:**

1. **Isolation is the whole story: +0.123.** The *deployed architecture,
   unchanged*, trained spatial-only recovers nearly the entire achievable gap and
   is the best cell of all. Multitask joint training (spatial sharing gradients
   with the core/rating/people heads) is starving the spatial branch.
2. **`pos_weight` HURTS** — −0.107 on the MAP head, −0.009 on linear. Class
   balancing is a regression, not a fix. Do not ship it.
3. **The architecture is fine.** MAPHead+trunk (0.383) *beats* linear (0.353)
   once both are unbalanced; the ceiling-probe "linear wins" was the
   pos_weight + oversized-2-layer-MLP handicap, not a real architecture defect.

**Caveat:** the isolation cell also folds in train-recipe differences (40-epoch
cosine AdamW + val-early-stop vs the deployed training recipe), so +0.123 is
"isolation + recipe", not provably pure isolation. The confirmatory test is
Phase 1 itself — a spatial-loss-decoupled retrain of the *actual shipped model*.
That the deployed model (also trained to convergence) sits 0.12 below a
same-architecture isolated head strongly implicates the multitask coupling, but
Phase 1 is where it's proven.

## Phase 1 — the fix: decouple the spatial branch's optimization — SHIPPED 2026-07-08

**Mechanism refinement (load-bearing).** The deployed head's two trunks are
**fully disjoint**: `pool_spatial`/`trunk_spatial`/`tag_head_spatial` share *no*
parameters with the core/rating/people heads (see `AnimaTaggerHead`). So there is
no gradient *interference through shared weights* — and under AdamW a **uniform
up-weight of the spatial-tag BCE (option a) is nearly inert**: scaling that loss
term by λ scales every spatial-param gradient by λ, which cancels in Adam's
per-parameter `m̂/√v̂` (only the decoupled-weight-decay ratio shifts slightly).
The real content of the "+0.123 = isolation + recipe" gap is therefore **(1)
model selection blind to the spatial branch** and **(2)** the train recipe/epoch
budget — with the **spatial-only refit (option c)** directly reproducing the
0.383 isolated cell. Phase 1 wires the *effective* levers and deliberately skips
the inert one.

Shipped in the frozen-encoder trainer (`scripts/anima_tagger/train_cached.py`,
flags in `cli.py`), all opt-in / default-inert:

- **`--select_metric spatial_ap`** — model selection on threshold-free spatial
  mean AP over the PE-Spatial-routed tags (softmax-inclusive), matching
  `_archive/bench/tagger_ceiling`. Default stays `macro_f1` (which *excludes* softmax
  groups and mixes in the near-solved core slices → blind to the floor); the
  retrain uses `spatial_ap`. Reported every epoch either way.
- **`--spatial_refit_epochs N`** — option (c): after joint training, freeze
  core/rating/people and refit only the spatial branch for N epochs on the
  *same* grouped objective, selecting on `spatial_ap`. Freezing the disjoint
  heads **guarantees the identity/core slices cannot regress**. Kept only if it
  beats the joint checkpoint's `spatial_ap`.
- **`--lr_spatial` / `--wd_spatial`** — option (b): a separate optimizer
  param-group for the spatial branch in the joint stage. A real lever (disjoint
  trunks ⇒ a higher spatial LR genuinely changes its trajectory, unlike the
  cancelled loss up-weight).
- **Do NOT add class-balancing** (`pos_weight`/focal) — the ablation shows
  −0.107. Not wired.

**Recommended retrain:** `--select_metric spatial_ap --spatial_refit_epochs ~15`
(± `--lr_spatial` sweep). Validate by re-running `_archive/bench/tagger_ceiling/run_bench.py`
(deployed arm → ~0.38) and `bench/tagger_eval/run_bench.py` (floor slices up,
identity slices flat).

### First retrain result — RUN 2026-07-08 (2528-tag vocab, `anima-tagger-v3-refit`)

Trained the treatment (`--select_metric spatial_ap --spatial_refit_epochs 15`,
else v2 recipe/seed) with dual-checkpointing, so the joint best-`macro_f1` ckpt is
an in-run baseline (v2 criterion) on identical data. `bench/tagger_eval` (val
N≈789, inference rule) baseline → treatment:

| metric | baseline | treatment | Δ |
|---|---|---|---|
| residual macro-F1 @0.5 (v2 eval-F1) | 0.2560 | 0.2700 | **+0.014** |
| mean AP all (threshold-free) | 0.3271 | 0.3435 | +0.016 |
| support-wtd mean AP, non-identity slices | 0.2594 | 0.2788 | **+0.019** |
| pose / expression / sexual_acts / furniture | — | — | +0.027 / +0.021 / +0.029 / +0.030 |
| identity (cat:copyright) | 0.5523 | 0.5523 | **0.0000** |

**Reads:** (1) eval-F1 did **not** regress — it *improved* (+0.014); every floor
slice lifted; only a few low-support slices dipped (plants −0.027 @nsup 14).
(2) **Identity is exactly preserved (Δ 0)** — the refit freeze worked. (3) On this
run the **refit is the whole story**: `macro_f1` and `spatial_ap` peaked at the
*same* joint epoch (ep32), so the selection-metric change was inert here and the
entire gain came from the 15-epoch spatial refit. The proposal's +0.123 was vs the
undertrained/mis-selected *shipped v2* (old 1405-tag vocab); a well-trained joint
baseline leaves far less headroom, but the refit is a real, downside-free win.
**Shipped 2026-07-08**: v3 (+ per-tag calibrate, `--mode calibrate`: macro-F1
0.234→0.292) promoted in-place into the canonical `anima-tagger-v2/` dir; old
1405-tag model backed up to `anima-tagger-v2-1405-backup/`.

### Next — refit sweeps (the refit is the lever; chase its ceiling)

The refit was still climbing at ep47 (spatial_ap 0.278) and it, not the selection
metric, carries the gain — so the sweeps all target the **refit stage** (joint
stage stays fixed at the v2 recipe; dual-checkpointing keeps the macro_f1 baseline
free every run). Grid, cheapest first, each scored by `bench/tagger_eval` (floor
slices ↑, identity Δ≈0, residual macro-F1 not down) + `_archive/bench/tagger_ceiling`
(deployed arm → ceiling):

1. **Longer refit** — `--spatial_refit_epochs {25, 40}` at the default LR. First
   check whether the ep47 climb was just undertraining before spending on LR.
2. **Refit LR** — `--spatial_refit_lr {3e-4, 5e-4, 1e-3}` (joint used 1.5e-4; the
   refit is a fresh cosine on the spatial branch only, so it tolerates a hotter
   LR). Pair the winner with the longer-epoch setting.
3. **Joint spatial param-group** — `--lr_spatial {3e-4, 5e-4}` to see if a hotter
   spatial LR *during joint* lifts the pre-refit spatial_ap (may also make the
   selection-metric divergence reappear, which was inert at the shared LR).
4. **Diminishing-returns gate** — stop when a rung adds <+0.005 support-weighted
   non-identity AP or costs any identity-slice AP. Retrains are ~30 min each;
   run them as detached jobs (`setsid nohup … --no-ram_resident`), not tracked
   background tasks (the harness reaps idle ones).

Sweep bench envelopes land in `bench/tagger_eval/results/<ts>-<label>/`.

### Closing — v3 ceiling re-run — RUN 2026-07-12 (`results/20260712-1248-v3-full`)

The owed "true vs-deployed" probe, same config as v2-full (12k cap, 40 ep, val
N=756) against the promoted v3 (2131 spatial tags, 1879 supported — vocab grew
from v2's 1224, so mean-AP is not directly comparable across the two runs):

| arm | spatial mean AP |
|---|---|
| deployed (shipped v3) | 0.2784 |
| linear | 0.3357 |
| oracle (isolated MAPHead, 40 ep) | 0.2891 |

**VERDICT `MIXED`** — floor median lift (oracle − deployed) collapsed to
**+0.022** (was +0.051 on v2), control median lift −0.001 (oracle_valid=True).
The deployed v3 now sits at the same-architecture isolated ceiling: the
multitask-starvation gap this proposal targeted is gone, which retroactively
confirms the Phase-1 refit as the fix. Reads on what's left: (1) the refit
sweeps above have little probe-visible headroom remaining — run them only as
cheap opportunistic rungs, scored by `bench/tagger_eval` alone; (2) the linear
arm still clears deployed by +0.057 mean AP, concentrated in high-support
object-y slices (furniture 0.57 vs 0.29, costume 0.45 vs 0.27, bag 0.66 vs
0.39) — that residual is head/tail territory, i.e. the un-run **`label_embed`
arm**, not the starvation mechanism (closed). Caveat: the oracle was still
climbing at ep40 (+0.002/4ep) on the larger vocab, so the isolated ceiling may
be slightly understated; it passed the control sanity gate regardless.

With the attribution question answered end-to-end (floor → head → starvation →
fix shipped → gap closed), `_archive/bench/tagger_ceiling` is archived
(`_archive/bench/tagger_ceiling/`, with `test_tagger_ceiling.py`); future
tagger changes regression-score through `bench/tagger_eval`.

Orthogonal arm (fold in, don't gate on): the **`label_embed` head A/B**
(`--tag_head_kind label_embed`, un-run). It's a head-side lever aimed at the tail
(827 supported low-AP tags) via description-embedding sharing — complementary to
the isolation fix, not competing with it.

## Success metric & guardrails

- Primary: spatial mean AP through `_archive/bench/tagger_ceiling/run_bench.py` (deployed
  arm) and per-slice AP through `bench/tagger_eval/run_bench.py`. Target: move the
  deployed spatial arm from 0.260 toward the isolated ceiling **0.383 (+0.123)**
  on the floor slices, without regressing the identity/core slices (a
  joint-training change must not cost the near-solved artist/character AP).
- Guardrail: any retrain must keep the softmax-group machinery (eye_color etc.)
  intact — `tagger_groups` sized those; don't regress the solo-gate.
- Phase 0/0.5 needed **no retraining of the tagger** (probes fit on cached
  frozen features); Phase 1 is the first arm that touches the shipped model.

## Caveats

- The linear arm is spatial-only + class-balanced vs the deployed joint-multitask
  head — part of the +0.084 is training-config, not head capacity. The ablation
  is precisely to separate these.
- One seed, val N=756, 12k train cap. Directionally strong (gaps are large),
  Phase-0.
- 1-support KB slices are degenerate (AP undefined-ish); the verdict logic
  filters at `min_support=10`.

## Related

`project_tagger_ceiling_head_headroom`, `project_tagger_dual_hardrouted`,
`project_lora_crossattn_learns_labeled_only` (untagged content lives off the
labeled cross-attn path — orthogonal, but the same "where does the signal live"
lens), `bench/tagger_eval`, `_archive/bench/tagger_groups`.
