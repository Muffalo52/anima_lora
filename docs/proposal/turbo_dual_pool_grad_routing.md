# Dual-pool gradient routing — parameter-level div/DMD separation for DP-DMD

Status: **WIRED — Phase 0 pending first run.** The config-guarded routing,
exact-concat save, resume, and invariant tests are implemented
(`networks/methods/turbo_dmd.py`, `scripts/distill_turbo/{config,setup,distill,resume}.py`,
`tests/test_turbo_dual_pool.py`); the rank-matched A/B is staged under
`bench/turbo/dual_pool/` (`arm_A_dual_32_32.toml` vs `arm_B_single_r64.toml`,
symmetric 32/32 on the CDM+div0.1 superturbo recipe). Awaiting the first
rendered NFE grids to score the gate below.

## The idea

Split the turbo student into **two plain-LoRA pools on the same frozen DiT,
both active on every forward**:

- **Pool A (diversity)** — receives *only* the step-0 gradient: the
  first-step diversity MSE against the teacher anchor (`div_loss`), plus the
  soft-rank caption term that rides the same backward.
- **Pool B (quality)** — receives *only* the refinement-side gradients: the
  DMD reverse-KL, the GAN generator term, and L_CDM when enabled.

The student's velocity field is always `base + ΔW_A + ΔW_B`, in training and
at inference, so the post-train merge is **exact by construction**: saving
`ΔW_A + ΔW_B` as one LoRA reproduces the trained network bit-for-bit (see
"Save path" — factor concatenation, no SVD, no approximation). The output
stays a **plain stock LoRA** — `make merge` works, any loader works, concept
LoRAs compose. The headline turbo property is preserved.

### The variant this proposal explicitly rejects

"Train LoRA-A active only on step 0 and LoRA-B active only on steps 1..N−1
(a 2-way `per_step_expert`), then merge them post-train" — **unsound**. A
static merge applies both deltas at *every* step, but during training step 0
never saw B and the refinement steps never saw A. The merged network computes
a velocity field neither objective ever supervised — a train/inference
mismatch at exactly the tuned points. This is the same reason `make merge`
refuses `per_step_expert` checkpoints (step-conditioned adapters don't fold
into one static weight); a post-hoc merge doesn't dodge the constraint, it
silently violates it. Always-on + gradient routing is the only formulation
where "merge → plain LoRA" is honest.

## Why this line, here

- **It closes a real, documented gap.** `detach_after_first` is load-bearing
  (their Fig 5: preference rises while diversity falls without it) — but it
  severs the *graph*, not the *parameters*. Both losses still update the same
  LoRA weights, so the mode-seeking DMD gradient still overwrites the
  diversity mapping at the parameter level; the detach only stops it flowing
  through step 0's activations. Dual pools finish the separation the detach
  started.
- **It is the `per_step_expert` hypothesis without its costs.**
  `per_step_expert` tests the same interference question but (a) gives up the
  plain-LoRA bake, (b) is incompatible with the warm start (needs plain
  single-head Linear modules), and (c) is incompatible with
  `dynamic_schedule=true` (heads keyed to fixed grid steps) — i.e. it can't
  even run against the shipped defaults. Dual pools have none of these
  conflicts: both pools are plain `LoRANetwork` stacks, nothing is keyed to a
  grid step, and the warm start applies per-pool.
- **The mechanics are already there.** Under `split_bwd`
  (`use_anchor and detach_after_first`, shipped true) the loop *already*
  backwards the diversity term separately (`distill.py` step-0 branch) from
  the DMD/GAN backward. Routing is two `requires_grad_` toggles around
  backwards that already exist — no new forwards, no new backward passes, no
  view machinery beyond what `TurboDMDNetwork.set_view` does today.
- **It buys a free knob.** With Pool A zero-initialized (see seeding), A
  converges to an isolated, inspectable "de-collapse delta" on top of the
  warm-start map. At save/merge time A can carry its own multiplier — a
  post-train **diversity dial** (0 = the warm-start-ish quality map, 1 =
  full anchor correction), scalable without retraining. No current turbo
  artifact offers that.

## Mechanism

### Gradient routing (the whole trick)

Per training iteration, inside the existing `split_bwd` structure:

1. **Before the step-0 forward**: `pool_B.requires_grad_(False)`. Step-0
   forward runs with both pools active; `div_loss` (+ soft-rank) backwards
   immediately, exactly as today. Only A accumulates grads. This works
   *because* the step-0 graph is built and consumed inside this window — the
   `detach_after_first` early backward is what makes the routing a two-line
   change rather than a graph surgery.
2. **After the step-0 backward, before the rollout continues**:
   `pool_B.requires_grad_(True)`, `pool_A.requires_grad_(False)`. The
   refinement rollout, DMD surrogate, CDM branch, and GAN generator forward
   all build graphs that only reach B. The combined student backward
   (`loss_student.backward()`) lands on B alone. Restore A after.
3. `grad_clip` covers both pools' accumulated grads once, as today. One
   AdamW with two param groups (per-pool LR — the diversity signal is a
   single MSE at `div_weight=0.05`, so A likely wants a hotter LR than B;
   start matched, tune later).

Both pools stay **enabled** (activations flow) throughout — only grad
accumulation is gated. The fake/critic side is untouched: the fake still
tracks the full student `x_θ.detach()` distribution, which is A+B's output.

Guards (config resolve): requires `detach_after_first=true` (the routing
window *is* the split backward — without it there is nothing to route);
mutually exclusive with `per_step_expert` (both restructure the student);
plain DMD mode (`use_anchor=false`) has no diversity gradient to route, so
dual-pool is dpdmd-only.

### Seeding — don't double the warm start

Naively warm-starting both pools from the turboV10 delta doubles it in the
sum. The clean init:

- **Pool B** ← `warm_start_plain_lora` from the extracted delta (B owns the
  few-step quality map — same role as today's student init).
- **Pool A** ← `lora_up = 0` (standard zero-init up-proj, kaiming down). At
  step 0 the merged student *is* the warm start; A grows a pure diversity
  correction from zero.

`fake_init_weights` unchanged (fake tracks A+B ≈ B at init — still
calibrated).

### Rank budget — the user-level knob

The merged checkpoint has rank ≤ `rank_A + rank_B`, so the **fair baseline is
a single pool of the summed rank**, not the shipped 96. Two arms worth
having:

| Arm | A | B | Baseline it must beat |
|---|---|---|---|
| symmetric (the original sketch) | 32 | 32 | single r=64, joint grads |
| asymmetric (recommended) | 16 | 96 | shipped single r=96 |

The asymmetric prior: the diversity signal is one MSE at λ=0.05 on one step —
a full r=32 pool for it is likely overkill, while r=32 for the quality side is
a real downgrade from the r=96 ASVD warm start (rank-96 ASVD captures ≈ a
plain rank-128 extraction; truncating it to 32 throws capture away). r=16 for
a zero-init correction delta is generous; r=96 keeps B a strict superset of
today's student. Cost of the asymmetric arm: +16 ranks of params/grads/Adam —
noise next to the fake stack.

### Save path — exact merge, no SVD

Two rank-r LoRAs on the same Linear merge exactly by factor concatenation:
`down = concat(A_down, B_down)` (rows), `up = concat(A_up, B_up)` (cols) →
one plain LoRA of rank `r_A + r_B` with `ΔW = ΔW_A + ΔW_B` identically.
`save_student` gains a concat step; optional `--soup_rank`-style SVD
truncation back to a smaller rank can come later if checkpoint size matters
(it shouldn't at 112). Stamp `ss_turbo_dual_pool` + per-pool ranks in
metadata for arm verification — never trust the TOML you think you ran
([[project_turbo_sectioned_config_silent_default]]). A merge-time
`--div_scale α` writes `concat(α·A_up, B_up)` — the diversity dial.

### What it costs

- **Compute**: zero extra forwards, zero extra backwards. Each Linear runs
  two LoRA GEMMs instead of one in the student view (the fake view already
  pays this class of cost); at r=16+96 vs r=96 this is noise.
- **Memory**: +1 LoRA stack (params/grads/Adam moments) at `rank_A`. At r=16,
  ~1/6 of the current student stack.
- **Compile**: both pools apply before `compile_dit_blocks`
  (compile-after-apply invariant); `requires_grad_` flips on module params
  don't re-trace. Needs the same verification pass as any view-adjacent
  change ([[project_turbo_view_ckpt_recompute_hazard]] — the GAN gen forward
  must stay the last view flip before backward; routing toggles do not move
  it).
- **Resume**: bundle layout changes (two student stacks) — dual-pool runs
  resume only into dual-pool configs; refuse cross-loading, like the τ-bank
  split did.
- **Nonstationarity**: B optimizes DMD against a slowly-moving A and vice
  versa. At λ=0.05 with A zero-init this is mild, but it is one more reason
  `fake_warmup_steps` and the `grad_signal_rms` watch stay on.

## Config surface (sketch)

```toml
[network]
dual_pool          = false   # off = byte-identical shipped loop
div_pool_rank      = 16      # pool A (zero-init up, kaiming down)
div_pool_lr        = 0.0     # 0 = inherit student_lr
# student_rank / student_init_weights keep their meaning — they become pool B
```

Every key gets a CLI override flag, per the bespoke-schema convention.
`dual_pool=false` constructs one stack and skips every toggle — byte-identical
to the shipped loop (the same kill-switch discipline as `weight_gen=0` and
`fake_tau_banks=1`).

## Phase 0 — wiring + smoke A/B

One variable: parameter routing. Warm start, GAN, `dynamic_schedule`,
`div_weight`, seed, data — all fixed across arms
([[project_turbo_warmstart_scope]]: never let an objective A/B double as an
init A/B).

- **Arm A**: `dual_pool=true`, A=16 zero-init / B=96 warm-started.
- **Arm B (baseline)**: shipped single r=96 student, same everything.
- 750 iterations (the shipped fine-tune length), ckpt at 250/500/750.
- **Primary read**: rendered 4-step grids at `--cfg 1.0` (`make gen`), fixed
  prompt set × seed sweep, human A/B
  ([[project_turbo_lr_instability_threshold]]: rank by grids, not `fm_mse`).
- **Diversity read**: cross-seed `ac_sim` + seed-grid spread per prompt —
  this is the axis the whole proposal exists for; `div_loss` is *not* it
  (it measures anchor-hitting, not diversity survival — the k-anchor A/B
  caveat applies verbatim).
- **Quality guard**: glyph probe + saturation vs baseline — B must not
  regress with the DMD grad now denied A's parameters.
- **Bonus read (cheap, unique to this design)**: render arm A at
  `--div_scale 0 / 0.5 / 1.0` — if the dial visibly trades diversity against
  the init's mode, that is direct evidence the routing isolated the axis,
  independent of whether the arm wins outright.

**Gate to adoption**: arm A ≥ baseline on rendered NFE=4 AND measurably
better on cross-seed diversity AND no glyph/saturation regression.
**Kill**: arm A ties on diversity → the single student was never
parameter-interference-bound (the same verdict framework as
`per_step_expert`: "if the shared LoRA was never capacity/interference-bound
it buys a heavier checkpoint for nothing") — close the line, keep the doc as
the record. A *diversity win with a quality loss* is not an auto-kill:
re-run at B=128 before deciding (capacity, not routing, may be binding).

Per CONTRIBUTING tier rules this is a numerics change: Phase 0 lands with a
bench script under `bench/turbo/` (result envelope per `bench/_common.py`)
and an invariant test (routing correctness: after one synthetic step, A's
grads are nonzero/B's zero under the step-0 backward and vice versa; plus
merged-save equivalence: saved concat LoRA reproduces `ΔW_A + ΔW_B`).

## Relationship to open lines

- **CDM (`docs/proposal/cdm.md`)**: orthogonal and compatible — L_CDM is a
  refinement-side loss, routes to B like the rest. If both lines win they
  stack.
- **`per_step_expert`**: superseded for the div/DMD split specifically (this
  design tests the same hypothesis while keeping the bake, the warm start,
  and the dynamic schedule). Still the only option for *per-refinement-step*
  role separation, which dual-pool deliberately does not attempt.
- **τ-split critic (CLOSED 2026-07-20)**: superficially similar
  ("split a stack in two") but on the fake side and split by τ, not by
  objective; its closure verdict (interference was an LR artifact) is a
  cautionary precedent — hence the rank-matched baseline and the fixed-LR
  first arm here.

## References

- `docs/methods/turbo.md` — DP-DMD loop, `detach_after_first`,
  `per_step_expert` and its constraints, warm-start contract.
- `scripts/distill_turbo/distill.py` — the split backward this routing rides.
- `networks/methods/turbo_dmd.py` — `TurboDMDNetwork` view toggling,
  `warm_start_plain_lora`.
- Wu, Li, Zhang, Ma — arXiv:2602.03139 (DP-DMD; Fig 5 is the interference
  evidence the detach — and this proposal — respond to).
