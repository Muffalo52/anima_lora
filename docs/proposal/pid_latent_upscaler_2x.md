# PiD-2x — a Qwen-latent-native 2× upscaler, by regression-finetuning PiD, then distilling

Status: **PROPOSAL, no GPU work done.** Every architectural claim below is
verified by code-reading / meta-device construction / checkpoint introspection
(cited inline). Nothing has been rendered or trained. Phase 0 is inference-only
and is the kill-shot.

## TL;DR

PiD is already a direct latent upscaler in *our* latent space. Retargeting it
from 4× to 2× needs **zero architecture surgery** (verified: 456/456 param keys,
zero shape mismatches). The build order is:

1. **Phase 0 (inference only, no training):** does `down(PiD@4×)` beat the shipped
   RSD-x2 student on art? If no → **kill the line.**
2. **Phase 1:** regression-finetune a `sr_scale=2` PidNet, warm-started from the 4×
   EMA, against `down(PiD@4×)` targets. **Not DMD** — DMD is ill-typed for this axis
   (§4).
3. **Phase 2 (gated):** *then* consider distillation — cut 4 steps → 1–2 via
   LoRA-DMD, teacher = the Phase-1 student. Here DMD types correctly.

The two axes (**scale** 4×→2×, **steps** 4→1) are separate problems with different
correct objectives. Conflating them is the main design error this proposal exists
to prevent.

## Premise: what PiD actually is

Established by reading `~/ComfyUI-Anima-PiD/{pid_core.py,pid_net/*}` and
introspecting `comfy/models/pid/pid_qwenimage_2kto4k_4step.pth`.

- PiD (NVIDIA PixelDiT, `official_qwenimage` ckpt) consumes a **normalized 16-ch
  Qwen VAE latent and emits RGB pixels directly**. There is **no VAE decode** at the
  end. `lq_in_channels=0`, so the pixel/image LQ branch is not even constructed —
  conditioning is *latent-only*, via `lq_proj` injected controlnet-style every
  `lq_interval=2` blocks.
- **PiD's latent space IS Anima's latent space** — same Qwen VAE, and
  `pid_core.QWEN_LATENTS_MEAN/STD` are literally our `qwen_vae.latents_mean/std`.
  This is why the shipped decode node works so well on our gens, and it is the
  whole basis of this proposal.
- Output size = `latent_grid × 8 (VAE) × sr_scale(=4)`. A 1024px gen (128×128
  latent) → 4096px.
- **1,362.3M params** (measured, bf16): patch_blocks 1189.5M, pixel_blocks 105M,
  lq_proj 59.9M. This number drives the whole VRAM argument in §5.
- Flow-matching velocity net; the released checkpoint is *already a distilled
  4-step student* (`STUDENT_T_LIST = [0.999, 0.866, 0.634, 0.342, 0.0]`).

So PiD is a **super-resolving replacement for `vae.decode`**, not an image-SR model.
It reads the latent the generator already produced. That distinction drives §3.

## The core structural finding: 2× is free, architecturally

`sr_scale` reaches the weights through exactly one quantity:

```
z_to_patch_ratio = (sr_scale × latent_spatial_down_factor) / patch_size
                 = (sr_scale × 8) / 16
```
(`pid_net/lq_projection_2d.py:250`)

- At **4×**: ratio = 2 → nearest-upsample the latent onto the patch grid.
  **Parameterless** (`lq_projection_2d.py:253-257` — the learned PixelShuffle
  upsampler was removed upstream over DDP issues).
- At **2×**: ratio = 1 → identity path. `latent_proj_in_ch` stays `latent_channels`
  = 16 either way, so `lq_proj.latent_proj.0.weight` remains `(512, 16, 3, 3)` —
  confirmed against the on-disk checkpoint.

Verified by constructing `PidNet` on the meta device at both scales:

```
sr4 params: 456   sr2 params: 456
keys only in sr4: set()      keys only in sr2: set()
SHAPE MISMATCHES: {}
z_to_patch_ratio: sr4 = 2.0  | sr2 = 1.0
latent_upsample_ratio: sr4 = 2 | sr2 = None
```

**The released 4× EMA loads into a 2× net with no surgery.** Everything else in the
net — 14 patch_blocks, 2 pixel_blocks, the LQ trunk — is patch-grid-agnostic DiT
with RoPE.

RoPE is *not* the obstacle here (a common worry, checked and dismissed): positions
are a normalized `linspace(0, 16, grid)` span with NTK theta scaled by grid/ref
(`pixeldit_official.py:167-201`), and a 2× decode of a 1024px gen is a 128 grid,
which the 4× teacher trained on (its 2048px outputs). The real distribution shift is
**content density**: each 16×16 output patch must now carry one full latent cell
(4× the semantic content) and paint texture at half its trained zoom.

## Why this is worth doing at all (and the honest ceiling)

Be blunt about what this is:

- **Against PiD itself, it is a pure compute play.** The student can never beat
  `down(PiD@4×)`, which we can already run today. What it buys is cost: 4× fewer
  output pixels, and — because full attention is quadratic in the
  `(H/16)×(W/16)` patch grid (`pid_net.py:291-292`) — a **16× cut in the attention
  term**. A 1024px gen at 2× is a 128² = 16.4k-token grid vs 256² = 65.5k at 4×.
- **Against the shipped RSD-x2 sidecar, it is a quality play** — and this is the
  only reason to build it. The RSD x2 line is **CLOSED at a ceiling** (24k steps ≡ 2k
  steps, dead tie; root-caused to the teacher plus the **VQ-f4 reconstruction
  floor**, not to undertraining). Memory says: do not retrain that teacher without a
  genuinely new lever.

  **PiD is that lever.** It is pixel-space with no recon floor, and it reads the true
  generation latent instead of re-encoding decoded pixels through a second, lossier
  autoencoder. It deletes precisely the floor that closed the line.

Cost honesty even on success: PiD-2x @ 4 steps ≈ 4 NFE × 1.36B at 16k tokens, versus
RSD's 1 NFE × 119M SwinUNet — order **5–10× RSD's cost**. So the line only *fully*
lands if Phase 2 (step-cut) also works. Phase 2 should be treated as part of the
commitment, not a stretch goal.

## The rejected alternative: "RSD-arch direct latent upscaler"

Considered and **rejected**: keep the RSD machinery (SwinUNet student, DMD critic +
GAN + LPIPS, 1-step) but retarget it to eat the Qwen 16-ch latent and emit pixels
directly, deleting VQ-f4.

It fails on its own premise — *"we reuse a training loop that already works"* is
false. Every load-bearing component of RSD is defined **inside** ResShift/VQ-f4
space:

- `L_theta` is a fake-vs-teacher x0 gap in the teacher's noisy *residual-shift*
  latent space (`sr/distill_rsd/train.py:248-267`). No ResShift → no `q_sample`, no
  DMD gradient.
- The GAN head taps the **fake critic's bottleneck**
  (`sr/distill_rsd/rsd_models.py:294-306`). No critic → the discriminator has
  nowhere to live; you're back to a standalone from-scratch disc, i.e. exactly the
  classical GAN-SR instability RSD exists to sidestep.
- The student warm-start grafts teacher weights strict
  (`rsd_models.py:337-351`); a 16-ch stem voids it.

What survives the port is **LPIPS and the dataloader**.

Nor is the architecture a head swap. Today's SwinUNet does **zero net spatial
expansion** — it is latent→latent (`in_channels: 3, out_channels: 3`), and *all*
pixel synthesis is done by the co-trained VQ-f4 decoder. Retargeted, the student
must itself perform a 16× linear / 256× area expansion from a 16-ch latent, i.e.
bolt on a pixel-synthesis stack about the size of the Qwen decoder (73.3M measured)
— entirely random-init.

Stripped of the parts that don't survive, this proposal is **"finetune the Qwen VAE
decoder into a 2× SR decoder with GAN + LPIPS + L1"**. That is a legitimate and
well-trodden idea (SD-VAE decoder finetunes, consistency decoders, latent
upscalers) and it deserves its right name — but it is a from-scratch trunk with
from-scratch GAN stabilization, whose quality target is set by a **1.36B, 4-NFE
model trained at scale on the identical latent space**. A ~150M 1-NFE GAN on one
16GB card plausibly lands *below* the VQ-f4 line it was meant to replace.

**Verdict: don't build it.** If it is ever revisited, the strongest form starts from
`QwenImageDecoder3d` (`library/models/qwen_vae.py:905`) with one extra 2× up-stage
appended to `dim_mult`, not from the SwinUNet.

## Objective: why regression, not DMD, for the scale axis

The decisive argument is **not** VRAM — it is that **DMD cannot be formulated** for
a 4×→2× retarget.

DMD's generator gradient evaluates a *frozen teacher's score* on re-noised
**student** outputs (`sr/distill_rsd/DESIGN.md:91-106`). The student emits
2×-regime images. The frozen teacher `f*` only defines a score in **its own 4×
space**. Feeding a 2×-regime image to the 4× teacher lands you in precisely the
off-distribution regime you are trying to train *away* — the "teacher score" is
garbage exactly where it is needed. To obtain a valid teacher score in 2× space you
would first need a flow model of `down(PiD@4×)`'s distribution, which is the thing
being built. Chicken-and-egg.

So for the **scale** axis, DMD is off the table on type grounds alone. (VRAM agrees
independently — see §5.)

**Is regression safe here?** Our own precedents say to be suspicious: SPD's plain
Eq-14 MSE distillation **blurred**, and RSD itself needed LPIPS + GAN. But both
precedents involve *unconditioned target stochasticity*. Three things break that
pattern here:

1. The teacher is **heavily LQ-conditioned SR** — the latent pins content; seed
   variance lives mostly in fine texture.
2. The `/2` downsample **annihilates the finest stochastic band** — exactly the
   most seed-dependent components.
3. **We can match the noise.** The sampler draws fresh ε per step
   (`pid_core.py:389`), and `ε_2x = 2·avg_pool2(ε_4x)` is exactly N(0,1) white — so
   the student can be conditioned on the coarse projection of every teacher draw.

Residual conditional variance is then only the nonlinear leakage of fine-only noise
into coarse output — and that is **measurable for free in Phase 0** (two teacher
seeds sharing a coarse noise projection → PSNR between their downsampled outputs).

### Recommended objective (Phase 1)

**Matched-noise 4-step trajectory-endpoint regression.** The student rolls out its
own 4-step SDE at 2× using coarse-grained teacher noise; the loss compares the
rollout *endpoint* against `down(teacher@4×)`:

```
L = L1 + λ_lpips · LPIPS + λ_dc · dc_loss
```

- Reuse `dc_loss` from `sr/distill_rsd/train.py:29` — **load-bearing**, not
  optional: PiD has a known color-drift / flat-desaturated issue
  (`pid_core.py:178-181`), and the DC anchor is exactly the fix that worked for the
  RSD student.
- **Roll out the real sampler**, do not fit a plain flow-matching loss on
  `(latent, down(teacher_out))` pairs. The released net is *already a distilled
  4-step student*; plain FM finetuning would drift its velocities back toward the
  mode-averaged FM optimum — **undoing the distillation** and reintroducing 4-step
  blur. Keep plain-FM as a fallback arm only.
- Backprop through 4 forwards with per-block grad checkpointing.
- Keep the RSD **GAN head as a rescue lever, not a default**. A small disc on
  `down(teacher)`-real vs student-fake types fine (discriminators don't need teacher
  scores), but only add it if Phase 0's seed-variance probe says regression will
  blur.

## Phase 2 (gated): the step-cut, where DMD *does* type correctly

Once a 2× 4-step student exists, cutting **4 steps → 1–2** is the axis where
regression genuinely blurs (steps 2–4 inject noise a 1-step student never sees →
conditional averaging). That is RSD's exact problem shape, and it now types
correctly: **teacher = our Phase-1 2× student**, critic = a flow net in the *same*
2× space.

Implement as **LoRA-DMD**, lifting the turbo two-optimizer pattern
(`scripts/distill_turbo/distill.py:348-361` — LoRA student + LoRA fake over one
shared frozen base; two optimizers at `:492-511`). This is also the **only** DMD
that fits in 16GB (§5).

Do the axes **in sequence, scale first**. Doing both at once conflates failure
attribution and forces DMD into the ill-typed cross-scale form.

## VRAM budget (16GB RTX 5070 Ti, Blackwell sm_120, no xformers)

At 1.36B params/net:

| Config | Weights | Grads | Optim | Total (pre-activation) | Verdict |
|---|---|---|---|---|---|
| Full-param DMD (teacher + student + critic) | 2.7+2.7+2.7 | 2.7+2.7 | 2×2.7 (8-bit) | **~19 GB** | **Dead.** (fp32 AdamW ≈ 30 GB) |
| **Phase 1**: endpoint regression, full-FT student, *offline* targets | 2.7 | 2.7 (bf16) | 2.7 (adamw8bit) | **~8.2 GB** | **Fits** (+2–4 GB activations at 24–32² latent crops w/ block-ckpt) |
| **Phase 2**: LoRA-DMD, shared frozen base | 2.7 + 2 LoRA sets | tiny | tiny | ~6–8 GB | Fits |

Full-param DMD not fitting is **fine** — regression is the right objective for the
scale axis anyway, and Phase 2's LoRA form is what the turbo loop already does.

Pre-generating teacher targets **offline** is what keeps Phase 1 in budget (no 1.36B
teacher resident during training).

## Phased plan

### Phase 0 — kill-shot. Inference only. No training code. (~1 day GPU)

1. **Ceiling gate (THE gate).** `down(PiD@4×)` vs the shipped RSD-x2 student vs
   bicubic, on ~30 real art gens. MUSIQ/CLIPIQA + eyes. **PiD-downsampled must
   clearly win. If it does not → DON'T BUILD, close the line.** Real risks that
   could fail this: PiD's flat/desaturated color drift, and its photo-domain
   checkpoint on art.
2. **Seed-variance gate** (decides the objective). Fixed latent, two teacher seeds —
   and two *fine* seeds sharing one coarse projection. PSNR/LPIPS between the
   downsampled outputs.
   - High (≳33 dB) → plain regression is safe; matched noise optional.
   - Low → matched noise mandatory.
   - Very low → budget the GAN head from day 1.
3. **Zero-shot `sr_scale=2` render** (diagnostic, *not* a gate). Decode 5–10 latents
   at `sr_scale=2` with the EMA loaded as-is.
   - *Encouraging*: coherent structure, correct color, defects confined to texture
     scale (over-sharp / duplicated micro-detail, "zoomed-in" crunch) — a
     texture-frequency shift that finetuning plausibly fixes.
   - *Hopeless*: patch-grid tearing, content duplication across patches, or
     noise-like output — structural failure a few-thousand-iter FT won't recover.
   - Note: a poor render does **not** kill the line; it means warm-start buys less
     and Phase 1 costs more. Only gate #1 kills.

### Phase 1 — scale retarget by regression

- Offline teacher targets: 32² latent crops → 1024² teacher tiles, ~3–5 crops ×
  2,972 cached latents (`post_image_dataset/lora/**/*_anima.npz` — **already on
  disk, already exactly PiD's input distribution**; zero degradation pipeline
  needed). ~0.5–1.5 GPU-days.
- Endpoint-regression finetune, full-param student, warm-started from the 4× EMA.
- **500-iter A/B gate before committing** to the full ~5–10k-iter run (~1–2 days) —
  RSD's own protocol.
- Eval vs `down(PiD@4×)` (the ceiling) and vs RSD-x2 (the incumbent).

### Phase 2 — step-cut by LoRA-DMD (gated on Phase 1)

4 → 1–2 steps. Teacher = the Phase-1 student. Direct reuse of the turbo/RSD loops.

## What must be written vs reused

**Reuse:**
- Teacher inference is *done* — `pid_core.py` decode + tiled decode with the
  `global_lq` GroupNorm fix (`:431-501`), null caption, color calib.
- From `sr/distill_rsd/`: `dc_loss`, LPIPS wiring, `library.training.ema`, the eval
  harness (`infer.py` + pyiqa), the bench/report pattern.
- From `scripts/distill_turbo/`: two-optimizer loop, resume bundle,
  `selective_block_grad_ckpt` — **Phase 2 only**.
- Data: 2,972 cached Qwen latents, already on disk.

**Write:**
- `sr/distill_pid2x/` train script — nothing in-tree trains a `PidNet`.
- Per-block grad-ckpt for `patch_blocks`/`pixel_blocks` (~20 lines; the compile
  helper at `pid_core.py:292-316` already shows the block structure).
- A parametrized decode wrapper — `pid_core.py:86` hardcodes `SR_SCALE=4` at module
  level and `:363` bakes it into H,W. `PidNet` itself already takes `sr_scale`, so
  this is a thin fix.
- Offline teacher-target generation script.

## Top-3 risks

1. **Texture-scale regime resists finetuning** — the student keeps hallucinating
   4×-zoom micro-texture at 2×. This is the main "hopeless" scenario. Detected by the
   Phase-0 zero-shot probe plus the 500-iter A/B trend; mitigation is full-FT (not
   LoRA) and more crops.
2. **Regression blurs despite the downsample** — PiD hallucinates *coarse*-scale
   content (strokes, glyphs) that moves with seed, so conditional variance survives
   `/2`. Caught by Phase-0 gate #2 *before* any training. Rescue ladder: matched
   noise → GAN head → LoRA-DMD in 2× space (valid once a Phase-1 student exists to
   serve as base).
3. **Economics** — either `down(PiD@4×)` fails to beat RSD-x2 on art (killed free by
   gate #1), or it wins but the 4-step 2× student remains too expensive to displace
   a 1-NFE 119M model. The latter is why **Phase 2 is part of the commitment**, not
   optional.

## Open questions

- **Caption conditioning.** PiD takes a gemma-2304 caption embedding; the node feeds
  a fixed bundled null. When upscaling *our own* gens we actually **have** the
  caption — but using it requires gemma, which the node deliberately avoids loading.
  Worth a look *after* Phase 1; not on the critical path.
- **v1 vs v1.5.** The on-disk ckpt is v1 (456 keys, no `pit_head`); the node
  auto-downloads v1.5 (wider LQ trunk, `sigma_aware_per_token` gate, `pit_lq_inject`,
  replicate padding). The 2× shape-invariance holds for both, but **Phase 1 should
  warm-start from v1.5** — it is the better model and fixes the corner-grid artifacts.
  Note v1.5's color calib is off by default (upstream fixed color), so re-check the
  `dc_loss` weight against v1.5, not v1.
