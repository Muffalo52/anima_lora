# ResShift ×4 SR — a standalone super-resolution sidecar

> Proposal to stand up a **dedicated ResShift super-resolution model**
> (*Efficient Diffusion Model for Image SR by Residual Shifting*, Yue, Wang,
> Loy — NeurIPS 2023 Spotlight / TPAMI 2025; repo
> `github.com/zsyOAOA/ResShift`; PDF context from the RSD distillation paper
> `22490_One_Step_Residual_Shifti.pdf` in repo root) trained on **our**
> high-res art, with a Phase-2 path to a **one-step** student via RSD/SinSR
> distillation.
>
> Read [[project_qwen_vae_2d_fold]] and the "Methods" / "Critical invariants"
> sections of `CLAUDE.md` first **only to understand why this is NOT an Anima
> adapter** — the conclusion below is that it deliberately lives outside the
> adapter system.

Status: **Phase 0 DONE (2026-06-29) — gate PASSED, released model transfers well.**
Sidecar stood up under `sr/` with an isolated `sr/.venv` (Blackwell torch, no
xformers; see `sr/README.md`). Released ResShift ×4 v3 beats bicubic on every metric
(LPIPS 0.12 vs 0.35, MUSIQ 74 vs 41) with no hallucination/color-shift on flat art —
the feared photo-prior domain gap did **not** materialize. The env-pin assumptions in
§0/§3 are corrected in `sr/README.md` (this box is sm_120; the pinned torch/xformers
don't run). Phase 1 (×2 finetune + degradation ablation) is next but now justified by
the true-×2 product + pipeline-matched degradation, not by a large domain gap.
Originally written in response to "can we train
an SR model here, e.g. 1024²→2048²." The honest answer drove the shape: a small
convolutional SR diffusion model is the right tool, and it is *independent of
Anima by design* — so this is a **sibling subproject**, not a `networks/` method.

---

## 0. Why a sidecar, not an Anima adapter (read this first)

The instinct "train an SR LoRA on the Anima DiT" (the OSEDiff / PiSA-SR recipe)
is feasible only at modest resolution and is the **wrong architecture** for the
stated goal (1024²→2048²). Two hard walls, both from the DiT being a global-
attention transformer:

1. **Trained token bands top out at edge 1536 (~8640 tokens)** (`EDGE_TOKEN_BANDS`,
   `CLAUDE.md` free-fit invariant). A 2048² latent is ~16k tokens — beyond every
   trained tier *and* the compile token-family budget. Out of distribution.
2. **Attention is O(tokens²)**, and EasyControl's *extended* self-attention
   concatenates cond+target tokens, making it worse. VRAM scales badly exactly
   where we need it to scale well.

ResShift sidesteps both because it is a **small SwinU-Net (~120–170M params) in an
LDM VQ-f4 latent**, and it **tiles** (`chop_size`/`chop_stride`, e.g. 512/448) —
so peak VRAM is flat regardless of output size. Table 4 of the RSD paper: ~539 MB
inference VRAM vs OSEDiff 1.7 GB / SUPIR 52 GB. The lightness is *architectural*
(local conv in a compact latent), not a property of the SR task.

**The cost of the sidecar:** we forfeit Anima's rich generative prior. ResShift is
faithful-but-conservative (can stay soft on severely degraded input); a T2I-prior
SR like SUPIR hallucinates rich-but-wrong detail. For *clean upscaling of our own
art* (not blind real-photo restoration) the faithful end is what we want, so the
tradeoff favors ResShift. If we later want "invent plausible texture in our house
style," that is the EasyControl-SR route and a *different* product — see §7.

**Consequence for integration:** ResShift does not touch `train.py`, the adapter
families, the Qwen VAE, or the config merge chain. It brings its **own** VQGAN,
its **own** BasicSR/`torchrun` training loop, and its **own** latent space. We
reuse exactly one thing from this repo: the **image data**. Everything else lives
under a new top-level `sr/` (or a sibling checkout) and is wired into `tasks.py`
only as thin `make sr-*` dispatch targets so it is discoverable.

---

## 1. What ResShift actually is (grounded in the repo)

- **Model**: SwinU-Net denoiser predicting `x₀` in **VQ-f4 latent space** (the LDM
  VQGAN, downsampled by 4; ResShift ships this VQGAN separately — *not* Anima's
  Qwen VAE). A residual-shifting Markov chain transports HR↔LR latents by
  shifting the residual `e₀ = y₀ − x₀`, so it converges in **15 steps (v1/v2)**
  or **4 steps (v3, journal)** with no post-hoc acceleration.
- **Training**: `main.py --cfg_path configs/realsr_swinunet_realesrgan256.yaml`
  under `torchrun`. HR crops are **256²**; LR is synthesized **on the fly** by the
  **Real-ESRGAN degradation pipeline** (blur → resize → noise → JPEG). So we do
  **not** pre-build a paired dataset — we point it at a folder of HR images and it
  degrades them live. (`bicsr` config = clean bicubic degradation instead.)
- **Scale**: ships **×4** (`realsr_*`) and **×2** (added 2023-12-02). A pretrained
  VQGAN goes in `weights/`; SR checkpoints (v1/v2/v3) are on the GitHub releases.
- **Inference**: `inference_resshift.py -i <in> -o <out> --task realsr --scale 4
  --version v3`, tiled via `--chop_size 512 --chop_stride 448`.
- **License**: **S-Lab License 1.0 — non-commercial.** Flagging up front since this
  is a real project; fine for research/personal, gating for any commercial use.

### 1a. The ×2 vs ×4 decision (your exact case)

You assumed "no ×2, only ×4." Good news: **ResShift ships a ×2 config**, which is
the *direct* fit for 1024→2048. The ×4 path still works for the same goal — feed a
512² crop, get 2048² — but for true ×2 the ×2 config trains a model whose
degradation/shift schedule is matched to a 2× gap, which is easier and sharper
than asking a ×4 model to operate out of its trained regime. **Recommendation:
train the ×2 config for the 1024→2048 product; keep ×4 as a second model if you
also want 512→2048.** (Both reuse the identical data + degradation machinery.)

### 1b. Degradation: Real-ESRGAN vs bicubic (consequential)

Real-ESRGAN degradation models *photographic* sensor/codec artifacts. Our data is
**clean line/paint art**, where the realistic LR is closer to **bicubic / mild
blur + light JPEG**, not heavy sensor noise. Training on the full Real-ESRGAN
pipeline risks a model that "denoises" art that was never noisy and softens clean
edges. **Recommendation: start from the `bicsr` (bicubic) degradation and add only
light JPEG + mild blur**, rather than the full real-photo pipeline. This is the
single biggest quality lever for an art-domain SR model and should be the first
ablation in Phase 0.

---

## 2. Data — what we already have, what we build

- **Source HR**: `image_dataset/` (symlink → nested artist dirs; use `find -L` /
  rglob, never plain `find` — [[project_image_dataset_symlink_nested]]). These are
  our high-res masters; they become the **HR target** pool.
- **Filtering**: SR training needs genuinely sharp HR. Drop any image whose native
  short edge < the HR crop size after our own upstream resize, and drop
  already-soft scans (a Laplacian-variance floor). This mirrors preprocess's
  `drop_lowres_images` intent but is SR-specific — a new `sr/scripts/build_hr_pool.py`.
- **No pre-paired LR needed** for training (degradation is live). We only build a
  **held-out eval set** of fixed LR↔HR pairs (degrade once, freeze) so quality is
  measured on a stable benchmark across runs — `sr/data/eval_pairs/`.
- **Crop size**: paper uses 256² HR. For a 1024→2048 product, training at 256² HR
  (→512² output region after ×2... ) is fine — the model is fully convolutional /
  tiled, so train-crop and inference-output sizes are decoupled. Keep 256² to
  match released hyperparameters in Phase 0; revisit at Phase 1.

---

## 3. Phasing

### Phase 0 — reproduce + sanity (no training of our own yet)
**Goal: confirm the released ResShift ×4/×2 works on our images at all**, before
spending GPU-weeks. Pure inference.
1. Clone ResShift into a sibling dir; create a `resshift` env (Python 3.10,
   torch 2.1.2, xformers 0.0.23 — **separate env from Anima's uv project**, do not
   pollute `pyproject.toml`).
2. Download VQGAN + v3 (4-step) SR checkpoint from their releases.
3. Run released `inference_resshift.py --version v3` on ~30 of our art images
   (downscaled to make synthetic LR) at ×2 and ×4.
4. **Gate**: do the outputs look like plausible upscales of *our* art, or does the
   ImageNet/photo-trained prior fight the art domain (texture hallucination, color
   shift)? Eyeball + CLIPIQA/MUSIQ on the eval set. This decides whether Phase 1
   is "light finetune" or "train from scratch."
   - *Expected*: photo-domain prior will be visibly off on flat-shaded art → Phase
     1 is needed. This is the whole reason to train on our data.

### Phase 1 — domain finetune (the core deliverable)
**Goal: a ResShift ×2 (and optionally ×4) model that upscales *our* art well.**
1. Build the HR pool + frozen eval set (§2).
2. Pick degradation (§1b) — first ablation: `bicsr` vs light-Real-ESRGAN, 2 short
   runs, pick on eval CLIPIQA/MUSIQ + eyeball.
3. **Finetune from the released checkpoint** (not from scratch) — converges far
   faster and inherits a working SR prior; our data shifts it to the art domain.
   `torchrun --nproc_per_node=<N> main.py --cfg_path configs/<our_x2>.yaml
   --resume <released_ckpt> --save_dir output/sr/<run>`.
4. Budget: released v1 = 300k iters / v2 = 500k *from scratch*; a **domain
   finetune should need ~20–50k** iters. On a single GPU this is days, not weeks —
   measure the eval curve and stop when it plateaus.
5. **Deliverable**: a 4-step (v3-schedule) art-SR checkpoint + a `make sr-test`
   that tiles 1024→2048.

### Phase 2 — RSD one-step distillation (PROMOTED to the primary goal, 2026-06-29)
**Goal: reconstruct RSD and collapse the 15-step ResShift teacher → 1 step.** After
Phase 0 showed the released model already transfers to our art, the objective
re-centered on **RSD itself** (the distillation) rather than domain-finetuning
ResShift — start from the released ×4 teacher; ×2/art finetune is deferred.

- **RSD code is NOT actually released** — `github.com/Daniil-Selikhanovych/RSD` is a
  bare "Coming soon." README placeholder (checked 2026-06-29), despite the abstract
  claiming code. So it **must be reconstructed** from the paper. Digested method
  (§3.2–3.5, Fig 2): it is **DMD2 distribution-matching distillation ported to
  ResShift** — frozen teacher `f*`, trainable one-step student `G_θ` (z_T,z_y,ε→ẑ0),
  trainable "fake" ResShift critic `f_φ` (Prop 3.1 / Eq 10, = training a fake ResShift
  on student outputs with objective Eq 7), a GAN head on `f_φ`'s bottleneck (Eq 12),
  and image-space LPIPS. Generator loss `L_θ + λ1·L_LPIPS + λ2·L_GAN` (Eq 13). `L_θ`
  and `L_GAN` in VQ-f4 latent; LPIPS in pixel space. This is **the same machinery as
  our `turbo` DP-DMD loop** ([[project_daemon_wiring_pattern]], `docs/methods/turbo.md`)
  — fake-critic + GAN + two-timescale alternation — just over ResShift's
  residual-shift forward (Eq 1/4) and SwinUNet+VQGAN instead of flow-matching on the
  Anima DiT. The ResShift repo already ships the forward process, nets, and a BasicSR
  trainer to borrow from. Reconstruction risk is real but bounded.
- **Decisions (2026-06-29):** teacher = **v2 (15-step)** realsr ×4 (paper-faithful);
  **validate benchmark-first** — reproduce the paper's 1-step number on a standard set
  with the released teacher (so a miss is a reconstruction bug, not a domain gap),
  THEN apply to our art. Single-step first; multistep (§3.3) is a later add.
- **SinSR** (Wang et al. 2024, *released code*) stays the safe fallback if the
  reconstruction stalls. RSD reportedly beats SinSR perceptually.
- Build lives in `sr/distill_rsd/`. GPU note: this is the heaviest step — teacher +
  student + fake + discriminator + LPIPS + VQGAN resident at once on 16 GB; lean on
  grad-checkpointing + small microbatch (no block-swap in ResShift). Measure a real
  number on a 1-step dry run before committing.

---

## 4. Repo wiring (minimal, non-invasive)

- New `sr/` top-level (or sibling checkout symlinked in) holding the ResShift
  clone, our configs, and `sr/scripts/`. **Not** an installed package; keep the
  `sys.path` bootstrap pattern used by `bench/` and `scripts/`.
- `tasks.py` gets thin `sr-prep` / `sr-train` / `sr-test` / `sr-distill` dispatch
  targets that shell into the `resshift` env (it cannot run under Anima's torch —
  different pinned versions). Document the env split loudly.
- Outputs under `output/sr/` to mirror the `output/{ckpt,tests}` split.
- **No** changes to `train.py`, `networks/`, `library/`, the config merge chain, or
  the Qwen VAE. If a reviewer sees a diff touching those, the design has drifted.

---

## 5. Risks & honest unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| Photo-domain prior doesn't transfer to flat art | High → it's *why* we finetune | Phase 0 gate measures it directly before committing |
| Real-ESRGAN degradation over-smooths clean art | High | `bicsr`/light-degradation ablation is Phase-1 step 1 |
| RSD code never ships / hard to reconstruct | Med | SinSR (released) is the fallback; Phase 2 is optional |
| Two-env split (resshift vs uv) friction | Med | Isolate fully; `make sr-*` documents the boundary |
| S-Lab non-commercial license | Low–Med | Fine for research; flag before any commercial use |
| ×4 model misused for ×2 → soft output | Low | Train the ×2 config for the 1024→2048 product (§1a) |
| GPU budget for 20–50k-iter finetune | Med | Finetune-from-checkpoint, not from-scratch; single-GPU days |

**Biggest unknown**: whether ResShift's SwinU-Net + VQ-f4 latent has enough
capacity/sharpness for *high-frequency line art* specifically (it was tuned for
natural photos). Phase 0 + the degradation ablation answer this before any large
spend. If line edges stay soft even after finetune, that is the signal to revisit
the heavier EasyControl-SR route (§7) despite its VRAM cost.

---

## 6. Success criteria

- **Phase 0**: released model runs tiled on our images; documented verdict on
  domain gap. (1 day.)
- **Phase 1**: art-SR ×2 checkpoint beating bicubic and the released ResShift on
  our frozen eval set (CLIPIQA/MUSIQ + blind eyeball A/B), tiling 1024→2048 within
  flat VRAM. (Core deliverable.)
- **Phase 2** (optional): 1-step student within perceptual parity of the 4-step
  Phase-1 teacher.

---

## 7. Alternative considered & rejected (for the record)

**EasyControl-SR LoRA on the Anima DiT** (OSEDiff/PiSA-SR style: bicubic-LR as the
EasyControl cond image, LoRA + LPIPS, optionally turbo-distilled to one step).
Reuses *everything* in this repo and inherits Anima's prior — but **cannot reach
2048² in-distribution** (token bands cap at 1536 edge) and is VRAM-heavy (global
attention, extended self-attn). It is the right tool for "modest-res SR that looks
like our model invented the detail," and a *worse* tool for "cheap faithful
high-res upscaling." Kept on the shelf as a distinct future product, not a
competitor to this proposal. Do not conflate the two.
