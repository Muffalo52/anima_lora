# SR sidecar — ResShift ×4/×2 super-resolution

Standalone super-resolution for our art, **deliberately outside the Anima adapter
system** (it has its own VQGAN, latent space, and training loop). Design rationale,
phasing, and the ×2-vs-×4 / degradation decisions live in
[`docs/proposal/resshift_sr_sidecar.md`](../docs/proposal/resshift_sr_sidecar.md).
This README is the ops surface.

## Self-contained: vendored source, no external clone

ResShift's source is **vendored** under [`sr/resshift/`](resshift/) (committed —
`models/`, `ldm/`, `utils/`, `configs/`, `LICENSE`; ~530 KB, basicsr-free). There is
**no external `ResShift/` clone to fetch or patch.** Weights live separately under
`sr/weights/` (gitignored, ~1.6 GB; auto-downloaded by `sr_infer.py` from the v2.0
release if missing).

The sidecar runs in the **root Anima venv** — there is no separate `sr/.venv` anymore.
Its deps are an opt-in dependency group in the root `pyproject.toml`, installed with
`uv sync --group sr` (wrapped by `make sr-setup`). Keeping them in a group rather than
core `dependencies` keeps the heavy metrics closure (`pyiqa` → `opencv-python-headless`,
`bitsandbytes`, `facexlib`, `datasets`, plus dev-tools as runtime deps) out of every
`uv sync`. The old isolated venv was for torch-conflict reasons that no longer exist —
the SR env long ago converged on root's torch (see below), and ResShift is vendored
basicsr-free, added to `sys.path` at import time (venv-independent).

- **Same Blackwell torch as root** (`torch 2.12 + cu132`, cu132 index). Python 3.13
  to match root and kill a version-drift axis.
- **No xformers — and not worth building one.** Its only candidate use here is the
  VQGAN single-head `head_dim=512` mid-attention, which xformers can't accelerate
  anyway; the vendored VQGAN ships **query-chunked exact SDPA** (`ldm/modules/
  diffusionmodules/model.py`, bit-faithful to the trained single-head math, `O(Bq·N)`
  memory) that fixes the OOM, and the UNet's Swin attention gets SDPA-flash for free.
- **No basicsr.** The vendored tree drops it (and the data/inference paths that needed
  it); `sr_infer.py` reimplements released-model inference over the vendored core.

```bash
make sr-setup        # one-time: uv sync --group sr into the root venv + verify
make sr-prep         # build frozen synthetic-LR eval set from image_dataset/ (--n 30)
make sr-phase0       # released ResShift x4 (v3) on eval set + metrics + montages
make sr-test IN=foo.png [OUT=… VERSION=v3 CHOP=512]   # tiled SR on any image/dir
```

`make sr-setup` is idempotent (`uv sync` is). The vendored VQGAN attention patch lives
in the source (committed), so there's nothing to re-apply.

## Phase 0 — verdict (2026-06-29): released model transfers well to our art

Ran released **ResShift ×4 v3 (4-step)** on 30 art images (synthetic LR = bicubic
÷4 of a 1024-long-edge HR), scored vs HR and vs a bicubic baseline:

| metric | ResShift | bicubic |
|---|---|---|
| PSNR ↑ | **27.80** | 25.82 |
| SSIM ↑ | **0.875** | 0.835 |
| LPIPS ↓ | **0.116** | 0.352 |
| MUSIQ ↑ | **73.8** | 41.0 |

ResShift wins **every** axis, hugely on perceptual (LPIPS/MUSIQ). Eyeball
(`sr/data/montage/`, panels = bicubic \| ResShift \| HR): sharp lineart + recovered
hair strands, **no texture hallucination and no color shift** on flat-shaded
regions. The feared photo-prior-vs-art domain gap **did not materialize** — a more
positive result than the proposal expected.

**Implication:** the released model is already usable for clean upscaling. Phase 1
finetune is still worth it but for narrower reasons — (a) a true ×2 (1024→2048)
model, (b) matching our pipeline's actual degradation (the `bicsr`/light-degradation
ablation, proposal §1b), (c) pushing line-edge sharpness — not for closing a large
domain gap. Caveat: this eval used *clean bicubic* LR; the realsr model handling it
well is encouraging, but a `bicsr` checkpoint may do even better on clean input.

Artifacts: `sr/data/{hr_eval,lr_eval,results,montage}/`, `sr/data/phase0_summary.json`.

## Next (Phase 1, gated on the above)

Finetune the **×2** config (`sr/resshift/configs/realsr_realesrgan256_x2.yaml`) from the released
checkpoint on our HR pool; first ablation = `bicsr` vs light-Real-ESRGAN degradation.
The realsr inference path is ×4-only (`assert scale==4`); wiring ×2 inference is a
Phase-1 task. See proposal §3 Phase 1.
