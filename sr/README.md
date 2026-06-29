# SR sidecar — ResShift ×4/×2 super-resolution

Standalone super-resolution for our art, **deliberately outside the Anima adapter
system** (it has its own VQGAN, latent space, and training loop). Design rationale,
phasing, and the ×2-vs-×4 / degradation decisions live in
[`docs/proposal/resshift_sr_sidecar.md`](../docs/proposal/resshift_sr_sidecar.md).
This README is the ops surface.

## Hard rule: isolated env (NOT Anima's torch)

ResShift lives in its **own** venv `sr/.venv` and never imports Anima's torch.
The proposal assumed ResShift's pinned `torch==2.1.1` + `xformers==0.0.23`; that is
**wrong for this box** — the GPU is an **RTX 5070 Ti (Blackwell, sm_120)** and those
wheels have no Blackwell kernels. So the real env is:

- **Modern Blackwell torch** (`torch 2.12.1+cu132`, installed via the cu128 index).
- **No xformers** — it's optional in ResShift; we use a query-chunked SDPA fallback
  instead (see patches below). Avoids a Blackwell xformers source build.
- `basicsr` needs a one-line shim (`torchvision.transforms.functional_tensor` was
  removed) — applied by `setup_env.sh`.

```bash
make sr-setup        # one-time: create sr/.venv, install deps, patch basicsr + ResShift
make sr-prep         # build frozen synthetic-LR eval set from image_dataset/ (--n 30)
make sr-phase0       # released ResShift x4 (v3) on eval set + metrics + montages
make sr-test IN=foo.png [OUT=… SCALE=4 VERSION=v3 CHOP=512]   # tiled SR on any image/dir
```

`make sr-setup` is idempotent; re-run it after re-cloning ResShift to re-apply the
source patches.

## ResShift source patches (ResShift/ is a gitignored clone)

Because `ResShift/` is an upstream clone (gitignored), our edits don't ride in git
history — `sr/scripts/patch_resshift.py` re-applies them idempotently:

- **VQGAN vanilla `AttnBlock`** (`ldm/modules/diffusionmodules/model.py`): the
  single-head, `head_dim=512`, ~65k-token mid-attention has **no** flash/efficient
  SDPA kernel (head_dim > 256), so plain SDPA falls to the math backend and OOMs on
  a 16 GiB attention map. Replaced with **query-chunked exact SDPA** — bit-faithful
  to the trained single-head math, `O(Bq·N)` memory.
- `basicsr/data/degradations.py` import shim — applied in-venv by `setup_env.sh`.

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

Finetune the **×2** config (`configs/realsr_realesrgan256_x2.yaml`) from the released
checkpoint on our HR pool; first ablation = `bicsr` vs light-Real-ESRGAN degradation.
The realsr inference path is ×4-only (`assert scale==4`); wiring ×2 inference is a
Phase-1 task. See proposal §3 Phase 1.
