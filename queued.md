# Queued: x4 student re-distill A/B (bare teacher vs finetuned teacher)

Status: **queued** · 2026-07-11

## Background: the `lq` bug (fixed 2026-07-11)
ResShift's UNet takes two LR-derived inputs and they are NOT the same tensor. `z_y` (VQ
latent of the upsampled LR) is the **residual-shift base**; `lq` (model_kwargs) is a
**separate 3-ch conditioning map**, and the released weights want the **LR PIXELS at
latent resolution** there. `sr_infer.py` (since 2026-06-30) and the RSD distill loop
(always) passed the latent instead. The released teacher's x0 prediction under
latent-`lq`: img-MSE vs GT **0.038 / 0.51 / 1.50** at t = 0 / 7 / 14, vs **0.004** flat
with pixel-`lq`. So **the shipped x4 student was distilled from a garbage teacher signal**
— the "student saturated the teacher" plateau is confounded and must be re-tested.

Fixed via `rsd_models.make_cond_lq` + `--cond_lq pixel|latent` (per-version default,
recorded in student ckpt meta). x2 stays `latent` — its teacher AND student were both
trained under it, so it's self-consistent. (The old "bf16 VQ-decode" gotcha was a
**misdiagnosis** — fp32 is equally neon.)

## The A/B (the actual queued work)
Distill an x4 student twice, **identical recipe, only the teacher differs**:

| arm | teacher | command |
|---|---|---|
| A | released v2 (bare) | `make sr-rsd-train VERSION=x4 ARGS="--iters 2000 --bs 4 --compile --save_dir output/sr/rsd_probe_bare"` |
| B | our 30k art finetune | `make sr-rsd-train VERSION=x4ft ARGS="--iters 2000 --bs 4 --compile --save_dir output/sr/rsd_probe_ft"` |

Score both with `sr/distill_rsd/infer.py --eval --ckpt <final> --chop 256` against the
shipped 12k student: **MUSIQ 67.98 · lpips_z0 0.1232 · dc_z0 0.0101**.

- **Arm A beating the shipped student at 2k steps** ⇒ the plateau WAS the bug ⇒ go
  straight to a full-length pixel-`lq` re-distill from the released teacher.
- **Rank on lpips_z0, not MUSIQ.** MUSIQ is no-reference and rewards texture whether or
  not it's real — that trap has already fired twice today (v3 beats v2 on MUSIQ only; the
  x4 finetune beats v2 on CLIPIQA only). A MUSIQ-only win is not a win.
- ~1–1.5 h per arm (2k gen steps, K=5).

## Done 2026-07-11 (don't redo)
**Teacher eval, post-fix, 30-img frozen set** — first honest measurement of either:

| teacher | MUSIQ | CLIPIQA | LPIPS | PSNR | SSIM |
|---|---|---|---|---|---|
| released v2 (15-step) | 66.92 | **0.742** | **0.0780** | **30.13** | 0.9104 |
| released v3 (4-step) | **68.35** | 0.702 | 0.0956 | 29.92 | **0.9146** |

v3's "68.4 wins" from the old draft was **MUSIQ-only** — v2 wins every fidelity metric
(LPIPS 19% better) and the other no-reference metric too. **v2 is the better teacher.**

**x4 teacher finetune (30k, `make sr-train VERSION=x4`) — DONE, and it FAILED.** Wiring
is shipped and works (`sr/train_sr/` parametrized by `--version {x2,x4,x4s4}`,
`configs/realsr_x4{,_s4}_art.yaml`, strict 564/564 warm-start), but the resulting teacher
is **worse than the released v2 it started from**:

| model | MUSIQ | CLIPIQA | LPIPS | PSNR | SSIM |
|---|---|---|---|---|---|
| released v2 (start point) | **66.92** | 0.742 | **0.0780** | **30.13** | **0.9104** |
| x4ft 5k / 15k / final | 62.97 / 63.64 / 64.48 | 0.772 / 0.764 / 0.763 | 0.1683 / 0.1422 / **0.1129** | 29.43 / 29.58 / 29.32 | 0.8394 / 0.8593 / 0.8758 |

Training moved the model AWAY from its init and spent 30k steps crawling back without
reaching it. It wins only CLIPIQA ⇒ it learned **texture, not detail** (visible as speckle
on flat fills in the step-17500 montage). Ckpt at `output/sr/x4_art/resshift_x4_final.pth`
— usable as the Arm-B teacher, **not** shippable.

Prime suspect if anyone retries: `--scale_jitter_max` auto-widened to **[4, 8]** at sf=4
(a quarter of every batch degraded 4–8× and asked to reconstruct the same GT = trained
hallucination), while the eval set is *clean* bicubic ×4. Secondary: `--lambda_lpips 0.5`,
`--lr 5e-5`. **Do not retry Stage 1 before the A/B answers whether a better teacher is
even needed.**

## Open
- **x2 redo?** Gated on the A/B. x2 is self-consistent (not broken), but it wasted its
  warm-start re-learning the conditioning ("neon → coherent by ~400 steps" in its log).
  Its plateau ("24k ≈ 2k") is NOT confounded, so a pixel-`lq` retrain may just re-derive
  the same checkpoint at the cost of a teacher run + re-distill + node republish.


# Queued: EasyControl colorize — fix yellowish/sepia tone drift


---



# Queued: warm-start from official turbo-extracted lora for faster distillation (nfe=2 4k)

# Queued: warm-start from official turbo-extracted lora for faster distillation (nfe=4 4k Revised LR)