# Queued: x4 teacher finetune → x4 student re-distill

Status: **queued / not started** · drafted 2026-07-09

## Goal
Lift the 1-step x4 RSD student by first lifting its teacher: domain-finetune the
released x4 ResShift teacher on our HR art pool, then re-distill the x4 student from
the finetuned teacher. Mirrors what the x2 line already did (`make sr-train` +
`sr/configs/realsr_x2_art.yaml`), one scale up.

## Why (evidence, 2026-07-09, 30-img frozen eval)
The current 1-step student has **nearly saturated the teacher** — so the student can't
improve without a better teacher.

| Model | Steps | MUSIQ ↑ | vs-HR |
|---|---|---|---|
| Teacher v3 (phase0, valid path) | 4 | 68.4 | LPIPS 0.095 · SSIM 0.91 · PSNR 29.9 |
| Student prev-12k (shipped) | 1 | 68.0 | — |
| Student new-9000 (killed run) | 1 | 67.5 | — |

- Retraining the student longer does nothing: 9k ≈ shipped 12k (plateau, same as the
  x2 "24k ≈ 2k" result). Keep shipped **12k**; the killed run earned no swap.
- Teacher already *transfers* to art (LPIPS 0.095), so finetune headroom may be
  **modest** → run a Phase-0 probe first, do NOT commit to a long run blind.

## Plan
**Stage 1 — finetune x4 teacher on art**
- Warm-start released x4 (`sr/weights/resshift_realsrx4_s15_v2.pth`; consider **v3 s4**
  too — it scored higher on art, 68.4).
- Train on the HR art pool (`sr/data/rsd_hr_cap2048` / `hr_pool`), realesrgan-style
  degradation, at **sf=4**.
- Wiring gap: `make sr-train` + `sr/train_x2/train.py` are **x2-only** (hardcoded
  `sr/configs/realsr_x2_art.yaml`, sf=2). Need an **x4 analog**: an `realsr_x4_art.yaml`
  (= released x4 config, sf=4, trainer/degradation sections kept) + an x4 training
  entry (mirror train_x2, or parametrize it by `--sf`/`--config`).
- Open Q: keep 15-step (v2) vs 4-step (v3) teacher schedule for the finetune target.

**Stage 2 — re-distill x4 student from the finetuned teacher**
- `make sr-rsd-train` with `--version x4 --teacher <finetuned_x4_ckpt>` (must pass
  `--teacher` explicitly — see footgun below).
- Settings that worked today: `--compile --bs 4 --save_every 4500`; **~9k steps is
  plenty** (plateaus early — don't pay for 18k).
- Eval each ckpt against the 30-img set (MUSIQ + `dc_z0`/`lpips_z0`) and eyeball a
  montage. Beat the shipped 12k (68.0 MUSIQ) to justify shipping.

## Gotchas found today (fix or remember)
1. **`--teacher` x4 footgun**: `train.py`'s `--teacher` default points at the *x2*
   checkpoint, and `resolve_version` uses it whenever non-empty — so `--version x4`
   without an explicit `--teacher` silently distills from the **x2 teacher**. Always
   pass `--teacher` for x4 (or fix the default to empty and let the resolver own it).
2. **`sr_infer.py` / `make sr-test` color bug**: `decode_first_stage` runs the VQ
   codebook lookup **under `torch.autocast(bf16)`** (sr_infer.py ~L195); bf16 breaks
   the nearest-code argmin → structure preserved, colors scrambled (neon green/magenta).
   The clean student path uses `force_not_quantize=True`. Fix = force_not_quantize (or
   fp32 decode) in the sr_infer decode. `--no_amp` alone OOMs at fp32 on the 16 GB card.
   Teacher evals here therefore used the **phase0 path** (`make sr-phase0`), which is clean.

## Eval assets (this session, under `output/sr/rsd/`)
- `eval_new9000/`, `eval_prevfinal/` — student evals (summaries have MUSIQ + dc_z0/lpips_z0)
- `eval_teacher_v2/`, `eval_teacher_v3/` — **BROKEN** teacher runs (bf16 bug; don't trust)
- `montage_teacher_vs_students.png`, `montage_v2_v3_check.png` — visual proof
- Valid teacher baseline: `sr/data/phase0_summary.json` (v3, MUSIQ 68.4)

---

# Queued: EasyControl colorize — fix yellowish/sepia tone drift


---



# Queued: warm-start from official turbo-extracted lora for faster distillation (nfe=2 4k)

# Queued: warm-start from official turbo-extracted lora for faster distillation (nfe=4 4k Revised LR)