# Queued: x4 teacher finetune → x4 student re-distill

Status: **queued / not started** · drafted 2026-07-09

## Goal
Lift the 1-step x4 RSD student by first lifting its teacher: domain-finetune the
released x4 ResShift teacher on our HR art pool, then re-distill the x4 student from
the finetuned teacher. Mirrors what the x2 line already did (`make sr-train` +
`configs/realsr_x2_art.yaml`), one scale up.

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
  `configs/realsr_x2_art.yaml`, sf=2). Need an **x4 analog**: an `realsr_x4_art.yaml`
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

Status: **queued / not started** · drafted 2026-07-09

## Symptom
Colorize adapter (`anima_colorize`, retrained on the full corpus after
`only_data_includes` was widened from a comic-page tag set to `[]`) renders
uniformly sepia/warm-cream at inference — both on manga-page conds AND on
clean illustration conds (e.g. chicke_iii refs), with empty prompt (the
workflow's default mode).

## Root cause (measured, not guessed — see `scratchpad/colorize_target_stats.py`
and `colorize_target_stats.json` from this session)
The colorize training TARGET pool (`post_image_dataset/resized`, paired via the
staged cond tree `post_image_dataset/easycontrol/colorize/staging/`, 2,951
pairs) is corpus-wide warm/desaturated — tag-based filtering can't see this
because the tagger doesn't emit tone tags for this corpus (only 1/2951 image
tagged `monochrome`, 0 tagged `sepia`):

| Metric | Value |
|---|---|
| Median per-image mean HSV saturation | **0.182** |
| Targets with mean sat < 0.35 | **94.4%** (mirrors the sanitize-adapter wash-out finding, 94% < 0.35 sat) |
| Bright-pixel white-point R−B (warm cast) | mean **+0.089**, 84.7% of images > +0.02 |
| Sepia-like (sat<0.20 & warm-dominant) | **33.0%** of targets (n=974; artist clusters `coro_fae`, `sincos`, `ama_mitsuki`, `oldsickkim`) |

Even the corpus's biggest/cleanest illustration artists skew warm (sincos
n=334 sat=0.151/91% warm; ama_mitsuki n=106 sat=0.116/90% warm). The rendered
outputs land almost exactly on the training-target white-point (measured
+0.11…+0.16 vs. corpus p75 +0.137), vs. the old comic-only checkpoint's clean
+0.022 — i.e. this is the adapter faithfully reproducing its target
distribution's mean tone, reinforced by `caption_dropout_rate = 0.8` collapsing
most gradient into one shared unconditional colorization mode. Cond content
(manga page vs. illustration) doesn't gate tone — text does (color tags only,
and only 20% of the time), so illustrations get the same corpus-mean pull.

**Filtering is not viable** — the low-sat tail is 33-94% of the data depending
on threshold; dropping it guts the dataset. Fix must be **correction**, not
exclusion.

## Plan
1. **White-balance the targets at prep time.** Add a target-encode stage to
   `easycontrol_adapters/colorization/prep.py` that neutralizes each target's
   white-point (scale channels so bright-region R≈G≈B; gray-world clamp for
   images without a clear bright/paper region) and VAE-encodes the corrected
   image into a **colorize-specific target latent cache** (new subset-level
   cache dir, same pattern as `cond_cache_dir`/`text_cache_dir`) — targets
   currently ride the shared `post_image_dataset/lora` cache, which is why they
   can't be touched today without affecting every other method.
2. **Handle the true-sepia tail separately** (mean sat ≲ 0.10–0.15, ~13-34% of
   the set): white-balancing a near-mono image just yields flat gray, a bad
   colorize target. Either drop these specifically (small, targeted — not the
   blanket 94% cut) or keep them but inject a computed `muted color`/`sepia`
   caption tag for the borderline band.
3. **If (2) uses tag-binding**: note `caption_dropout_rate = 0.8` drops the
   *whole* caption, so a tagged muted image still trains empty-prompt mode 80%
   of the time — tone tags only separate modes if given a lower dropout rate or
   exempted from caption_dropout_rate, mirroring how `text_keep_comic` /
   `text_keep_copyright` are protected-prefix + dropout-immune.
4. **Optional palette-steering feature**: keep the **artist tag** in the
   protected/dropout-immune prefix alongside copyright — lets `chicke_iii` (or
   any artist) in the prompt pull toward that artist's palette instead of the
   corpus-mean tone. Separate from the white-balance fix, additive.
5. **Re-run staging + preprocess + train** (`make easycontrol-staging` →
   `easycontrol-preprocess` → `easycontrol EASYADAPTER=colorize`) once the
   target cache change lands; re-render the same comfy colorize test prompts
   and re-measure white-point/saturation the same way to confirm convergence
   (compare against old clean checkpoint's +0.022 white-point baseline).

## Inference-side mitigation (cheap, try first, don't expect much)
`workflows/colorize.json` currently has an empty negative prompt (only
`quality_neg` set). Add `sepia, muted color, limited palette` to the negative
and/or `colorful` to positive — but since these tokens never appear in the
colorize text channel's training vocab (color-tag-only filter), only the base
model's CFG prior can act on them, so expect a weak effect at best. Useful as
a diagnostic probe (if it measurably helps, corroborates the corpus-mean
explanation) but not a substitute for the data-side fix.

## Assets (this session)
- `scratchpad/colorize_target_stats.py` — color-statistics script (HSV sat,
  bright-pixel white-point R−B, warm-fraction) over the staged colorize pair
  set; walks `post_image_dataset/easycontrol/colorize/staging/` and pairs
  against `post_image_dataset/resized/`.
- `scratchpad/colorize_target_stats.json` — per-image stats dump for all 2,951
  pairs (re-slice by artist/band as needed).

---



# Queued: warm-start from official turbo-extracted lora for faster distillation
