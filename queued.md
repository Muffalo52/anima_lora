# Queued: merge-interference probe Phase 0 (calibration bench)

`docs/proposal/merge_interference_probe.md` — noise band + N-scaling +
overlapping-artist information cell (~1.5h GPU, existing ckpts, no training).
Gates the `check_merge.py` ship; kill if Tier-1 geometry predicts everything.

# Queued: x4 student re-distill using finetuned teacher (it beats the bare teacher)


# Queued: EasyControl colorize — fix yellowish/sepia tone drift


---



# Queued: warm-start from official turbo-extracted lora for faster distillation (nfe=2 4k)

# Queued: turbo _T series follow-up — div 0.05 + shift 3 + 8k long run (warm start)

**Verdict from the _T sweep** (T750 / T / T_LR, all warm-started from turboV10 ASVD r96):
longer training pays exactly when the per-step objective is healthy — and `div_weight=0.1`
made it unhealthy.

- **T_LR** (div 0.1, lr 2e-5/3e-5): renders **1k ≫ 4k**. Logs agree: `div_loss` floors at
  0.070 from step 2k (doubled anchor = binding constraint), grinding step-0 toward the
  half-resolved teacher target all back half. Early ckpts ≈ warm-start polish; late = washed.
- **T** (div 0.05, lr 5e-5/5e-5): renders **1k ≪ 4k** — refinement accumulates when the
  anchor isn't over-weighted. 5e-5 fully stable under warm start (old 2e-5 instability
  threshold is a cold-start fact).
- **_S** (cold, 7.2k): real strength ⇒ objective ceiling is past 4k.
- **flow_shift A/B** (T_LR@4k, 8 matched seed pairs, shift 2 vs 3 render): parity, mild
  edge to 3 (shift-2 side had the anatomy fumbles). Off-grid render is NOT the manga
  fidelity culprit ⇒ standardize on 3.0 (= stock ComfyUI Anima `sampling_settings`,
  `INFERENCE_BASE`; anchor returns to σ=0.75 with k_anchor=6 untouched).

**Run config** (deltas vs current turbo.toml):
```toml
[sampling]  flow_shift = 3.0
[dpdmd]     div_weight = 0.05
[optim]     student_lr = 5e-5
            fake_lr = 5e-5
iterations = 8000
save_every = 1000
```
Keep `student_init_weights`/`fake_init_weights` (warm start: free stability, init washes
out by 8k anyway).

**Eval protocol**: rank ckpts 1k–8k by rendered 4-step grids ONLY (`--flow_shift 3`,
cfg 1.0) on the failing caststation prompts + sweetonedollar seed sweep. Watch for
caption-adherence slip at 6k–8k (known DP-DMD div_gap erosion — series val peaked early);
if it shows, the prepared counter is `[softrank] weight = 0.05` (off by default), NOT a
shorter run.

**Optional second arm** (after primary): same settings, cold start — clean test of whether
_S's strength was the long schedule or freedom from the V10 basin.
