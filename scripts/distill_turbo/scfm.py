"""SCFM — velocity-space self-distillation loop (turbo ``base_loss="scfm"``).

Implements the Phase-1 minimal port of *Shortcutting Pre-trained Flow Matching
Diffusion Models is Almost Free Lunch* (Cai et al., NeurIPS 2025) as a selectable
turbo objective. Two terms, mixed per-sample by ``k_ratio`` (Bernoulli per step
at the turbo default ``batch_size=1``):

- **Term A — teacher rectification.** A renoised real latent's coarse-step
  velocity is pinned to the CFG-guided teacher field (the quality ceiling).
- **Term B — velocity self-consistency.** One coarse Euler step on a stop-grad
  EMA copy of the student (θ⁻) must equal two finer sub-steps → the trajectory
  straightens, which is what makes few-step Euler work. ``term_b_point`` picks
  where the consistency point comes from: ``"renoise"`` (paper-faithful) uses an
  on-manifold renoised real latent — where Anima's field is already straight, so
  Term B measures ~0 and does nothing; ``"rollout"`` rolls θ⁻ from noise on its
  own coarse student grid to the OFF-manifold states the few-step Euler rollout
  actually visits (the washout source), where the consistency is non-trivial.

The fake/critic stack is repurposed as θ⁻ (``set_view("fake")``); it carries no
optimizer and is updated by parameter EMA (Eq. 14) with a cyclic restart. No
critic, no GAN, no diversity anchor — the saved artifact is a plain velocity
LoRA, identical inference to the DP-DMD student.

Design, decision gates, and the Phase-0/Phase-1 progress log all live in
``docs/proposal/turbo_scfm.md`` (§9). Driven from
``scripts/distill_turbo/distill.py::main`` under ``base_loss="scfm"``; reuses that
file's loaded DiT, compile, dataloader, ``_forward`` closure, and σ grids.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from library.anima.uncond import uncond_for_batch
from library.inference.sampling import get_timesteps_sigmas

from .diversity import run_diversity_validation
from .primitives import make_scheduler, renoise, sample_t

logger = logging.getLogger(__name__)


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, else raw count (mirror distill.py)."""
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


def _velocity_mse(
    v_pred: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Mean-squared velocity error, optionally foreground-masked.

    ``mask`` is ``[B, 1, H, W]`` (broadcast over the ``[B, C, H, W]`` channels);
    the masked variant is a per-element mean over the kept region so its scale
    matches the unmasked ``.mean()`` (``use_masked_loss`` parity with the DMD
    path). Both operands are upcast to fp32 — the loss is assembled in fp32 like
    every other turbo objective.
    """
    sq = (v_pred.float() - v_target.float()) ** 2
    if mask is None:
        return sq.mean()
    # denom = kept spatial cells × B × channels (mask has 1 channel).
    denom = mask.sum() * sq.shape[1] + 1e-8
    return (sq * mask).sum() / denom


def run_scfm(
    *,
    cfg,
    turbo,
    model,
    forward_fn,
    teacher_cfg_velocity_fn,
    dataloader,
    student_sigmas: list[float],
    uncond_base: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    writer,
    val_cond,
    val_latent_shape,
    val_clean,
) -> None:
    """Run the SCFM distillation loop to completion (saves like the DMD path).

    All heavy setup (DiT load, block-swap, compile, dataloader, the ``_forward``
    closure, σ grids, TB writer, config snapshot) is done by the caller and
    passed in — this owns only the SCFM-specific optimizer, EMA bookkeeping,
    per-step loss, metrics, validation, and checkpoint saves.
    """
    # Student optimizer (the EMA stack has none). Mirrors the DMD student opt.
    student_opt = torch.optim.AdamW(
        turbo.student_params(),
        lr=cfg.student_lr,
        weight_decay=cfg.weight_decay,
        fused=torch.cuda.is_available(),
    )
    student_sched = make_scheduler(student_opt, cfg.iterations, cfg.student_lr)

    # EMA student θ⁻ := the fake stack. Detach it from autograd (no optimizer)
    # and seed θ⁻ ← θ exactly. Both stacks are zero-init (LoRA up = 0) so this is
    # 0 ← 0 at start, but the explicit reset keeps the invariant honest.
    turbo.freeze_ema()
    turbo.reset_ema()

    n_student = sum(p.numel() for p in turbo.student_params())
    logger.info(
        f"SCFM: student={n_student:,} trainable params; EMA stack frozen "
        f"(μ={cfg.scfm_ema_mu}, restart every {cfg.scfm_ema_restart or 'never'})"
    )

    # Finer consistency grid Term B samples adjacent (t_i, t_i+1, t_i+2) triples
    # from. n+1 points, σ: 1 → 0; valid triple-start indices are 0 … n-2.
    grid = get_timesteps_sigmas(cfg.scfm_n_consistency_grid, cfg.flow_shift, "cpu")[
        1
    ]  # [n+1]
    grid_dev = grid.to(device=device, dtype=dtype)
    n_grid = cfg.scfm_n_consistency_grid
    logger.info(
        f"SCFM consistency grid ({n_grid} sub-steps, flow_shift={cfg.flow_shift}): "
        f"σ={['%.3f' % s for s in grid.tolist()]}; student grid "
        f"σ={['%.3f' % s for s in student_sigmas]}"
    )

    # term_b_point="rollout": Term B's consistency point is rolled along the COARSE
    # student grid (the inference σ-grid) from noise on θ⁻, landing on the
    # off-manifold states the few-step Euler rollout actually visits — where the
    # washout lives and where the on-manifold renoise variant has nothing to
    # straighten (turbo_scfm.md §9.1/§9.4). Each coarse step is sub-divided once at
    # its midpoint for the "1 big step == 2 half-steps" target.
    rollout_mode = cfg.scfm_term_b_point == "rollout"
    student_sig_dev = torch.tensor(student_sigmas, device=device, dtype=dtype)
    n_stu = len(student_sigmas) - 1  # number of coarse Euler steps (= student_steps)
    if rollout_mode:
        logger.info(
            f"SCFM Term B = OFF-trajectory rollout on the {n_stu}-step student grid "
            "(θ⁻ from noise → coarse-step midpoint consistency); n_consistency_grid "
            "is renoise-mode-only and unused."
        )

    # CPU RNG for the per-step role draw + grid-index draw — keeps both off the
    # GPU sync path (mirrors the DMD loop's grad_step='random' g draw), seeded
    # for reproducibility alongside the global torch.manual_seed(cfg.seed).
    cpu_gen = torch.Generator()
    # Offset from the global seed so the role/grid draws don't alias the data
    # shuffle or noise streams that also key off cfg.seed.
    cpu_gen.manual_seed(cfg.seed + 9973)

    # Logging accumulators (GPU tensors flushed every log_interval → no per-step
    # .item() sync). Counts are CPU ints (role drawn on CPU), also sync-free.
    loss_sum = torch.zeros((), device=device)
    lossA_sum = torch.zeros((), device=device)
    lossB_sum = torch.zeros((), device=device)
    residB_sum = torch.zeros((), device=device)  # Term-B straightening headroom
    nA = nB = nlog = 0
    # Last non-empty per-term values for the tqdm postfix. At high k_ratio a whole
    # log window can be all-Term-A (nB=0), so the live bar would flicker to nan;
    # carry the most recent real value instead (TB curves stay gated below).
    last_a = last_b = float("nan")

    data_iter = iter(dataloader)
    use_masked = cfg.use_masked_loss
    accum = cfg.gradient_accumulation_steps
    logger.info(
        f"SCFM grad accumulation: {accum} micro-step(s)/optimizer step "
        f"(effective batch {cfg.batch_size * accum}). "
        + (
            "Term A / Term B mix WITHIN each optimizer window."
            if accum > 1
            else "single micro-step (pure-A-or-B per step)."
        )
    )
    progress = tqdm(range(cfg.iterations), desc="turbo-scfm")

    for step in progress:
        # One optimizer step = `accum` micro-steps accumulated. Zero once up front;
        # each micro-step's loss is scaled by 1/accum so the summed grad is the
        # MEAN over the effective batch (LR is accum-invariant). At batch_size=1
        # the per-micro Bernoulli role draw mixes Term A / Term B across the window
        # → the optimizer step sees both terms (the paper's batched k/N mix).
        student_opt.zero_grad(set_to_none=True)

        for _micro in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            latents = batch["latents"].to(device, dtype=dtype, non_blocking=True)
            crossattn_emb = batch["crossattn_emb"].to(
                device, dtype=dtype, non_blocking=True
            )
            B = latents.shape[0]
            if use_masked:
                mask = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)
            else:
                mask = None

            c_null = uncond_for_batch(uncond_base, crossattn_emb)

            torch.compiler.cudagraph_mark_step_begin()

            # --- per-sample role: Bernoulli(k_ratio). At B=1 this is a per-MICRO
            # coin flip (deterministic threshold int(round(k_ratio*B)) would be 0
            # at B=1 → never Term A; Bernoulli generalizes the "k/N of a batch"
            # ratio to any B, expected Term-A fraction = k_ratio). Drawn on CPU
            # (no GPU sync). Over `accum` micro-steps the window mixes both roles.
            is_termA_cpu = torch.rand(B, generator=cpu_gen) < cfg.scfm_k_ratio  # bool
            is_termA = is_termA_cpu.to(device)
            any_A = bool(is_termA_cpu.any())
            any_B = bool((~is_termA_cpu).any())
            is_termA_b = is_termA.view(B, *([1] * (latents.dim() - 1)))  # broadcast

            # --- consistency grid (t_lo, t_mid, t_hi) + Term-B point x_t --------
            # Term A always renoises a real latent at a random level t_a (→ teacher
            # rectification target, on-manifold). Term B enforces the "1 step over
            # [t_lo, t_hi] == 2 sub-steps via t_mid" velocity consistency at its
            # point. The two modes differ only in where that point and grid come
            # from:
            #   renoise : t_lo/t_mid/t_hi = an adjacent triple of the fine
            #             consistency grid; Term-B point = renoise(real, t_lo).
            #   rollout : t_lo/t_mid/t_hi = one COARSE student step sub-divided at
            #             its midpoint; Term-B point = θ⁻ rolled from noise to t_lo
            #             (off-manifold — the states the rollout actually visits).
            t_a = sample_t(
                B,
                distribution=cfg.t_distribution,
                sigmoid_scale=cfg.sigmoid_scale,
                device=device,
                dtype=dtype,
            )
            view_b = (B, *([1] * (latents.dim() - 1)))
            x_rollout = None
            if rollout_mode:
                # One coarse interval per micro-step, shared across the batch.
                j = int(torch.randint(0, n_stu, (1,), generator=cpu_gen).item())
                s_lo = student_sig_dev[j]
                s_hi = student_sig_dev[j + 1]
                s_mid = 0.5 * (s_lo + s_hi)
                t_lo = s_lo.expand(B)
                t_mid = s_mid.expand(B)
                t_hi = s_hi.expand(B)
                # Roll θ⁻ from noise to σ_j (j Euler steps, all no-grad) → the
                # off-manifold visited state. j=0 ⇒ the pure-noise start (σ=1).
                if any_B:
                    x_rollout = torch.randn_like(latents)
                    for kk in range(j):
                        sk = student_sig_dev[kk]
                        sk1 = student_sig_dev[kk + 1]
                        vk = forward_fn(
                            "fake",
                            x_rollout,
                            sk.expand(B),
                            crossattn_emb,
                            no_grad=True,
                        ).squeeze(2)
                        x_rollout = (x_rollout - (sk - sk1) * vk).detach()
            else:
                # Adjacent triple of the fine consistency grid (per-sample index).
                idx = torch.randint(0, max(1, n_grid - 1), (B,), generator=cpu_gen)
                t_lo = grid_dev[idx]
                t_mid = grid_dev[idx + 1]
                t_hi = grid_dev[idx + 2]
            d_i = (t_lo - t_mid).view(view_b)
            d_ip1 = (t_mid - t_hi).view(view_b)

            # Student/teacher evaluation level: t_a for Term A, t_lo for Term B.
            t_samp = torch.where(is_termA, t_a, t_lo)  # [B]

            # Renoise real at t_samp (Term-A point, and the renoise-mode Term-B
            # point). In rollout mode the Term-B positions are then overwritten
            # with the off-manifold rollout state.
            eps = torch.randn_like(latents)
            x_t = renoise(latents, t_samp, eps)
            if rollout_mode and x_rollout is not None:
                x_t = torch.where(is_termA_b, x_t, x_rollout)

            # --- Term A target: CFG-guided teacher velocity (no grad) -----------
            v_tea = None
            if any_A:
                v_tea = teacher_cfg_velocity_fn(x_t, t_samp, crossattn_emb, c_null).to(
                    dtype
                )

            # --- Term B target: two EMA sub-steps on the stop-grad student θ⁻ ---
            v_target_B = None
            resid_B = None
            if any_B:
                v1 = forward_fn(
                    "fake", x_t, t_samp, crossattn_emb, no_grad=True
                ).squeeze(2)
                x_next = x_t - d_i * v1  # Euler sub-step lo → mid
                v2 = forward_fn(
                    "fake", x_next, t_mid, crossattn_emb, no_grad=True
                ).squeeze(2)
                denom = (d_i + d_ip1).clamp_min(1e-8)
                v_target_B = ((d_i * v1 + d_ip1 * v2) / denom).detach()
                # Straightening headroom: how far the coarse one-step velocity v1
                # is from the two-sub-step composition (relative). Near 0 ⇒ field
                # already straight at this point (Term B saturated); large ⇒ room.
                # In rollout mode this is measured at the OFF-manifold visited
                # state, so a non-trivial value here is the signal Term B now bites.
                resid_B = (
                    (v1 - v_target_B).float().norm()
                    / (v_target_B.float().norm() + 1e-8)
                ).detach()

            # Combine per-sample targets. Where only one branch ran, fill the
            # other side with that same tensor so torch.where has a valid operand
            # (the filled side is masked out by is_termA_b anyway).
            if v_tea is None:
                v_tea = v_target_B
            if v_target_B is None:
                v_target_B = v_tea
            target = torch.where(is_termA_b, v_tea, v_target_B).detach()

            # --- student grad forward at the SAME (x_t, t_samp) -----------------
            x_t.requires_grad_()  # grad-ckpt safety; harmless when ckpt off
            turbo.set_student_step(0)  # no-op (single-head student)
            v_stu = forward_fn(
                "student", x_t, t_samp, crossattn_emb, no_grad=False
            ).squeeze(2)
            loss = _velocity_mse(v_stu, target, mask)

            # Scale so accumulated grad = mean over the `accum`-micro-step window.
            (loss / accum).backward()

            # --- metrics (UNSCALED micro loss; flush at log_interval) -----------
            loss_sum = loss_sum + loss.detach()
            nlog += 1
            nA_micro = int(is_termA_cpu.sum())
            nB_micro = B - nA_micro
            if nA_micro and nB_micro == 0:
                # Pure Term-A micro-step (the only A case at B=1): loss IS A's.
                lossA_sum = lossA_sum + loss.detach()
                nA += 1
            if nB_micro and nA_micro == 0:
                # Pure Term-B micro-step (the only B case at B=1): loss IS B's and
                # resid_B stays in lockstep with the nB counter.
                lossB_sum = lossB_sum + loss.detach()
                if resid_B is not None:
                    residB_sum = residB_sum + resid_B
                nB += 1

        # --- optimizer step (once per `accum` micro-steps) ---------------------
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                turbo.student_params(), max_norm=cfg.grad_clip
            )
        student_opt.step()
        student_sched.step()

        # --- EMA update θ⁻ ← μθ⁻ + (1−μ)θ (Eq. 14) + cyclic restart -------------
        # Per OPTIMIZER step: θ⁻ is held fixed across the micro-step window so all
        # Term-B targets in a window share one stop-grad teacher (restart counts
        # optimizer steps, matching ema_restart's intent).
        turbo.update_ema(cfg.scfm_ema_mu)
        if cfg.scfm_ema_restart and (step + 1) % cfg.scfm_ema_restart == 0:
            turbo.reset_ema()

        if (step + 1) % cfg.log_interval == 0:
            scalars = {
                "train/scfm_loss": (loss_sum / max(1, nlog)).item(),
                "train/scfm_termA_frac": nA / max(1, nlog),
                "train/student_lr": student_sched.get_last_lr()[0],
            }
            if nA:
                scalars["train/scfm_loss_a"] = (lossA_sum / nA).item()
                last_a = scalars["train/scfm_loss_a"]
            if nB:
                scalars["train/scfm_loss_b"] = (lossB_sum / nB).item()
                scalars["train/scfm_consistency_residual"] = (residB_sum / nB).item()
                last_b = scalars["train/scfm_loss_b"]
            if writer is not None:
                for k, v in scalars.items():
                    writer.add_scalar(k, v, step + 1)
            progress.set_postfix(
                loss=f"{scalars['train/scfm_loss']:.4f}",
                a=f"{last_a:.4f}",
                b=f"{last_b:.4f}",
            )
            loss_sum = torch.zeros((), device=device)
            lossA_sum = torch.zeros((), device=device)
            lossB_sum = torch.zeros((), device=device)
            residB_sum = torch.zeros((), device=device)
            nA = nB = nlog = 0

        # --- diversity validation (DAVE same-prompt probe; structure-sensitive
        # checkpoint signal — CMMD is blind to pose, §5.5) ----------------------
        if val_cond is not None and (step + 1) % cfg.validate_every_n_steps == 0:
            dm = run_diversity_validation(
                model=model,
                forward_fn=forward_fn,
                set_student_step=turbo.set_student_step,
                student_sigmas=student_sigmas,
                crossattn_emb=val_cond,
                latent_shape=val_latent_shape,
                num_seeds=cfg.val_diversity_seeds,
                seed0=cfg.seed,
                device=device,
                dtype=dtype,
                clean_latent=val_clean,
            )
            if writer is not None:
                writer.add_scalar("val/div_ac_sim", dm.ac_sim, step + 1)
                writer.add_scalar("val/div_dc_sim", dm.dc_sim, step + 1)
                writer.add_scalar("val/div_gap", dm.gap, step + 1)
                writer.add_scalar("val/div_xpred_ac_sim", dm.xpred_ac_sim, step + 1)
                writer.add_scalar("val/fm_mse", dm.fm_mse, step + 1)
            logger.info(
                f"[val@{step + 1}] diversity: AC sim={dm.ac_sim:.4f} "
                f"(lower=more diverse) | DC sim={dm.dc_sim:.4f} | gap={dm.gap:+.4f} "
                f"| x_pred AC sim={dm.xpred_ac_sim:.4f} | FM MSE={dm.fm_mse:.4f}"
            )

        # --- checkpoint (mirror the DMD save: step-tagged subdir + final bare) --
        if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
            n = step + 1
            is_final = n == cfg.iterations
            metadata = {
                "ss_turbo_objective": cfg.base_loss,  # "scfm"
                "ss_turbo_student_rank": str(cfg.student_rank),
                "ss_turbo_student_alpha": str(cfg.student_alpha),
                "ss_turbo_student_steps": str(cfg.student_steps),
                "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                "ss_turbo_step": str(n),
                "ss_scfm_k_ratio": str(cfg.scfm_k_ratio),
                "ss_scfm_ema_mu": str(cfg.scfm_ema_mu),
                "ss_scfm_n_consistency_grid": str(cfg.scfm_n_consistency_grid),
            }
            ckpt_subdir = Path(cfg.output_dir) / cfg.output_name
            ckpt_subdir.mkdir(parents=True, exist_ok=True)
            save_paths = [
                str(ckpt_subdir / f"{cfg.output_name}_{_step_tag(n)}.safetensors")
            ]
            if is_final:
                save_paths.append(
                    str(Path(cfg.output_dir) / f"{cfg.output_name}.safetensors")
                )
            for save_path in save_paths:
                turbo.save_student(save_path, dtype=torch.bfloat16, metadata=metadata)
                logger.info(f"saved checkpoint: {save_path}")

    if writer is not None:
        writer.close()
    logger.info("SCFM distillation complete.")
