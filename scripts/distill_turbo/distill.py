"""Turbo distillation main loop — DP-DMD (diversity-preserved DMD).

Usage:
    python -m scripts.distill_turbo.distill [--config configs/methods/turbo.toml] ...

The math walkthrough lives in :mod:`scripts.distill_turbo`; this file is the
per-step orchestrator (teacher K-step anchor → diversity-supervised first step →
DMD-refined N-step student rollout → fake/critic update → save). Run construction
(model, adapters, optimizers, dataloader, resume, warmup) lives in
:mod:`scripts.distill_turbo.setup`; ``run_loop`` below consumes the ``RunContext``
it returns.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from library.anima.models import Anima
from library.anima.uncond import uncond_for_batch
from library.training.progress import run_scope
from networks.methods.turbo_dmd import gan_loss_discriminator, gan_loss_generator

from .config import build_argparser, load_turbo_config, resolve_config
from .diversity import run_diversity_validation
from .metrics import (
    console_step_line,
    tqdm_postfix,
    tqdm_rate,
    write_scalars,
)
from .primitives import (
    cdm_extrapolate,
    renoise,
    sample_dynamic_sigmas,
    sample_t,
    sample_t_routed,
)
from .resume import resume_path_for, save_resume_state
from .setup import RunContext, build_run
from .softrank import caption_rank_loss

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _step_tag(step: int) -> str:
    """Human checkpoint suffix: 1000 -> ``1k``, 8000 -> ``8k``, else raw count.

    Matches the hand-rolled ``_1k`` / ``_500`` naming the runs already use.
    """
    return f"{step // 1000}k" if step % 1000 == 0 else str(step)


# --- f-distill reweighting (FastGen idea 2; f_distill.py:20 + _get_f_div_weighting_h)
# h = f'(r) where the density ratio r = exp(disc_logits) comes free from the GAN
# head (idea 1). "rkl" ≡ uniform h ≡ plain DMD2 (the off-by-default no-op).
_F_DIV_WEIGHTING = {
    "rkl": lambda r: torch.ones_like(r),
    "kl": lambda r: r,
    "js": lambda r: 1.0 - 1.0 / (1.0 + r),
    "sf": lambda r: 1.0 / (1.0 + r),
    "neyman": lambda r: 1.0 / torch.clamp(r, min=1e-8),
    "sh": lambda r: r**0.5,  # squared Hellinger
    "jf": lambda r: 1.0 + r,  # Jeffreys
}


def f_div_weighting_h(
    fake_logits: torch.Tensor,
    t: torch.Tensor,
    *,
    f_div: str,
    ratio_lower: float,
    ratio_upper: float,
    ema_rate: float,
    bins: torch.Tensor | None,
    bin_num: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-sample f-divergence reweight ``h(t, r)`` for the DMD signal.

    Port of ``FdistillModel._get_f_div_weighting_h`` (f_distill.py:59). ``r =
    exp(mean disc logits)`` clamped to ``[ratio_lower, ratio_upper]`` after a ±10
    logit clamp; an optional per-τ EMA histogram (``bins``) normalizes ``r`` so
    ``h`` isn't dominated by the batch's τ-distribution; ``h`` is renormalized to
    unit batch-mean. Everything is fp32 and detached — ``h`` only *scales* the
    already-detached DMD signal. Returns ``(h [B], updated_bins)``; ``bins`` is
    ``None`` when normalization is off.
    """
    logits = fake_logits.float()
    clamped = torch.clamp(logits.mean(dim=1), min=-10.0, max=10.0)
    ratio = torch.exp(clamped).detach()
    ratio = torch.clamp(ratio, ratio_lower, ratio_upper)
    if bins is not None:
        # τ is on [0, 1] (renoise level); bin directly over that range.
        tt = t.float().clamp(0.0, 1.0)
        bin_width = 1.0 / bin_num
        idx = (tt / bin_width).floor().long().clamp(0, bin_num - 1)
        cnt = torch.bincount(idx, minlength=bin_num).float()
        ratio_sum = torch.bincount(idx, weights=ratio, minlength=bin_num).float()
        valid = cnt > 0
        new_vals = ratio_sum / (cnt + 1e-6)
        bins = bins.clone()
        bins[valid] = bins[valid] * ema_rate + (1.0 - ema_rate) * new_vals[valid]
        ratio = ratio / (bins[idx] + 1e-6)
    h = _F_DIV_WEIGHTING[f_div](ratio)
    h = h / (h.mean() + 1e-6)
    return h.detach(), bins


@contextmanager
def selective_block_grad_ckpt(model: Anima):
    """Arm per-block gradient checkpointing for one forward, then restore.

    ``Block.forward`` self-checkpoints when ``gradient_checkpointing`` is set
    (gated on ``self.training`` + grad enabled). The decision is read eagerly per
    block, so flipping it per call costs no recompile. We snapshot each block's
    three checkpoint flags and restore them on exit, so this composes cleanly with
    a global ``--grad_ckpt`` run without clobbering it.

    We arm the **unsloth-offload** variant, NOT the standard ``torch_checkpoint``
    path. ``block._forward`` (the actual compute) is ``torch.compile``'d, and
    ``checkpoint(compiled_fn, use_reentrant=False)`` is unsupported: the recompute
    diverges from the inductor forward graph (dynamo recompile-storms on the
    GLOBAL_STATE ``num_threads`` flip, falls back to a non-autocast eager path →
    fp32 recompute, mismatched saved-tensor set, ``CheckpointError``). The unsloth
    path carries ``@torch._disable_dynamo`` (``models.py``), so the compiled
    ``_forward`` runs eager in BOTH forward and recompute → consistent, and it
    offloads saved tensors to CPU (extra VRAM win). The reentrant grad-drop bug
    ([[project_unsloth_reentrant_drops_grad]]) does not apply here: the frozen
    teacher view has no grad-requiring params inside the region, so grad flows
    purely through the grad-requiring input (x_renoised_gan → student).

    Used to wrap ONLY the grad-bearing GAN gen teacher forward: the frozen teacher
    retains ~half the DiT's block activations there purely to backprop into
    x_pred → student, so recomputing them in backward reclaims that peak VRAM
    (~one half-depth forward of compute) — numerically exact (no dropout).
    """
    saved = [
        (
            b.gradient_checkpointing,
            b.unsloth_offload_checkpointing,
        )
        for b in model.blocks
    ]
    for b in model.blocks:
        b.gradient_checkpointing = True
        b.unsloth_offload_checkpointing = True
    try:
        yield
    finally:
        for b, (g, u) in zip(model.blocks, saved):
            b.gradient_checkpointing = g
            b.unsloth_offload_checkpointing = u


def main():
    args = build_argparser().parse_args()
    cfg = resolve_config(args, load_turbo_config(args.config))
    ctx = build_run(args, cfg)
    run_loop(ctx, cfg)


def run_loop(ctx: RunContext, cfg):
    """Per-step DP-DMD training loop over the objects built by ``build_run``.

    ``ctx`` fields are bound to locals up front so the loop body reads as the
    plain algorithm; only ``data_iter`` and ``fdistill_bins`` are mutated
    (epoch re-iter / f-distill EMA), and neither is read after the loop.
    """
    model = ctx.model
    turbo = ctx.turbo
    device = ctx.device
    dtype = ctx.dtype
    student_opt, fake_opt, disc_opt = ctx.student_opt, ctx.fake_opt, ctx.disc_opt
    student_sched = ctx.student_sched
    fake_sched = ctx.fake_sched
    disc_sched = ctx.disc_sched
    dataloader = ctx.dataloader
    data_iter = ctx.data_iter
    _forward = ctx.forward
    _teacher_cfg_velocity = ctx.teacher_cfg_velocity
    student_sigmas = ctx.student_sigmas
    teacher_anchor_sigmas = ctx.teacher_anchor_sigmas
    t_k_anchor = ctx.t_k_anchor
    use_anchor = ctx.use_anchor
    dyn_n_min = ctx.dyn_n_min
    uncond_base = ctx.uncond_base
    softrank_on = ctx.softrank_on
    softrank_pool = ctx.softrank_pool
    softrank_min_pool = ctx.softrank_min_pool
    cdm_on = ctx.cdm_on
    fdistill_on = ctx.fdistill_on
    fdistill_bins = ctx.fdistill_bins
    writer = ctx.writer
    progress_sink = ctx.progress_sink
    console_steps = ctx.console_steps
    metrics = ctx.metrics
    tau_profiles = ctx.tau_profiles
    val_cond = ctx.val_cond
    val_latent_shape = ctx.val_latent_shape
    val_clean = ctx.val_clean
    start_step = ctx.start_step

    progress = tqdm(
        range(start_step, cfg.iterations),
        desc="turbo",
        initial=start_step,
        total=cfg.iterations,
    )

    # Full-run lifecycle for the progress.jsonl sink (issue #1): run_scope
    # maps a clean return / KeyboardInterrupt / crash onto the matching
    # run_end status, so a reader can tell 'done' from 'died'.
    step = start_step - 1  # sentinel: valid final_step() if the loop is empty
    with run_scope(progress_sink, final_step=lambda: step + 1):
        for step in progress:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            latents = batch["latents"]
            crossattn_emb = batch["crossattn_emb"]
            if cfg.use_masked_loss:
                # float (not bf16): the student loss is assembled in fp32. [B,1,H,W]
                # broadcasts over the [B,16,H,W] grad signal.
                mask = batch["mask"].to(device, dtype=torch.float32, non_blocking=True)
            else:
                mask = None

            latents = latents.to(device, dtype=dtype, non_blocking=True)
            crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
            B = latents.shape[0]

            # No-op today (mode="default" doesn't enable cudagraphs), but the right
            # cadence if/when the script switches to "reduce-overhead".
            torch.compiler.cudagraph_mark_step_begin()

            # Student update: roll an N-step Euler grid from pure noise ε (dpdmd
            # anchors step 1 to a teacher K-step target then refines; dmd is plain).
            eps = torch.randn_like(latents)  # shared start for anchor + student
            c_null = uncond_for_batch(uncond_base, crossattn_emb)  # anchor + DMD eval

            # --- teacher K-step CFG anchor (no grad) → v_target (DP-DMD only) ---
            v_target = None
            if use_anchor:
                z = eps
                for i in range(cfg.k_anchor):
                    s_i = teacher_anchor_sigmas[i]
                    s_next = teacher_anchor_sigmas[i + 1]
                    t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                    v = _teacher_cfg_velocity(z, t_b, crossattn_emb, c_null)
                    z = (z.float() - (s_i - s_next) * v).to(dtype)
                # Average velocity ε→z_tk over [t_k, 1]; this is exactly the target
                # for the student's t=1 first step (Euler x_next = x − dt·v_first).
                v_target = ((eps.float() - z.float()) / (1.0 - t_k_anchor)).detach()

            # --- student rollout → x_pred (= x_θ, B,16,H,W) + v_student (metric) ---
            # dpdmd: step-0 diversity anchor + DMD-refined steps 1..N-1.
            # dmd:   plain DMD2; cfg.dmd_grad_step picks which step(s) grad.
            split_bwd = use_anchor and cfg.detach_after_first
            # This iteration's rollout grid: the static inference grid, or a fresh
            # CDM dynamic draw. Everything below indexes sigmas_it/n_steps_it so
            # the two modes share one code path.
            if cfg.dynamic_schedule:
                sigmas_it = sample_dynamic_sigmas(dyn_n_min, cfg.student_steps)
                n_steps_it = len(sigmas_it) - 1
            else:
                sigmas_it, n_steps_it = student_sigmas, cfg.student_steps
            last_step = n_steps_it - 1

            # Soft-rank caption auxiliary state (metrics + non-split backward read it
            # even on steps/branches where the term doesn't fire). Zero leaf ⇒ adds
            # nothing to any backward when off/skipped.
            softrank_loss = torch.zeros((), device=device)
            softrank_ran = False

            # L_CDM launch point: the DMD grad step's on-trajectory (x_in, v, σ),
            # captured raw here and detached at use (cdm_extrapolate). Under
            # grad_step='random' the launch point sweeps the whole grid over
            # training; under 'all'/'last' it is the final (cleanest-σ) step.
            cdm_src = None

            if use_anchor:
                # Step 0 is the diversity anchor (supervised toward v_target, then
                # detached under split_bwd); steps 1..N-1 carry the DMD-refine grad,
                # routed by grad_step ('all' BPTT | 'last' tail-only | 'random' grid).
                x = eps
                x.requires_grad_()  # grad-ckpt needs a grad-requiring forward input
                s0, s0_next = sigmas_it[0], sigmas_it[1]
                t_b = torch.full((B,), s0, device=device, dtype=dtype)
                turbo.set_student_step(0)  # head 0 (no-op unless per-step-expert)
                v_first = _forward(
                    "student", x, t_b, crossattn_emb, no_grad=False
                ).squeeze(2)
                x = x - (s0 - s0_next) * v_first
                div_loss_t = nn.functional.mse_loss(v_first.float(), v_target)

                # --- soft-rank caption-discrimination auxiliary (Phase 1) ---------
                # k extra no_grad student forwards at the SAME step-0 (ε, t=1, head 0),
                # only crossattn_emb swapped for a pooled sample's caption (the
                # `shuffled` negatives where Phase 0 measured the worst damage). The
                # matched caption should explain the diversity anchor v_target better
                # than the mismatched ones; soft-rank its position and push it to 0.
                # Grad flows through v_first ONLY (negatives detached under no_grad) —
                # "make the matched caption explain the anchor better", and the term
                # stays bounded (no negative-push). It rides the step-0 backward below.
                if softrank_on:
                    if step % cfg.softrank_every_n == 0 and softrank_pool.ready(
                        softrank_min_pool
                    ):
                        # Pool negatives → works at any batch size (B=1 included). Head
                        # 0 stays selected → no per-step-expert recompute hazard.
                        v_negs = [
                            _forward("student", eps, t_b, c_neg, no_grad=True).squeeze(
                                2
                            )
                            for c_neg in softrank_pool.draw(cfg.softrank_k, B)
                        ]
                        softrank_loss = caption_rank_loss(
                            v_first, v_negs, v_target, tau=cfg.softrank_softness
                        )
                        softrank_ran = True
                    # Fill AFTER drawing so an anchor never draws its own caption.
                    softrank_pool.add(crossattn_emb)

                if split_bwd:
                    # Load-bearing stop-grad: the DMD reverse-KL from steps 1..N-1 must
                    # NOT flow into the diversity mapping (their Fig 5). Backward the
                    # diversity term now, then re-leaf for a fresh DMD-chain root. The
                    # soft-rank term joins THIS backward (both ride v_first's step-0
                    # graph), so the DMD graph separation is untouched.
                    (
                        cfg.div_weight * div_loss_t
                        + cfg.softrank_weight * softrank_loss
                    ).backward()
                    softrank_loss = softrank_loss.detach()  # metrics-only from here
                    x = x.detach().requires_grad_()
                if cfg.dmd_grad_step == "random":
                    # Memory-flat anchored DMD: sample ONE refinement step g~U{1..N-1},
                    # backward-simulate the prefix under no_grad, grad only step g's
                    # one-step x0-prediction (x_g − σ_g·v_g). Supervises every grid
                    # point over training (vs 'last') and trains head g under
                    # per_step_expert. Step 0's diversity graph rides v_first untouched.
                    g = int(torch.randint(1, n_steps_it, (1,)).item())
                    for i in range(1, g):  # backward simulation (no graph kept)
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=True
                        ).squeeze(2)
                        x = x - (s_i - s_next) * v
                    x = x.detach().requires_grad_()  # fresh leaf; head g trains
                    s_g = sigmas_it[g]
                    t_b = torch.full((B,), s_g, device=device, dtype=dtype)
                    turbo.set_student_step(g)
                    v_g = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=False
                    ).squeeze(2)
                    if cdm_on:
                        cdm_src = (x, v_g, s_g)
                    x_pred = x - s_g * v_g  # one-step x0-prediction at step g
                else:
                    # 'all' → full BPTT over 1..N-1; else ('last') → only the final step
                    # grads (1..N-2 backward-simulated under no_grad). Both memory-flat
                    # except 'all', and land the DMD grad on the true rollout endpoint.
                    grad_dmd_last_only = cfg.dmd_grad_step != "all"
                    for i in range(1, n_steps_it):
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        step_no_grad = grad_dmd_last_only and i != last_step
                        if grad_dmd_last_only and i == last_step:
                            x = (
                                x.detach().requires_grad_()
                            )  # fresh leaf after no_grad prefix
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=step_no_grad
                        ).squeeze(2)
                        if cdm_on and i == last_step:
                            cdm_src = (x, v, s_i)
                        x = x - (s_i - s_next) * v
                        if step_no_grad:
                            x = x.detach()
                    x_pred = x
                v_student = v_first  # step-0 velocity for the runaway-student metric
            else:
                # Plain DMD2. Non-grad steps are backward-SIMULATED under no_grad (the
                # generator trains on its OWN trajectory — DMD2's train/inference input
                # match, Yin et al. 2024 — not forward-noised real latents).
                div_loss_t = torch.zeros((), device=device)  # uniform metrics path
                if cfg.dmd_grad_step == "all":
                    # Full-rollout BPTT: every step grads into the endpoint x_pred.
                    x = eps
                    x.requires_grad_()
                    v_student = None
                    for i in range(n_steps_it):
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=False
                        ).squeeze(2)
                        if v_student is None:
                            v_student = v
                        if cdm_on and i == last_step:
                            cdm_src = (x, v, s_i)
                        x = x - (s_i - s_next) * v
                    x_pred = x
                else:
                    # Single grad-step: 'last' pins g=N-1; 'random' samples g~U{0..N-1}
                    # (canonical DMD2 — supervises every grid point, not just the clean
                    # tail). Roll to g under no_grad, grad ONLY step g, supervise its
                    # one-step x0-prediction x_g − σ_g·v_g. Memory-flat (1 forward graph).
                    if cfg.dmd_grad_step == "random":
                        # CPU RNG → no per-step GPU sync (seeded by torch.manual_seed).
                        g = int(torch.randint(0, n_steps_it, (1,)).item())
                    else:
                        g = last_step
                    x = eps
                    for i in range(g):  # backward simulation (no_grad → no graph kept)
                        s_i = sigmas_it[i]
                        s_next = sigmas_it[i + 1]
                        t_b = torch.full((B,), s_i, device=device, dtype=dtype)
                        turbo.set_student_step(i)
                        v = _forward(
                            "student", x, t_b, crossattn_emb, no_grad=True
                        ).squeeze(2)
                        x = x - (s_i - s_next) * v
                    x = x.detach().requires_grad_()  # fresh leaf; head g trains
                    s_g = sigmas_it[g]
                    t_b = torch.full((B,), s_g, device=device, dtype=dtype)
                    turbo.set_student_step(g)
                    v_g = _forward(
                        "student", x, t_b, crossattn_emb, no_grad=False
                    ).squeeze(2)
                    if cdm_on:
                        cdm_src = (x, v_g, s_g)
                    x_pred = x - s_g * v_g  # one-step x0-prediction at step g
                    v_student = v_g

            # --- DMD on x_θ (steps 2..N), against teacher + fake ---
            # The real score MUST be CFG-GUIDED (v_u + α·(v_c − v_u)), not cond-only:
            # without guidance v_real≈v_fake (both unguided cond preds collapse,
            # dm_cos≈0.9999) and the quality gradient is noise. Fake stays cond-only
            # (matches the reference compute_dmd_loss).
            # τ-split critic: the DMD query routes to the owner bank too (training a
            # specialist and letting one bank answer all queries would ignore it).
            # The query τ is uniform by design (independent of t_distribution);
            # banks=1 resolves to the identical torch.rand call/RNG stream.
            tau_dm = sample_t_routed(
                B,
                turbo=turbo,
                fake_tau_banks=cfg.fake_tau_banks,
                fake_tau_boundary=cfg.fake_tau_boundary,
                distribution="uniform",
                sigmoid_scale=cfg.sigmoid_scale,
                device=device,
                dtype=dtype,
            )
            eps_dm = torch.randn_like(x_pred)
            x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
            v_real_cond_dm = _teacher_cfg_velocity(
                x_renoised_dm, tau_dm, crossattn_emb, c_null
            )
            v_fake_cond_dm = _forward(
                "fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
            ).squeeze(2)
            delta_dm = v_real_cond_dm - v_fake_cond_dm

            tau_dm_e = tau_dm.view(B, 1, 1, 1).float()
            grad_dm = tau_dm_e * delta_dm.float()
            if cfg.dm_x0_norm:
                denom = (
                    (tau_dm_e * v_real_cond_dm.float())
                    .abs()
                    .mean(dim=(1, 2, 3), keepdim=True)
                    .clamp_min(cfg.norm_floor)
                )
                grad_dm = grad_dm / denom
            grad_signal = grad_dm.detach()

            # --- L_CDM off-trajectory loss (CDM §3.3; docs/proposal/cdm.md Phase 1) ---
            # From the grad step's on-trajectory (x_g, v_g, σ_g), Euler-extrapolate a
            # large random stride to x_off at t' ~ U(0,1) (velocity-driven, their
            # Eq. 7 — cdm_extrapolate detaches, so this is a fresh leaf like the DM
            # renoise path: one grad forward, no second BPTT chain). The student's
            # local clean estimate there, x0_off = x_off − t'·v_off, gets the same
            # real-vs-fake DMD surrogate as grad_dm (variant A: CFG'd real score,
            # consistent with the fused DM; fresh τ̂ draw). This supervises exactly
            # the truncation-drift region few-step Euler traverses off-manifold,
            # which on-trajectory rollouts never visit even under dynamic_schedule.
            # ORDER MATTERS: this whole branch must run BEFORE the GAN gen forward —
            # that forward's checkpointed recompute happens at backward under the
            # then-current view, so it must stay the last view flip of the step
            # (project_turbo_view_ckpt_recompute_hazard).
            #
            # VRAM: the CDM student forward is unsloth-checkpointed (same lever as
            # gan.grad_ckpt — a full second grad graph next to the step-g graph
            # OOMs a 16 GB card) and BACKWARDED IN-BRANCH, so its graph is freed
            # before the GAN forward builds. The ckpt'd forward recomputes at that
            # backward under the then-current view, so the view is restored to
            # 'student' first (the teacher/fake delta forwards flipped it) —
            # backward-while-view-live, the same contract the GAN forward honors.
            cdm_loss = torch.zeros((), device=device)
            if cdm_on and cdm_src is not None:
                x_g_cdm, v_g_cdm, s_g_cdm = cdm_src
                # CPU RNG (seeded by torch.manual_seed, no GPU sync), per-sample.
                t_off = torch.rand(B).to(device=device, dtype=dtype)
                # requires_grad_ is LOAD-BEARING under the unsloth ckpt below:
                # the reentrant path silently drops the LoRA param grads when
                # every explicit checkpoint input is detached
                # (project_unsloth_reentrant_drops_grad) — the leaf's grad flag
                # is what forces the autograd node.
                x_off = (
                    cdm_extrapolate(x_g_cdm, v_g_cdm, s_g_cdm, t_off)
                    .to(dtype)
                    .requires_grad_()
                )
                with selective_block_grad_ckpt(model):
                    v_off = _forward(
                        "student", x_off, t_off, crossattn_emb, no_grad=False
                    ).squeeze(2)
                # Local clean estimate at t' (their Eq. 1, in our v-param).
                x0_off = x_off.float() - t_off.view(B, 1, 1, 1).float() * v_off.float()
                tau_cdm = sample_t_routed(
                    B,
                    turbo=turbo,
                    fake_tau_banks=cfg.fake_tau_banks,
                    fake_tau_boundary=cfg.fake_tau_boundary,
                    distribution="uniform",
                    sigmoid_scale=cfg.sigmoid_scale,
                    device=device,
                    dtype=dtype,
                )
                eps_cdm = torch.randn_like(latents)
                x_renoised_cdm = renoise(x0_off.detach().to(dtype), tau_cdm, eps_cdm)
                v_real_cdm = _teacher_cfg_velocity(
                    x_renoised_cdm, tau_cdm, crossattn_emb, c_null
                )
                v_fake_cdm = _forward(
                    "fake", x_renoised_cdm, tau_cdm, crossattn_emb, no_grad=True
                ).squeeze(2)
                delta_cdm = v_real_cdm - v_fake_cdm
                tau_cdm_e = tau_cdm.view(B, 1, 1, 1).float()
                grad_cdm = tau_cdm_e * delta_cdm.float()
                if cfg.dm_x0_norm:
                    denom_cdm = (
                        (tau_cdm_e * v_real_cdm.float())
                        .abs()
                        .mean(dim=(1, 2, 3), keepdim=True)
                        .clamp_min(cfg.norm_floor)
                    )
                    grad_cdm = grad_cdm / denom_cdm
                grad_cdm = grad_cdm.detach()
                if mask is not None:
                    cdm_loss = (grad_cdm * x0_off * mask).mean()
                else:
                    cdm_loss = (grad_cdm * x0_off).mean()
                # Backward NOW, with the student view restored: the ckpt'd forward
                # above recomputes here and must see the view it was recorded
                # under (the delta forwards flipped it to teacher/fake). Frees the
                # CDM graph before the GAN forward; grads accumulate, grad_clip
                # below clips the full student gradient once.
                turbo.set_view("student")
                (cfg.cdm_weight * cdm_loss).backward()
                cdm_loss = cdm_loss.detach()  # metrics-only from here
                metrics.add_cdm(grad_cdm)

            # --- GAN generator term + f-distill reweighting (ideas 1 & 2) ---
            # The disc scores the frozen TEACHER's block features of the student's
            # renoised x_pred. Grad must flow into x_pred → student, so this renoise
            # keeps x_pred attached (unlike the DMD path) and the teacher forward is
            # grad-enabled; the disc itself is frozen here. return_features_early stops
            # after the deepest tapped block (half-depth grad forward — full-stack OOM'd).
            gan_gen_loss = torch.zeros((), device=device)
            if turbo.disc is not None:
                turbo.set_disc_requires_grad(False)
                x_renoised_gan = renoise(x_pred, tau_dm, eps_dm)  # grad-bearing
                # Selectively checkpoint just this forward (the only GAN extra that
                # retains a backward graph); recompute trades ~half-depth compute for
                # the ~3 GB of retained teacher activations. nullcontext when off.
                gan_ckpt = (
                    selective_block_grad_ckpt(model)
                    if cfg.gan_grad_ckpt
                    else nullcontext()
                )
                with gan_ckpt:
                    feats_gen = _forward(
                        "teacher",
                        x_renoised_gan,
                        tau_dm,
                        crossattn_emb,
                        no_grad=False,
                        return_block_features=turbo.gan_feature_set,
                        return_features_early=True,
                    )
                fake_logits_gen = turbo.disc(
                    turbo.features_in_order(feats_gen)
                )  # (B, taps), grad→x_pred
                gan_gen_loss = gan_loss_generator(fake_logits_gen)

                # f-distill: scale the (detached) DMD signal by h(τ, r), r=exp(logits).
                if fdistill_on:
                    h, fdistill_bins = f_div_weighting_h(
                        fake_logits_gen,
                        tau_dm,
                        f_div=cfg.f_div,
                        ratio_lower=cfg.f_ratio_lower,
                        ratio_upper=cfg.f_ratio_upper,
                        ema_rate=cfg.f_ratio_ema_rate,
                        bins=fdistill_bins,
                        bin_num=cfg.f_bin_num,
                    )
                    grad_signal = grad_signal * h.view(B, 1, 1, 1)

            # --- assemble: DMD surrogate on x_θ ---
            # The diversity term was already backwarded above when split_bwd; otherwise
            # it rides this combined backward (graphs still entangled). grad_clip below
            # runs once on the ACCUMULATED .grad (div + DMD), so the clipped norm is the
            # full student gradient either way.
            if mask is not None:
                loss_dmd = (grad_signal * x_pred.float() * mask).mean()
            else:
                loss_dmd = (grad_signal * x_pred.float()).mean()
            loss_student = loss_dmd

            if use_anchor and not split_bwd:
                # div + soft-rank both ride v_first's retained step-0 graph here (no
                # split backward), so they join the combined student backward below.
                loss_student = (
                    loss_student
                    + cfg.div_weight * div_loss_t
                    + cfg.softrank_weight * softrank_loss
                )

            if turbo.disc is not None:
                loss_student = loss_student + cfg.gan_loss_weight_gen * gan_gen_loss

            loss_student.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    turbo.student_params(), max_norm=cfg.grad_clip
                )
            student_opt.step()
            student_opt.zero_grad(set_to_none=True)
            student_sched.step()

            # Fake update: fake_steps_per_student_step inner updates against the same
            # x_pred.detach(), resampling (τ_fake, ε_fake) each iteration. Standard
            # DMD2 practice — keep the fake's target ahead of the moving x_pred dist.
            x_pred_d = x_pred.detach()
            fake_loss_sum = torch.zeros((), device=device)
            gan_disc_sum = torch.zeros((), device=device)
            for _ in range(cfg.fake_steps_per_student_step):
                # τ-split critic: the drawn τ picks which bank trains this update
                # (banks=1 resolves to the identical sample_t call/RNG stream).
                tau_fake = sample_t_routed(
                    B,
                    turbo=turbo,
                    fake_tau_banks=cfg.fake_tau_banks,
                    fake_tau_boundary=cfg.fake_tau_boundary,
                    distribution=cfg.t_distribution,
                    sigmoid_scale=cfg.sigmoid_scale,
                    device=device,
                    dtype=dtype,
                )
                eps_fake = torch.randn_like(x_pred_d)
                x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
                v_fake = _forward(
                    "fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False
                ).squeeze(2)
                target_v_fake = eps_fake - x_pred_d  # flow-matching target
                fake_loss = nn.functional.mse_loss(
                    v_fake.float(), target_v_fake.float()
                )
                fake_loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        turbo.fake_params(), max_norm=cfg.grad_clip
                    )
                fake_opt.step()
                fake_opt.zero_grad(set_to_none=True)
                fake_sched.step()
                fake_loss_sum = fake_loss_sum + fake_loss.detach()
                tau_profiles[turbo.fake_bank].add(fake_loss, tau_fake)

                # Discriminator update (idea 1), co-located with the fake/critic update
                # (FastGen cadence). The disc scores frozen-TEACHER block features of
                # renoised fake (x_pred) vs renoised real latents — grad only to the disc
                # head. gan_use_same_t_noise reuses (τ_fake, ε_fake) for the real branch.
                if turbo.disc is not None:
                    turbo.set_disc_requires_grad(True)
                    if cfg.gan_use_same_t_noise:
                        tau_d, eps_d = tau_fake, eps_fake
                    else:
                        tau_d = sample_t(
                            B,
                            distribution=cfg.t_distribution,
                            sigmoid_scale=cfg.sigmoid_scale,
                            device=device,
                            dtype=dtype,
                        )
                        eps_d = torch.randn_like(x_pred_d)

                    # Feature-only teacher forwards (no_grad → grad only to the disc
                    # head). Early-exit at the deepest tap; each call returns its own
                    # feature dict, so the fake/real captures never alias.
                    def _disc_feats(latent_in):
                        return turbo.features_in_order(
                            _forward(
                                "teacher",
                                renoise(latent_in, tau_d, eps_d),
                                tau_d,
                                crossattn_emb,
                                no_grad=True,
                                return_block_features=turbo.gan_feature_set,
                                return_features_early=True,
                            )
                        )

                    fake_logits_d = turbo.disc(_disc_feats(x_pred_d))
                    real_logits_d = turbo.disc(_disc_feats(latents))
                    loss_disc = gan_loss_discriminator(real_logits_d, fake_logits_d)

                    # Approximate-R1 (APT): penalize disc logit change under a small
                    # perturbation of the real disc input. Perturb the renoised real
                    # latent directly (the tensor whose features feed the disc).
                    if cfg.gan_r1_weight > 0.0:
                        x_t_real_a = renoise(
                            latents, tau_d, eps_d
                        ) + cfg.gan_r1_alpha * torch.randn_like(latents)
                        feats_a = _forward(
                            "teacher",
                            x_t_real_a,
                            tau_d,
                            crossattn_emb,
                            no_grad=True,
                            return_block_features=turbo.gan_feature_set,
                            return_features_early=True,
                        )
                        real_logits_a = turbo.disc(turbo.features_in_order(feats_a))
                        loss_disc = (
                            loss_disc
                            + cfg.gan_r1_weight
                            * nn.functional.mse_loss(real_logits_d, real_logits_a)
                        )

                    loss_disc.backward()
                    if cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            turbo.disc_params(), max_norm=cfg.grad_clip
                        )
                    disc_opt.step()
                    disc_opt.zero_grad(set_to_none=True)
                    disc_sched.step()
                    turbo.set_disc_requires_grad(False)
                    gan_disc_sum = gan_disc_sum + loss_disc.detach()
            fake_loss_mean_t = fake_loss_sum / cfg.fake_steps_per_student_step
            gan_disc_mean_t = gan_disc_sum / cfg.fake_steps_per_student_step

            # --- logging accumulators (all GPU-side; flushed below every log_interval
            # in one stacked .tolist() so per-step CUDA syncs go to zero) ---
            metrics.accumulate_per_step(
                fake_loss_mean_t=fake_loss_mean_t,
                grad_signal=grad_signal,
                delta_dm=delta_dm,
                x_pred=x_pred,
                v_student=v_student,
                tau_dm_e=tau_dm_e,
                v_real_cond_dm=v_real_cond_dm,
                v_fake_cond_dm=v_fake_cond_dm,
            )
            metrics.add_div(div_loss_t)
            if turbo.disc is not None:
                metrics.add_gan(gan_gen_loss, gan_disc_mean_t)
            if softrank_on:
                metrics.add_softrank(softrank_loss, active=softrank_ran)

            if (step + 1) % cfg.log_interval == 0:
                m = metrics.flush(cfg.log_interval)
                if writer is not None:
                    write_scalars(writer, m, step + 1)
                    writer.add_scalar(
                        "train/student_lr", student_sched.get_last_lr()[0], step + 1
                    )
                    writer.add_scalar(
                        "train/fake_lr", fake_sched.get_last_lr()[0], step + 1
                    )
                    if disc_sched is not None:
                        writer.add_scalar(
                            "train/disc_lr", disc_sched.get_last_lr()[0], step + 1
                        )
                # log_interval cadence (per-step would re-introduce the syncs we
                # just eliminated).
                progress.set_postfix(**tqdm_postfix(m))
                if console_steps:
                    logger.info(
                        console_step_line(
                            m,
                            step=step + 1,
                            total=cfg.iterations,
                            rate=tqdm_rate(progress),
                        )
                    )
                if progress_sink is not None:
                    # FlushedMetrics → dict of scalar floats; sink emits a `step`
                    # event (no _cmmd key, so it's not misread as a val pass).
                    progress_sink.log(
                        dataclasses.asdict(m), global_step=step + 1, epoch=0
                    )
                metrics.reset()
                for tp in tau_profiles:
                    tp.write(writer, step + 1)

            # --- diversity validation (DAVE same-prompt probe) ---
            if val_cond is not None and (step + 1) % cfg.validate_every_n_steps == 0:
                dm = run_diversity_validation(
                    model=model,
                    forward_fn=_forward,
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
                    f"| x_pred AC sim={dm.xpred_ac_sim:.4f} | FM MSE={dm.fm_mse:.4f} "
                    f"(fidelity; not a quality score)"
                )

            # Each checkpoint is kept under a step-tagged name (no overwrite, so the
            # whole trajectory survives); the final step also writes the canonical
            # bare `{output_name}` that inference / merge / `make test` look for.
            if (step + 1) % cfg.save_every == 0 or (step + 1) == cfg.iterations:
                n = step + 1
                is_final = n == cfg.iterations
                metadata = {
                    "ss_turbo_objective": cfg.base_loss,
                    "ss_turbo_student_rank": str(cfg.student_rank),
                    "ss_turbo_student_alpha": str(cfg.student_alpha),
                    "ss_turbo_student_steps": str(cfg.student_steps),
                    "ss_turbo_dynamic_schedule": "1" if cfg.dynamic_schedule else "0",
                    "ss_turbo_teacher_cfg": str(cfg.teacher_cfg),
                    "ss_turbo_step": str(n),
                    "ss_turbo_k_anchor": str(cfg.k_anchor),
                    "ss_turbo_div_weight": str(cfg.div_weight),
                    "ss_turbo_gan_weight_gen": str(cfg.gan_loss_weight_gen),
                    "ss_turbo_cdm_weight": str(cfg.cdm_weight),
                    "ss_turbo_f_div": cfg.f_div,
                }
                if cfg.train_adaln:
                    # The student targets adaln_up_{branch}; save_student ships the
                    # adaln keys in the ComfyUI layout (adaln.md).
                    metadata["ss_turbo_train_adaln"] = "1"
                    if cfg.adaln_rank > 0:
                        # Provenance only — per-module rank/alpha live in the file.
                        metadata["ss_turbo_adaln_rank"] = str(cfg.adaln_rank)
                if cfg.student_init_weights:
                    # Provenance only — the warm start distills to a normal LoRA.
                    metadata["ss_turbo_student_init_weights"] = os.path.basename(
                        cfg.student_init_weights
                    )
                if cfg.per_step_expert:
                    # Drives loader detection (CLI + ComfyUI build StepExpertLoRAModule
                    # and keep it live instead of merging). step_expert_K == the head
                    # count == student_steps.
                    metadata["ss_turbo_per_step_expert"] = "1"
                    metadata["ss_turbo_step_expert_K"] = str(cfg.step_expert_K)
                # Step-tagged intermediates live in a per-run subdir so they don't
                # clutter output/ckpt/; the canonical bare {output_name} stays at the
                # root where inference / merge / `make test` look for it.
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
                    turbo.save_student(
                        save_path, dtype=torch.bfloat16, metadata=metadata
                    )
                    logger.info(f"saved checkpoint: {save_path}")
                    if progress_sink is not None:
                        progress_sink.ckpt(global_step=n, path=save_path)

                # Crash-resume bundle: everything save_student drops on the floor (fake,
                # disc, three optimizers, three schedulers, f-distill EMA, RNG). Rolling
                # single file, written atomically — see resume.py. Skipped on the final
                # step: the run is complete, and the bundle is ~10× a student ckpt.
                if not is_final:
                    rp = resume_path_for(cfg.output_dir, cfg.output_name)
                    save_resume_state(
                        rp,
                        step=n,
                        cfg=cfg,
                        turbo=turbo,
                        student_opt=student_opt,
                        fake_opt=fake_opt,
                        disc_opt=disc_opt,
                        student_sched=student_sched,
                        fake_sched=fake_sched,
                        disc_sched=disc_sched,
                        fdistill_bins=fdistill_bins,
                    )
                    logger.info(f"saved resume bundle: {rp} (step {n})")

    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
