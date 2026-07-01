"""RSD distillation training loop (reconstruction of paper 22490, Algorithm 1).

Two-timescale: K fake-critic updates per generator update. Frozen v2 teacher, 1-step
stochastic student, fake ResShift critic + GAN head, image-space LPIPS. See DESIGN.md.

Run inside sr/.venv:  python train.py --iters 3000 --bs 2 [--amp]
"""
import argparse
import copy
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from data import ArtSRDataset  # noqa: E402
from rsd_models import TEACHER_CKPT, make_eps, predict_x0  # noqa: E402


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def dc_loss(a, b, k=32):
    """L1 on the low-frequency (DC + coarse tone) band of the decoded image.

    VGG-LPIPS is ~invariant to a uniform color/tone shift and the DMD gradient is
    per-sample magnitude-normalized, so the global DC (mean color) is the objective's
    unconstrained null-space — the 1-step student settles into a small systematic tint
    (measured ~-0.027 on blue, -0.012 luma vs GT; the VQGAN roundtrip is color-faithful,
    so it's the student, not the decoder). Avg-pooling to a coarse map and matching L1
    pins that band without touching high-freq detail. Both a,b are image-space [-1,1].
    """
    return F.l1_loss(F.avg_pool2d(a, k), F.avg_pool2d(b, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000, help="generator updates")
    ap.add_argument("--K", type=int, default=5, help="fake updates per generator update")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda_lpips", type=float, default=2.0)
    ap.add_argument("--lambda_gan", type=float, default=3e-3)
    ap.add_argument("--lambda_dc", type=float, default=1.0,
                    help="low-freq DC/color match on the 1-step decoded image — pins the "
                         "global tone LPIPS+DMD leave free (fixes the student color drift). "
                         "0 disables; raise toward ~5 if a tint persists.")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--nodes", type=int, nargs="+", default=[4, 8, 12, 14],
                    help="N multistep timestep nodes (0-indexed in [0,T-1])")
    ap.add_argument("--sid_denom", choices=["student", "input"], default="student",
                    help="SiD per-sample normalization denominator for the DMD gradient. "
                         "'student' = |teacher_x0 - student_x0| + 1e-8 (matches official RSD "
                         "release, trainer.py:1836); 'input' = |z_t - teacher_x0| floored 0.05 "
                         "(this reconstruction's original form).")
    ap.add_argument("--noise_mode", choices=["add", "concat"], default="add",
                    help="student/fake noise injection. 'add' = zero-init 3ch conv added after "
                         "block0 (default, back-compatible with existing checkpoints); 'concat' = "
                         "widen the first conv by noise_channels and concat eps (matches official "
                         "RSD release — changes the first-conv shape, so old ckpts won't load).")
    ap.add_argument("--noise_channels", type=int, default=None,
                    help="injected-noise channel count (default 3 for add, 1 for concat/official)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")
    ap.add_argument("--no_grad_ckpt", action="store_true",
                    help="disable Swin gradient checkpointing on student/fake (on by "
                         "default — big activation-memory save, bit-exact)")
    ap.add_argument("--src", default=None,
                    help="HR source dir (default image_dataset; pass the prep_rsd_cache "
                         "4096-capped cache for faster decode at the right 1024->4096 scale)")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--save_dir", default=str(M.REPO / "output" / "sr" / "rsd"))
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=0, help="smoke cap on gen updates (0=off)")
    args = ap.parse_args()

    dev = M.DEVICE
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = M.load_configs()
    sf_scale = cfg.diffusion.params.scale_factor

    print("building nets...")
    gen_kw = dict(noise_mode=args.noise_mode, noise_channels=args.noise_channels)
    teacher = M.build_teacher(cfg, str(TEACHER_CKPT), dev)
    student = M.build_generator(cfg, str(TEACHER_CKPT), dev, grad_ckpt=not args.no_grad_ckpt, **gen_kw)
    fake = M.build_generator(cfg, str(TEACHER_CKPT), dev, grad_ckpt=not args.no_grad_ckpt, **gen_kw)
    disc = M.DiscHead().to(dev)
    vqgan = M.build_autoencoder(cfg, dev)
    diff = M.build_diffusion(cfg)
    T = diff.num_timesteps
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    # provenance for inference (rebuilds the matching noise-injection arch) + A/B tracking
    ckpt_meta = {"noise_mode": student.noise_mode, "noise_channels": student.noise_channels,
                 "sid_denom": args.sid_denom}

    import lpips
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_f = torch.optim.AdamW(list(fake.parameters()) + list(disc.parameters()),
                              lr=args.lr, betas=(0.9, 0.95))

    loader = DataLoader(ArtSRDataset(src=args.src, gt_size=256, scale=cfg.diffusion.params.sf,
                                     length=args.iters * (args.K + 1) * args.bs * args.grad_accum + 1000),
                        batch_size=args.bs, num_workers=args.num_workers, drop_last=True,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    it = iter(loader)
    nodes = torch.tensor(args.nodes, device=dev)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()

    def next_batch():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        return b["gt"].to(dev, non_blocking=True), b["lq"].to(dev, non_blocking=True)

    def encode(gt, lq):
        with torch.no_grad():
            return vqgan.encode(gt) * sf_scale, vqgan.encode(lq) * sf_scale

    def rand_t(B, src):
        return src[torch.randint(0, len(src), (B,), device=dev)] if torch.is_tensor(src) \
            else torch.randint(0, src, (B,), device=dev)

    log_path = save_dir / "progress.jsonl"
    t0 = time.time()
    print(f"training: iters={args.iters} K={args.K} bs={args.bs}x{args.grad_accum} "
          f"T={T} nodes={args.nodes} amp={args.amp}")

    for step in range(args.iters):
        # ===== K fake-critic updates =====
        for k in range(args.K):
            opt_f.zero_grad(set_to_none=True)
            gt, lq = next_batch(); B = gt.shape[0]
            z0, z_y = encode(gt, lq)
            t_n = rand_t(B, nodes); eps = make_eps(student, z0)
            with torch.no_grad(), autocast():
                z_tn = diff.q_sample(z0, z_y, t_n)
                z0_hat = predict_x0(diff, student, z_tn, z_y, t_n, eps)
            t = rand_t(B, T)
            z_t = diff.q_sample(z0_hat, z_y, t)
            with autocast():
                x0_fake = predict_x0(diff, fake, z_t, z_y, t, eps)
                L_fake = F.mse_loss(x0_fake.float(), z0_hat.float())
                d_real = disc(fake.encode_features(z0, z_y, eps=eps))
                d_fake = disc(fake.encode_features(z0_hat.detach(), z_y, eps=eps))
                L_gan_d = (F.softplus(-d_real) + F.softplus(d_fake)).mean()
            (L_fake + args.lambda_gan * L_gan_d).backward()
            opt_f.step()

        # ===== 1 generator update =====
        opt_g.zero_grad(set_to_none=True)
        g_logs = {}
        for _ in range(args.grad_accum):
            gt, lq = next_batch(); B = gt.shape[0]
            z0, z_y = encode(gt, lq)
            t_n = rand_t(B, nodes); eps = make_eps(student, z0)
            z_tn = diff.q_sample(z0, z_y, t_n)
            with autocast():
                z0_hat = predict_x0(diff, student, z_tn, z_y, t_n, eps)
            t = rand_t(B, T)
            z_t = diff.q_sample(z0_hat.detach().float(), z_y, t)
            with torch.no_grad(), autocast():
                x0_teacher = predict_x0(diff, teacher, z_t, z_y, t)
                x0_fakep = predict_x0(diff, fake, z_t, z_y, t, eps)
                # DMD/SiD per-sample normalization (App C, "loss normalization … SiD"):
                # divide the (fake−teacher) push by a TEACHER-derived magnitude — NOT by the
                # (fake−teacher) diff's own norm (self-normalizing flattens the
                # distribution-matching signal so every sample gets a unit push regardless of
                # how wrong it is, and amplifies critic noise once fake≈teacher).
                grad = (x0_fakep - x0_teacher).float()
                if args.sid_denom == "student":
                    # official RSD release (trainer.py:1836): |teacher_x0 − student_x0| + 1e-8
                    denom = (x0_teacher.float() - z0_hat.detach().float()).abs().mean(
                        dim=[1, 2, 3], keepdim=True) + 1e-8
                else:
                    # original reconstruction: ‖z_t − f*_x0‖ floored 0.05 (turbo_dmd norm_floor)
                    denom = (z_t.float() - x0_teacher.float()).abs().mean(
                        dim=[1, 2, 3], keepdim=True).clamp_min(0.05)
                grad = grad / denom
            L_theta = 0.5 * F.mse_loss(z0_hat.float(), (z0_hat.float() - grad).detach())
            # single-step LPIPS path from z_T ~ N(z_y, kappa^2)
            z_TT = z_y + diff.kappa * torch.randn_like(z_y)
            tT = torch.full((B,), T - 1, device=dev, dtype=torch.long)
            with autocast():
                z0_single = predict_x0(diff, student, z_TT, z_y, tT, make_eps(student, z_y))
                x0_img = vqgan.decode(z0_single.float(), force_not_quantize=True).clamp(-1, 1)
                L_lpips = lp(x0_img, gt).mean()
                L_dc = dc_loss(x0_img.float(), gt.float())
                L_gan_g = F.softplus(-disc(fake.encode_features(z0_hat, z_y, eps=eps))).mean()
            loss = (L_theta + args.lambda_lpips * L_lpips + args.lambda_dc * L_dc
                    + args.lambda_gan * L_gan_g) / args.grad_accum
            loss.backward()
            g_logs = {k: v.detach().item() for k, v in
                      {"L_theta": L_theta, "L_lpips": L_lpips, "L_dc": L_dc, "L_gan_g": L_gan_g,
                       "L_fake": L_fake, "L_gan_d": L_gan_d}.items()}
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_g.step()
        ema_update(ema, student, args.ema)

        if step % args.log_every == 0:
            rate = (step + 1) / (time.time() - t0)
            line = {"step": step, **{k: round(v, 4) for k, v in g_logs.items()},
                    "it_s": round(rate, 3)}
            print(line)
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        if step > 0 and step % args.save_every == 0:
            torch.save({"ema": ema.state_dict(), "student": student.state_dict(), "step": step,
                        **ckpt_meta}, save_dir / f"rsd_student_{step}.pth")
        if args.max_steps and step + 1 >= args.max_steps:
            print(f"max_steps {args.max_steps} hit — stopping (smoke).")
            break

    torch.save({"ema": ema.state_dict(), "student": student.state_dict(), "step": args.iters,
                **ckpt_meta}, save_dir / "rsd_student_final.pth")
    print(f"done. saved to {save_dir}")


if __name__ == "__main__":
    main()
