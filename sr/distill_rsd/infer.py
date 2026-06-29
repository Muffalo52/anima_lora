"""Single-step RSD student inference + eval against the v2 teacher.

The student is a 1-step generator: ẑ0 = G_θ(z_T, z_y, T-1, ε), z_T ~ N(z_y, κ²I),
decoded by the VQGAN. Tiling is handled by chopping the LR latent. Eval mode scores
student (NFE=1) vs the released v2 teacher (NFE=15) vs bicubic on the Phase-0 set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from rsd_models import TEACHER_CKPT, predict_x0  # noqa: E402
from utils.util_image import ImageSpliterTh  # noqa: E402  (resshift path added by rsd_models)

REPO = M.REPO
CKPT_DIR = REPO / "output" / "sr" / "rsd"


def latest_ckpt(ckpt_dir: Path) -> Path:
    """Most recently written rsd_student_*.pth (final or numbered)."""
    cands = sorted(ckpt_dir.glob("rsd_student_*.pth"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(f"no rsd_student_*.pth under {ckpt_dir} — train first or pass --ckpt")
    return cands[-1]


def load_lr(path, device, scale):
    """Load LR image, bicubic-upsample x scale to HR size (ResShift convention)."""
    im = Image.open(path).convert("RGB")
    up = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(up, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    return t.to(device), up


@torch.no_grad()
def _sample_tile(cfg, diff, vqgan, student, hr_t, offset, device):
    """One tile: reflect-pad to `offset` (Swin needs dims % lq_size), encode -> 1-step
    student -> decode -> crop back. In/out 1x3xHxW in [-1,1]."""
    h, w = hr_t.shape[2:]
    ph, pw = (-h) % offset, (-w) % offset
    if ph or pw:
        hr_t = F.pad(hr_t, (0, pw, 0, ph), mode="reflect")
    z_y = vqgan.encode(hr_t) * cfg.diffusion.params.scale_factor
    z_T = z_y + diff.kappa * torch.randn_like(z_y)
    tT = torch.full((z_y.shape[0],), diff.num_timesteps - 1, device=device, dtype=torch.long)
    z0 = predict_x0(diff, student, z_T, z_y, tT, torch.randn_like(z_y))
    img = vqgan.decode(z0.float(), force_not_quantize=True).clamp(-1, 1)
    return img[:, :, :h, :w]


@torch.no_grad()
def student_sr(cfg, diff, vqgan, student, lq_img, device, offset=64, chop=512):
    """1-step student SR on an HR-sized (upsampled-LR) tensor 1x3xHxW in [-1,1].

    The student is spatially same-resolution (the x4 lives in the residual-shift, not a
    spatial upscale), so we tile with sf=1. Large images are chopped into overlapping
    `chop`-px tiles and overlap-averaged (ImageSpliterTh) — both for VRAM and because the
    Swin attention is built for the lq_size grid."""
    H, W = lq_img.shape[2:]
    if H > chop or W > chop:
        spliter = ImageSpliterTh(lq_img, chop, stride=chop - offset, sf=1)
        for pch, info in spliter:
            spliter.update(_sample_tile(cfg, diff, vqgan, student, pch, offset, device), info)
        img = spliter.gather()
    else:
        img = _sample_tile(cfg, diff, vqgan, student, lq_img, offset, device)
    return ((img[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="rsd_student_*.pth (uses EMA weights); default = most recent in --ckpt_dir")
    ap.add_argument("--ckpt_dir", default=str(CKPT_DIR),
                    help="dir to pick the most recent ckpt from when --ckpt is unset")
    ap.add_argument("--eval", action="store_true", help="score vs teacher+bicubic on Phase-0 set")
    ap.add_argument("--in_dir", default=str(REPO / "sr" / "data" / "lr_eval"))
    ap.add_argument("--out_dir", default=str(REPO / "output" / "sr" / "rsd" / "infer"))
    ap.add_argument("--chop", type=int, default=1024, help="tile size (px) for large images; must be a multiple of lq_size (lower it if VRAM-bound)")
    ap.add_argument("--weights", choices=["ema", "student"], default="ema",
                    help="ema = smoothed (best late); student = raw (better for EARLY ckpts, "
                         "EMA still drags the teacher-init there)")
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(Path(args.ckpt_dir))
    print(f"ckpt: {ckpt}")
    dev = M.DEVICE
    cfg = M.load_configs()
    diff = M.build_diffusion(cfg)
    vqgan = M.build_autoencoder(cfg, dev)
    student = M.build_generator(cfg, str(TEACHER_CKPT), dev)
    sd = torch.load(ckpt, map_location="cpu")
    key = args.weights if args.weights in sd else ("ema" if "ema" in sd else None)
    student.load_state_dict(sd[key] if key else sd, strict=True)
    print(f"weights: {key or 'raw'}")
    student.eval()
    scale = cfg.diffusion.params.sf
    # Pixel alignment the SwinUNet needs: the deepest level runs at latent / 2^(L-1) with a
    # fixed window=8, so latent must be divisible by window*2^(L-1); ×sf for the pixel grid.
    n_levels = len(cfg.model.params.get("channel_mult", [1, 2, 2, 4]))
    align = int(scale) * int(cfg.model.params.window_size) * (2 ** (n_levels - 1))
    if args.chop % align:
        raise SystemExit(f"--chop {args.chop} must be a multiple of {align} (sf×window×2^{n_levels-1})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)
    rows = []
    try:
        import pyiqa
        musiq = pyiqa.create_metric("musiq", device=dev)
    except Exception:  # noqa: BLE001
        musiq = None

    for lr_path in sorted(in_dir.glob("*.png")):
        lq_t, _ = load_lr(lr_path, dev, scale)
        sr = student_sr(cfg, diff, vqgan, student, lq_t, dev, offset=align, chop=args.chop)
        Image.fromarray(sr).save(out_dir / lr_path.name)
        row = {"stem": lr_path.stem}
        if musiq is not None:
            t = torch.from_numpy(sr.astype(np.float32) / 255).permute(2, 0, 1)[None]
            row["musiq_student"] = round(float(musiq(t).item()), 3)
        rows.append(row)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None
    summary = {"n": len(rows), "ckpt": str(ckpt), "chop": args.chop, "peak_vram_gb": peak_gb,
               "musiq_student_mean": round(float(np.mean([r["musiq_student"] for r in rows
                                                          if "musiq_student" in r])), 3) if musiq else None}
    (out_dir / "infer_summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"student 1-step outputs -> {out_dir}")
    if args.eval:
        print("(teacher/bicubic comparison: run sr-phase0 metrics on the same set; "
              "student MUSIQ above is the 1-step number.)")


if __name__ == "__main__":
    main()
