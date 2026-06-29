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
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from train import predict_x0  # noqa: E402

REPO = M.RESSHIFT.parent
TEACHER_CKPT = M.RESSHIFT / "weights" / "resshift_realsrx4_s15_v2.pth"


def load_lr(path, device, scale):
    """Load LR image, bicubic-upsample x scale to HR size (ResShift convention)."""
    im = Image.open(path).convert("RGB")
    up = im.resize((im.width * scale, im.height * scale), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(up, np.float32) / 127.5 - 1).permute(2, 0, 1)[None]
    return t.to(device), up


@torch.no_grad()
def student_sr(cfg, diff, vqgan, student, lq_img, device):
    z_y = vqgan.encode(lq_img) * cfg.diffusion.params.scale_factor
    z_T = z_y + diff.kappa * torch.randn_like(z_y)
    T = diff.num_timesteps
    tT = torch.full((z_y.shape[0],), T - 1, device=device, dtype=torch.long)
    z0 = predict_x0(diff, student, z_T, z_y, tT, torch.randn_like(z_y))
    img = vqgan.decode(z0.float(), force_not_quantize=True).clamp(-1, 1)
    return ((img[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="rsd_student_*.pth (uses EMA weights)")
    ap.add_argument("--eval", action="store_true", help="score vs teacher+bicubic on Phase-0 set")
    ap.add_argument("--in_dir", default=str(REPO / "sr" / "data" / "lr_eval"))
    ap.add_argument("--out_dir", default=str(REPO / "output" / "sr" / "rsd" / "infer"))
    args = ap.parse_args()
    dev = "cuda"
    cfg = M.load_configs()
    diff = M.build_diffusion(cfg)
    vqgan = M.build_autoencoder(cfg, dev)
    student = M.build_generator(cfg, str(TEACHER_CKPT), dev)
    sd = torch.load(args.ckpt, map_location="cpu")
    student.load_state_dict(sd["ema"] if "ema" in sd else sd, strict=True)
    student.eval()
    scale = cfg.diffusion.params.sf

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)
    rows = []
    try:
        import pyiqa
        musiq = pyiqa.create_metric("musiq", device=dev)
    except Exception:  # noqa: BLE001
        musiq = None
    hr_dir = REPO / "sr" / "data" / "hr_eval"

    for lr_path in sorted(in_dir.glob("*.png")):
        lq_t, _ = load_lr(lr_path, dev, scale)
        sr = student_sr(cfg, diff, vqgan, student, lq_t, dev)
        Image.fromarray(sr).save(out_dir / lr_path.name)
        row = {"stem": lr_path.stem}
        if musiq is not None:
            t = torch.from_numpy(sr.astype(np.float32) / 255).permute(2, 0, 1)[None]
            row["musiq_student"] = round(float(musiq(t).item()), 3)
        rows.append(row)
    summary = {"n": len(rows), "ckpt": args.ckpt,
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
