"""Single-step RSD student inference + eval against the v2 teacher.

The student is a 1-step generator: ẑ0 = G_θ(z_T, z_y, T-1, ε), z_T ~ N(z_y, κ²I),
decoded by the VQGAN. Tiling is handled by chopping the LR latent. Eval mode scores
student (NFE=1) vs the released v2 teacher (NFE=15) vs bicubic on the Phase-0 set.
"""
import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from rsd_models import TEACHER_CKPT, make_eps, predict_x0  # noqa: E402
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
def _sample_tile(cfg, diff, vqgan, student, hr_t, align, device, amp=True):
    """One tile: reflect-pad to `align` (Swin needs dims % lq_size), encode -> 1-step
    student -> decode -> crop back. In/out 1x3xHxW in [-1,1].

    Heavy matmuls run under bf16 autocast by default (matches training's --amp, ~2x
    faster + ~half VRAM); the returned tile is cast back to fp32 for seam-averaging.

    The VAE is the VRAM peak (full pixel-res conv stack). When bf16, it runs in NATIVE
    bf16 *outside* autocast: autocast keeps GroupNorm on its fp32 list, which would
    materialize the full-res norm activations in fp32 (the dominant tensors). Native
    bf16 keeps them bf16 too — GroupNorm still reduces in fp32 internally, so it's safe.
    The student stays under autocast (matches training)."""
    h, w = hr_t.shape[2:]
    ph, pw = (-h) % align, (-w) % align
    if ph or pw:
        hr_t = F.pad(hr_t, (0, pw, 0, ph), mode="reflect")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    vdt = next(vqgan.parameters()).dtype   # bf16 when amp (see main), else fp32
    z_y = vqgan.encode(hr_t.to(vdt)) * cfg.diffusion.params.scale_factor
    z_T = z_y + diff.kappa * torch.randn_like(z_y)
    tT = torch.full((z_y.shape[0],), diff.num_timesteps - 1, device=device, dtype=torch.long)
    with ctx:
        z0 = predict_x0(diff, student, z_T, z_y, tT, make_eps(student, z_y))
    img = vqgan.decode(z0.to(vdt), force_not_quantize=True).clamp(-1, 1)
    return img[:, :, :h, :w].float()


@torch.no_grad()
def student_sr(cfg, diff, vqgan, student, lq_img, device, align=256, overlap=None,
               chop=512, tile_batch=1, amp=True):
    """1-step student SR on an HR-sized (upsampled-LR) tensor 1x3xHxW in [-1,1].

    The student is spatially same-resolution (the x4 lives in the residual-shift, not a
    spatial upscale), so we tile with sf=1. Large images are chopped into overlapping
    `chop`-px tiles and overlap-averaged (ImageSpliterTh) — both for VRAM and because the
    Swin attention is built for the lq_size grid.

    `align` is the Swin pad-alignment (each tile is reflect-padded to a multiple of it);
    `overlap` is the seam overlap that sets the tile stride (`chop - overlap`) and is
    decoupled from `align` so small tiles are possible (default = align).

    `tile_batch` stacks that many tiles into one forward (ImageSpliterTh.extra_bs) — small
    tiles underfill the GPU one-at-a-time, so batching amortizes launch/overhead for a big
    speedup; VRAM scales ~linearly with it, so it's a card-specific cap. Every tile is
    exactly chop×chop (chop is an align multiple), so the batch stacks with no ragged pad."""
    if overlap is None:
        overlap = align
    H, W = lq_img.shape[2:]
    if H > chop or W > chop:
        spliter = ImageSpliterTh(lq_img, chop, stride=chop - overlap, sf=1, extra_bs=tile_batch)
        for pch, info in spliter:
            spliter.update(_sample_tile(cfg, diff, vqgan, student, pch, align, device, amp), info)
        img = spliter.gather()
    else:
        img = _sample_tile(cfg, diff, vqgan, student, lq_img, align, device, amp)
    return ((img[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["x4", "x2"], default="x4",
                    help="scale family: x4 = released teacher schedule; x2 = our sr-train finetune. "
                         "Selects the config (hence sf) and the default ckpt_dir/out_dir.")
    ap.add_argument("--config", default=None, help="override the version's ResShift config yaml")
    ap.add_argument("--ckpt", default=None,
                    help="rsd_student_*.pth (uses EMA weights); default = most recent in --ckpt_dir")
    ap.add_argument("--ckpt_dir", default=None,
                    help="dir to pick the most recent ckpt from when --ckpt is unset "
                         "(default: output/sr/rsd for x4, output/sr/rsd_x2 for x2)")
    ap.add_argument("--eval", action="store_true", help="score vs teacher+bicubic on Phase-0 set")
    ap.add_argument("--in_dir", default=str(REPO / "sr" / "data" / "lr_eval"))
    ap.add_argument("--out_dir", default=None,
                    help="default: <ckpt_dir>/infer")
    ap.add_argument("--chop", type=int, default=256, help="tile size (px) for large images; must be a multiple of the align stride. 2048 single-tiles the 512->2048 eval (no overlap-redundancy ~2.2x compute saving); lower it (e.g. 1024) if VRAM-bound")
    ap.add_argument("--overlap", type=int, default=64,
                    help="tile seam overlap in px (default = the Swin align stride, currently 256). "
                         "Sets the tile stride = chop - overlap, which must be > 0 — so lowering it "
                         "is what lets you use a --chop at or near the align stride (e.g. --chop 256 "
                         "--overlap 128). Independent of Swin alignment; only --chop must be a "
                         "multiple of the align stride. Smaller overlap = fewer redundant tiles but "
                         "less seam blending.")
    ap.add_argument("--tile_batch", type=int, default=8,
                    help="stack this many chop-tiles into one forward (default 1 = serial). Small "
                         "tiles underfill the GPU one-at-a-time; batching amortizes launch overhead "
                         "for a big speedup. VRAM scales ~linearly, so tune to the card (e.g. "
                         "--chop 256 --tile_batch 8).")
    ap.add_argument("--no_bf16", action="store_true", help="disable bf16 autocast (inference matches training's --amp by default; ~2x slower + 2x VRAM if set)")
    ap.add_argument("--weights", choices=["ema", "student"], default="ema",
                    help="ema = smoothed (best late); student = raw (better for EARLY ckpts, "
                         "EMA still drags the teacher-init there)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(student, dynamic=True) — marks spatial dims dynamic so "
                         "the eval's all-different tile shapes share one graph (avoids a recompile "
                         "per image). Swin window_partition may force graph breaks; first tile pays "
                         "warmup. Worth it for many tiles, a loss for a handful.")
    args = ap.parse_args()
    config_path, _ = M.resolve_version(args.version, args.config)
    default_sub = "rsd" if args.version == "x4" else f"rsd_{args.version}"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else REPO / "output" / "sr" / default_sub
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_dir / "infer"
    args.out_dir = str(out_dir)
    ckpt = Path(args.ckpt) if args.ckpt else latest_ckpt(ckpt_dir)
    print(f"version={args.version} config={config_path.name} ckpt={ckpt}")
    dev = M.DEVICE
    cfg = M.load_configs(config_path)
    diff = M.build_diffusion(cfg)
    vqgan = M.build_autoencoder(cfg, dev)
    if not args.no_bf16:
        # native bf16 VAE — halves the full-res GroupNorm activations that autocast would
        # otherwise keep in fp32 (the VRAM peak). Codebook is unused (force_not_quantize).
        vqgan = vqgan.to(torch.bfloat16)
    sd = torch.load(ckpt, map_location="cpu")
    # rebuild the noise-injection arch the ckpt was trained with (ckpt_meta from train.py;
    # old ckpts predate it -> default to the "add" reconstruction).
    student = M.build_generator(cfg, str(TEACHER_CKPT), dev, pretrained=False,
                                noise_mode=sd.get("noise_mode", "add"),
                                noise_channels=sd.get("noise_channels"))
    key = args.weights if args.weights in sd else ("ema" if "ema" in sd else None)
    student.load_state_dict(sd[key] if key else sd, strict=True)
    print(f"weights: {key or 'raw'} | noise_mode: {student.noise_mode}({student.noise_channels}ch)")
    student.eval()
    if args.compile:
        # Bump the recompile ceiling: dynamic=True still specializes on a few discrete
        # guards (Swin window_partition reshapes), so the 24-distinct-shape eval can trip
        # the default limit of 8 and silently fall back to eager. (No backward pass here,
        # so the ContextVar-reversion gotcha that bites training doesn't apply.)
        import torch._dynamo as dynamo
        for attr in ("recompile_limit", "cache_size_limit"):  # renamed across torch versions
            if hasattr(dynamo.config, attr):
                setattr(dynamo.config, attr, 64)
        student = torch.compile(student, dynamic=True)
        print("torch.compile(dynamic=True) enabled on student")
    scale = cfg.diffusion.params.sf
    # Pixel alignment the SwinUNet needs: the deepest level runs at latent / 2^(L-1) with a
    # fixed window=8, so latent must be divisible by window*2^(L-1); ×sf for the pixel grid.
    n_levels = len(cfg.model.params.get("channel_mult", [1, 2, 2, 4]))
    align = int(scale) * int(cfg.model.params.window_size) * (2 ** (n_levels - 1))
    if args.chop % align:
        raise SystemExit(f"--chop {args.chop} must be a multiple of {align} (sf×window×2^{n_levels-1})")
    overlap = args.overlap if args.overlap is not None else align
    if not 0 <= overlap < args.chop:
        raise SystemExit(f"--overlap {overlap} must be in [0, chop={args.chop}) so the tile "
                         f"stride (chop - overlap) is > 0 (chop == overlap => zero stride)")
    if args.tile_batch < 1:
        raise SystemExit(f"--tile_batch {args.tile_batch} must be >= 1")
    print(f"tiling: chop={args.chop} overlap={overlap} stride={args.chop - overlap} "
          f"align={align} tile_batch={args.tile_batch}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    in_dir = Path(args.in_dir)
    rows = []
    try:
        import pyiqa
        musiq = pyiqa.create_metric("musiq", device=dev)
    except Exception:  # noqa: BLE001
        musiq = None

    lr_paths = sorted(in_dir.glob("*.png"))
    pbar = tqdm(lr_paths, desc="student SR", unit="img")
    for lr_path in pbar:
        lq_t, _ = load_lr(lr_path, dev, scale)
        sr = student_sr(cfg, diff, vqgan, student, lq_t, dev, align=align, overlap=overlap,
                        chop=args.chop, tile_batch=args.tile_batch, amp=not args.no_bf16)
        Image.fromarray(sr).save(out_dir / lr_path.name)
        row = {"stem": lr_path.stem}
        if musiq is not None:
            t = torch.from_numpy(sr.astype(np.float32) / 255).permute(2, 0, 1)[None]
            row["musiq_student"] = round(float(musiq(t).item()), 3)
            pbar.set_postfix(musiq=row["musiq_student"])
        rows.append(row)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None
    # alloc_retries > 0 means we hit the OOM-edge retry path (free-cache + cudaFree + retry):
    # the tell-tale of a memory-bound regime where a bigger --chop is SLOWER despite fewer
    # FLOPs. reserved >> allocated also signals fragmentation/large-segment churn.
    if dev == "cuda":
        ms = torch.cuda.memory_stats()
        summary_mem = {"peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                       "num_alloc_retries": ms.get("num_alloc_retries", 0),
                       "num_ooms": ms.get("num_ooms", 0)}
    else:
        summary_mem = {}
    summary = {"n": len(rows), "ckpt": str(ckpt), "chop": args.chop, "peak_vram_gb": peak_gb,
               **summary_mem,
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
