"""Art HR dataset + degradation for RSD distillation.

Yields (gt, lq) pairs in [-1,1], both at HR size (256²): gt = a random HR crop from
our art; lq = bicubic ×4 down + light JPEG/blur, then bicubic-upsampled back to HR
(ResShift feeds the LR upsampled-to-HR; the ×4 lives in the degradation). This light
degradation is art-appropriate (proposal §1b) and keeps the v2 teacher in-distribution
(Phase 0). Swap in ResShift's Real-ESRGAN pipeline later if needed.
"""
import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[2]
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ArtSRDataset(Dataset):
    def __init__(self, src=None, gt_size=256, scale=4, length=20000,
                 jpeg_range=(55, 95), blur_prob=0.3):
        self.src = Path(src or (REPO / "image_dataset")).resolve()
        self.files = sorted(p for p in self.src.rglob("*") if p.suffix.lower() in EXTS)
        if not self.files:
            raise SystemExit(f"no images under {self.src}")
        self.gt_size = gt_size
        self.scale = scale
        self.length = length
        self.jpeg_range = jpeg_range
        self.blur_prob = blur_prob
        print(f"ArtSRDataset: {len(self.files)} source images, virtual length {length}")

    def __len__(self):
        return self.length

    def _rand_crop(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        g = self.gt_size
        if w < g or h < g:
            s = g / min(w, h)
            im = im.resize((max(g, int(w * s + 1)), max(g, int(h * s + 1))), Image.LANCZOS)
            w, h = im.size
        x, y = random.randint(0, w - g), random.randint(0, h - g)
        return im.crop((x, y, x + g, y + g))

    def _degrade(self, gt: Image.Image) -> Image.Image:
        g = self.gt_size
        lr = gt.resize((g // self.scale, g // self.scale), Image.BICUBIC)
        if random.random() < self.blur_prob:
            from PIL import ImageFilter
            lr = lr.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
        lr = _jpeg(lr, random.randint(*self.jpeg_range))
        return lr.resize((g, g), Image.BICUBIC)  # upsample back to HR size

    def __getitem__(self, idx):
        for _ in range(8):
            try:
                im = Image.open(random.choice(self.files)).convert("RGB")
                break
            except Exception:  # noqa: BLE001
                continue
        gt = self._rand_crop(im)
        if random.random() < 0.5:
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        lq = self._degrade(gt)
        to_t = lambda p: torch.from_numpy(np.asarray(p, np.float32) / 127.5 - 1).permute(2, 0, 1)
        return {"gt": to_t(gt), "lq": to_t(lq)}
