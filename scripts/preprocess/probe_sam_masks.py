"""Dump SAM3's raw per-instance masks for one image, colour-coded.

Answers "is the mask actually broken, or are we mishandling it?" — the two look
identical downstream once ``crop_instance`` has blanked a crop white. So this
bypasses the audit entirely: run the ``girl`` prompt, print each mask's shape /
dtype / value range / fill against the image it is supposed to align with, and
render every instance in its own colour over the original.

If the overlay shows a clean silhouette, the mask is fine and our indexing is
wrong. If it shows the same speckle the crop did, the mask really is that bad.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Monkey-patch numpy for sam3 compatibility (upstream pins numpy<2 and uses np.bool)
if not hasattr(np, "bool"):
    np.bool = np.bool_

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from library.env import resolve_under_home  # noqa: E402

COLORS = [
    (255, 60, 60),
    (60, 140, 255),
    (255, 210, 60),
    (170, 100, 255),
    (60, 230, 190),
    (255, 120, 200),
    (150, 255, 90),
    (255, 165, 80),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", help="Path to the image (repo-relative ok)")
    p.add_argument("--prompt", default="girl")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Processor confidence floor — low on purpose, we want to see every "
        "proposal including the ones the audit filters out",
    )
    p.add_argument("--checkpoint", default="models/sam3/sam3.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="post_image_dataset/captions/mask_probe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    image_path = resolve_under_home(args.image)
    out_dir = resolve_under_home(args.out) / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    print(f"image: {image_path}  size={image.size} (W,H)")

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        checkpoint_path=str(resolve_under_home(args.checkpoint)),
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        out = processor.set_text_prompt(prompt=args.prompt, state=state)

    print(f"returned keys: {sorted(out.keys())}")
    boxes, scores, masks = out["boxes"], out["scores"], out.get("masks")
    print(f"instances: {len(boxes)}   masks present: {masks is not None}")

    overlay = np.asarray(image).astype(np.float32)
    legend: list[tuple[int, tuple[int, int, int], float]] = []

    for i, (box, score) in enumerate(zip(boxes, scores)):
        color = COLORS[i % len(COLORS)]
        coords = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
        print(f"\n--- instance {i}  score={float(score):.4f}")
        print(f"    box   = {[round(v, 1) for v in coords]}")
        if masks is None or i >= len(masks):
            print("    mask  = MISSING")
            continue
        m = masks[i]
        raw = m.float().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        print(f"    mask  shape={raw.shape} dtype={raw.dtype}")
        print(
            f"    values min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f}"
        )
        flat = raw[0] if raw.ndim == 3 else raw
        print(f"    2D shape={flat.shape}  vs image (H,W)=({image.height},{image.width})"
              f"  {'ALIGNED' if flat.shape == (image.height, image.width) else '*** MISMATCH ***'}")
        binary = flat > 0.5
        fill = float(binary.mean())
        bx = [max(0, int(coords[0])), max(0, int(coords[1])),
              min(image.width, int(coords[2])), min(image.height, int(coords[3]))]
        in_box = binary[bx[1]:bx[3], bx[0]:bx[2]]
        box_fill = float(in_box.mean()) if in_box.size else 0.0
        print(f"    fill@0.5 = {fill:.4f} of canvas   {box_fill:.4f} of its own box")
        legend.append((i, color, float(score)))

        Image.fromarray((binary * 255).astype(np.uint8)).save(
            out_dir / f"mask_{i}_raw.png"
        )
        if flat.shape == (image.height, image.width):
            sel = binary
            overlay[sel] = overlay[sel] * 0.45 + np.array(color, np.float32) * 0.55

    composite = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(composite)
    for i, (box, _) in enumerate(zip(boxes, scores)):
        color = COLORS[i % len(COLORS)]
        coords = [float(v) for v in (box.tolist() if torch.is_tensor(box) else box)]
        draw.rectangle(coords, outline=color, width=6)
        draw.rectangle([coords[0], coords[1], coords[0] + 44, coords[1] + 44], fill=color)
        draw.text((coords[0] + 14, coords[1] + 12), str(i), fill=(0, 0, 0))
    composite.save(out_dir / "overlay.png")

    print(f"\nlegend: {[(i, c, round(s, 3)) for i, c, s in legend]}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
