"""Phase 0 — learned-preservation DirectEdit (EasyControl cond stream vs V-injection).

Zero-training probe: can an existing EasyControl adapter (the inpaint
checkpoint fed a hole-free cond == the source image, i.e. a "copy everything"
reference) serve as a learned source-preservation prior on DirectEdit's
Δz-anchored edit trajectories, replacing the hand-tuned V-injection?

Expected regime (see docs/proposal discussion): all shipped EC adapters were
trained on spatially-aligned pairs, so the prior is position-locked — the probe
targets IN-PLACE attribute edits (default: "glasses"), where position-locked
appearance carry is exactly what V-injection is hand-tuned to provide.

Arms per image (all run with --no_compile_blocks for wall-time parity):

  recon_base  ψ_tar == ψ_src, CFG 1, no EC    reconstruction reference
  recon_ec    ψ_tar == ψ_src, CFG 1, EC on    gate: must ≈ recon_base (the
                                              cond stream must not break the
                                              anchor's exactness)
  base_t0     edit, t_inj=0, no EC            pure-anchor baseline
  vinj_t2     edit, t_inj=2, no EC            V-injection (edit.py default)
  vinj_t6     edit, t_inj=6, no EC            stronger V-injection (~T/5)
  ec_s<X>     edit, t_inj=0, EC scale X       the probe arms

Metrics: pixel MSE vs the (bucket-resized) source per arm. For recon arms MSE
IS the verdict (gate: recon_ec ≤ 2× recon_base). For edit arms MSE is only a
preservation proxy — the edit-quality verdict is human, via the per-image
contact sheets (grid.png). Per repo culture, no CMMD at this n.

Phase 1a (--phase 1a): masked-cond probe — feed the inpaint prior what it was
trained on (gray hole over the intended edit region, face box per image) and
run at b_offset 0, no tuning. Adds an outside/inside-hole MSE split for every
arm; gate = ec_mask's outside-hole MSE ≤ 2× recon_base's, per image.

Usage:
    uv run python project/directedit_ec/bench/run_bench.py --smoke      # 1 img, 4 arms
    uv run python project/directedit_ec/bench/run_bench.py              # full sweep
    uv run python project/directedit_ec/bench/run_bench.py --edit smile --n_images 4
    uv run python project/directedit_ec/bench/run_bench.py --phase 1a   # masked-cond
    uv run python project/directedit_ec/bench/run_bench.py --phase 1b   # edit types
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.env import default_checkpoints  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

RESIZED_DIR = REPO_ROOT / "post_image_dataset" / "resized"
DEFAULT_EC_WEIGHT = (
    REPO_ROOT / "output" / "ckpt" / "methods" / "anima_inpaint.safetensors"
)


@dataclass(frozen=True)
class Arm:
    name: str
    t_inj: int
    ec_scale: Optional[float]  # None = EC off
    edit: bool  # False = reconstruction (ψ_tar == ψ_src)
    guidance: Optional[float] = None  # None = args.guidance_scale
    b_offset: Optional[float] = None  # additive offset on the learned b_cond gates
    mask: bool = False  # gray-hole the EC cond over the edit region (Phase 1a)
    anchor_mask: bool = False  # drop the Δz anchor inside the region (Eq. 12)


# Phase 1a face-region boxes (fractional x0, y0, x1, y1 of the frame), placed
# by eye on the Phase-0 gate set — the "user box" mask source from the
# proposal. Unknown stems fall back to --mask_box.
FACE_BOXES: dict[str, tuple[float, float, float, float]] = {
    "dan_9596032": (0.28, 0.10, 0.74, 0.44),
    "10473210": (0.52, 0.15, 0.88, 0.40),
    "7538087": (0.24, 0.08, 0.64, 0.30),
}

# Phase 1b edit-type matrix on the gate set: per-image concrete edits for the
# four types (REMOVE a present tag / REPLACE hair color / expression / a
# geometry-pose control that the position-locked prior is EXPECTED to fail).
# 'drop' tags must exist in the caption (fail-loud); 'box' is the ec_mask
# hole for that edit (fractional x0, y0, x1, y1), placed by eye.
FULL_BOX = (0.02, 0.02, 0.98, 0.98)
EDITS_1B: dict[str, dict[str, dict]] = {
    "dan_9596032": {
        "remove": {
            "drop": ["hair ornament", "kanzashi", "tassel"],
            "add": [],
            "box": (0.60, 0.20, 0.90, 0.43),  # kanzashi + bun ornaments
        },
        "replace": {
            "drop": ["pink hair"],
            "add": ["blue hair"],
            "box": (0.25, 0.03, 0.80, 0.48),  # hair incl. bun
        },
        "expression": {
            "drop": ["parted lips"],
            "add": ["smile"],
            "box": FACE_BOXES["dan_9596032"],
        },
        "geometry": {"drop": ["squatting"], "add": ["standing"], "box": FULL_BOX},
    },
    "10473210": {
        "remove": {
            "drop": ["halo"],
            "add": [],
            "box": (0.36, 0.0, 0.63, 0.06),  # halo glow above the head
        },
        "replace": {
            "drop": ["white hair"],
            "add": ["black hair"],
            "box": (0.20, 0.02, 0.96, 0.60),
        },
        "expression": {
            "drop": [],
            "add": ["smile"],
            "box": FACE_BOXES["10473210"],
        },
        "geometry": {"drop": [], "add": ["arms up"], "box": FULL_BOX},
    },
    "7538087": {
        "remove": {
            "drop": ["blush"],
            "add": [],
            "box": FACE_BOXES["7538087"],
        },
        "replace": {
            "drop": ["brown hair"],
            "add": ["blonde hair"],
            "box": (0.20, 0.02, 0.70, 0.30),
        },
        "expression": {
            "drop": ["smile"],
            "add": ["parted lips"],
            "box": FACE_BOXES["7538087"],
        },
        "geometry": {"drop": ["cowboy shot"], "add": ["sitting"], "box": FULL_BOX},
    },
}


def apply_edit(cap: str, drop: list[str], add: list[str]) -> str:
    """Build ψ_tar from the caption by dropping/appending exact tags."""
    tags = [t.strip() for t in cap.split(",")]
    missing = [d for d in drop if d not in tags]
    if missing:
        raise SystemExit(f"edit drop tags not in caption: {missing}")
    tags = [t for t in tags if t not in drop]
    return ", ".join(tags + list(add))


def build_arms(
    ec_scales: list[float], b_offsets: list[float], smoke: bool, phase: str
) -> list[Arm]:
    if phase == "1a":
        # Masked-cond probe vs the Phase-0 sweet spots. ec_mask runs at the
        # trained operating point (cond_scale 1.0, b_offset 0 — no tuning);
        # the hole is the dial.
        return [
            Arm("recon_base", 0, None, edit=False, guidance=1.0),
            Arm("recon_ec", 0, 1.0, edit=False, guidance=1.0),
            Arm("base_t0", 0, None, edit=True),
            Arm("vinj_t6", 6, None, edit=True),
            Arm("ec_b-1", 0, 1.0, edit=True, b_offset=-1.0),
            Arm("ec_b-2", 0, 1.0, edit=True, b_offset=-2.0),
            Arm("ec_mask", 0, 1.0, edit=True, mask=True),
            # The Δz anchor is global — it pulls the hole back to the source
            # even after the EC prior releases it. These two split the blame:
            # anch_only = Eq.12 anchor mask alone (no EC), ec_mask_anch = EC
            # hole + anchor mask (both mechanisms release the edit region).
            Arm("anch_only", 0, None, edit=True, anchor_mask=True),
            Arm("ec_mask_anch", 0, 1.0, edit=True, mask=True, anchor_mask=True),
        ]
    if phase == "1b":
        # Edit-type generalization: the Phase-0/1a recipes head-to-head, per
        # edit type. Recon arms are omitted — the 1b gate is render-judged
        # (edit lands + composition held), MSE numbers are context only.
        # ec_mask_anch is the 1a keeper (EC hole + Eq. 12 anchor mask):
        # cond-hole alone landed the edit on only 1/3 images (the global Δz
        # anchor pulls the hole back to the source), so plain ec_mask isn't
        # re-swept here.
        return [
            Arm("base_t0", 0, None, edit=True),
            Arm("vinj_t6", 6, None, edit=True),
            Arm("ec_b-1", 0, 1.0, edit=True, b_offset=-1.0),
            Arm("ec_b-2", 0, 1.0, edit=True, b_offset=-2.0),
            Arm("ec_mask_anch", 0, 1.0, edit=True, mask=True, anchor_mask=True),
        ]
    arms = (
        [
            Arm("recon_base", 0, None, edit=False, guidance=1.0),
            Arm("recon_ec", 0, 1.0, edit=False, guidance=1.0),
            Arm("base_t0", 0, None, edit=True),
            Arm("vinj_t2", 2, None, edit=True),
            Arm("vinj_t6", 6, None, edit=True),
        ]
        + [Arm(f"ec_s{s:g}", 0, s, edit=True) for s in ec_scales]
        # b_cond-offset arms run at full cond_scale (the trained operating
        # point — cond_scale<1 disengages the prior, see phase0-full) and dial
        # the cond softmax mass down instead: each -1 ≈ e× less cond attention.
        + [Arm(f"ec_b{o:g}", 0, 1.0, edit=True, b_offset=o) for o in b_offsets]
    )
    if smoke:
        keep = {"recon_base", "recon_ec", "base_t0", f"ec_s{ec_scales[-1]:g}"}
        arms = [a for a in arms if a.name in keep]
    return arms


def write_face_mask(
    image: Path,
    box: tuple[float, float, float, float],
    out_path: Path,
) -> Path:
    """Write a white-box-on-black mask PNG at the source image's size."""
    import numpy as np
    from PIL import Image

    with Image.open(image) as im:
        w, h = im.size
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = 255
    Image.fromarray(m).save(out_path)
    return out_path


def masked_cond_preview(image: Path, mask_png: Path, out_path: Path) -> Path:
    """The cond image edit.py will actually feed the prior (gray-filled hole)."""
    import numpy as np
    from PIL import Image

    img = np.asarray(Image.open(image).convert("RGB")).copy()
    hole = np.asarray(Image.open(mask_png).convert("L")) > 127
    img[hole] = 128
    Image.fromarray(img).save(out_path)
    return out_path


def pick_images(n: int, seed: int) -> list[Path]:
    """Deterministically pick n resized images whose caption contains 1girl
    (the in-place edit set — 'glasses' etc. need a face to land on)."""
    candidates = []
    for png in sorted(RESIZED_DIR.rglob("*.png")):
        txt = png.with_suffix(".txt")
        if not txt.is_file():
            continue
        cap = txt.read_text(encoding="utf-8").strip()
        if "1girl" in cap and len(cap) < 900:
            candidates.append(png)
    if not candidates:
        raise SystemExit(f"no captioned 1girl images under {RESIZED_DIR}")
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def arm_argv(
    arm: Arm,
    image: Path,
    cap: str,
    tar: str,
    out_dir: Path,
    args,
    ck,
    mask_png: Optional[Path] = None,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "edit.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--image",
        str(image),
        "--prompt_src",
        cap,
        "--prompt_tar",
        tar if arm.edit else cap,
        "--save_path",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--infer_steps",
        str(args.infer_steps),
        "--guidance_scale",
        str(arm.guidance if arm.guidance is not None else args.guidance_scale),
        "--t_inj",
        str(arm.t_inj),
        "--no_compile_blocks",
    ]
    if arm.ec_scale is not None:
        argv += [
            "--easycontrol_weight",
            str(args.ec_weight),
            "--easycontrol_scale",
            str(arm.ec_scale),
        ]
    if arm.b_offset is not None:
        argv += ["--easycontrol_b_offset", str(arm.b_offset)]
    if arm.mask:
        assert mask_png is not None, f"arm {arm.name} needs a mask"
        argv += ["--easycontrol_mask", str(mask_png)]
    if arm.anchor_mask:
        assert mask_png is not None, f"arm {arm.name} needs a mask"
        argv += ["--mask", str(mask_png)]
    return argv


def mse_vs_source(
    out_png: Path, source_png: Path, mask_png: Optional[Path] = None
) -> dict:
    """MSE vs the source; with a mask also split into outside/inside-hole.

    Returns {"full": f, "outside": o|None, "inside": i|None}. outside is the
    Phase-1a preservation number (the prior should clamp there); inside is
    where the edit is supposed to move pixels — big is expected, not bad.
    """
    import numpy as np
    from PIL import Image

    out = Image.open(out_png).convert("RGB")
    src = Image.open(source_png).convert("RGB").resize(out.size, Image.LANCZOS)
    a = np.asarray(out, dtype=np.float64) / 255.0
    b = np.asarray(src, dtype=np.float64) / 255.0
    err = (a - b) ** 2
    res = {"full": float(err.mean()), "outside": None, "inside": None}
    if mask_png is not None:
        hole = (
            np.asarray(
                Image.open(mask_png).convert("L").resize(out.size, Image.NEAREST)
            )
            > 127
        )
        res["outside"] = float(err[~hole].mean())
        res["inside"] = float(err[hole].mean())
    return res


def make_grid(
    rows: list[tuple[str, list[tuple[str, Path]]]], out_path: Path, thumb_h: int = 384
) -> None:
    """rows = [(row_label, [(col_label, image_path), ...])]. Missing images
    render as a grey placeholder so a failed arm stays visible."""
    from PIL import Image, ImageDraw

    label_h = 22
    cols = max(len(r[1]) for r in rows)
    # Thumb each image to a common height; widths vary per image, use per-row max.
    thumbs: list[list[tuple[str, Optional[Image.Image]]]] = []
    for _, cells in rows:
        row_imgs = []
        for label, p in cells:
            if p is not None and p.is_file():
                im = Image.open(p).convert("RGB")
                w = int(im.size[0] * thumb_h / im.size[1])
                row_imgs.append((label, im.resize((w, thumb_h), Image.LANCZOS)))
            else:
                row_imgs.append((label, None))
        thumbs.append(row_imgs)
    cell_w = max(
        (im.size[0] for row in thumbs for _, im in row if im is not None),
        default=thumb_h,
    )
    cell = (cell_w, thumb_h + label_h)
    canvas = Image.new("RGB", (cell[0] * cols, cell[1] * len(rows)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for r, row_imgs in enumerate(thumbs):
        for c, (label, im) in enumerate(row_imgs):
            x, y = c * cell[0], r * cell[1]
            if im is not None:
                canvas.paste(im, (x + (cell_w - im.size[0]) // 2, y + label_h))
            else:
                draw.rectangle(
                    [x, y + label_h, x + cell_w, y + cell[1]], fill=(60, 60, 60)
                )
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255))
    canvas.save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Explicit source images (default: auto-pick)",
    )
    p.add_argument("--n_images", type=int, default=3)
    p.add_argument(
        "--edit", default="glasses", help="In-place edit tag appended to the caption"
    )
    p.add_argument("--ec_weight", default=str(DEFAULT_EC_WEIGHT))
    p.add_argument("--ec_scales", default="0.5,1.0")
    p.add_argument(
        "--ec_b_offsets",
        default="",
        help="Comma-separated additive b_cond offsets; each spawns an "
        "ec_b<off> arm at cond_scale 1.0 (e.g. '-1,-2,-3').",
    )
    p.add_argument(
        "--phase",
        default="0",
        choices=["0", "1a", "1b"],
        help="'1a' = masked-cond probe: fixed arm set {recon_base, recon_ec, "
        "base_t0, vinj_t6, ec_b-1, ec_b-2, ec_mask} with per-image face-box "
        "masks + outside/inside-hole MSE split (proposal Phase 1a). "
        "'1b' = edit-type matrix (EDITS_1B: remove/replace/expression/"
        "geometry per image) × {base_t0, vinj_t6, ec_b-1, ec_b-2, ec_mask}; "
        "gate is render-judged.",
    )
    p.add_argument(
        "--mask_box",
        default="0.30,0.08,0.70,0.38",
        help="Fallback face box as fractional 'x0,y0,x1,y1' for stems not in "
        "FACE_BOXES (phase 1a only).",
    )
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--timeout", type=int, default=2400, help="Per-arm subprocess timeout (s)"
    )
    p.add_argument("--smoke", action="store_true", help="1 image, 4 arms")
    p.add_argument(
        "--only_arms",
        default=None,
        help="Comma-separated arm-name filter (after phase arm-set build) — "
        "for topping up a prior run with new arms without re-rendering "
        "everything. Cross-run comparisons are valid: same seed + config.",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if not Path(args.ec_weight).is_file():
        raise SystemExit(f"EC checkpoint not found: {args.ec_weight}")

    ck = default_checkpoints()
    ec_scales = [float(s) for s in args.ec_scales.split(",") if s.strip()]
    b_offsets = [float(s) for s in args.ec_b_offsets.split(",") if s.strip()]
    arms = build_arms(ec_scales, b_offsets, smoke=args.smoke, phase=args.phase)
    if args.only_arms:
        wanted = {s.strip() for s in args.only_arms.split(",") if s.strip()}
        unknown = wanted - {a.name for a in arms}
        if unknown:
            raise SystemExit(f"--only_arms names unknown arms: {sorted(unknown)}")
        arms = [a for a in arms if a.name in wanted]
    default_box = tuple(float(s) for s in args.mask_box.split(","))
    assert len(default_box) == 4, "--mask_box wants 'x0,y0,x1,y1'"
    n_images = 1 if args.smoke else args.n_images
    if args.images:
        images = [Path(i).resolve() for i in args.images][:n_images]
    else:
        images = pick_images(n_images, args.seed)

    run_dir = make_run_dir(
        "directedit_ec",
        root=Path(__file__).resolve().parent / "results",
        label=args.label
        or (
            "smoke"
            if args.smoke
            else f"phase{args.phase}"
            if args.phase != "0"
            else "phase0"
        ),
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.info("Phase 0 run dir: %s", run_dir)
    logger.info("Images: %s", [i.name for i in images])
    logger.info("Arms: %s", [a.name for a in arms])

    # One job = one (image, edit) row of arms. Phases 0/1a: a single appended-
    # tag edit per image (mask box = face). Phase 1b: EDITS_1B rows per image,
    # each with its own ψ_tar and hole box.
    jobs: list[tuple[Path, str, str, str, tuple[float, float, float, float]]] = []
    for image in images:
        stem = image.stem
        cap = image.with_suffix(".txt").read_text(encoding="utf-8").strip()
        if args.phase == "1b":
            specs = EDITS_1B.get(stem)
            if specs is None:
                raise SystemExit(f"phase 1b: no EDITS_1B specs for stem {stem!r}")
            for edit_key, spec in specs.items():
                tar = apply_edit(cap, spec["drop"], spec["add"])
                jobs.append((image, cap, edit_key, tar, spec["box"]))
        else:
            tar = f"{cap.rstrip().rstrip(',')}, {args.edit}"
            box = FACE_BOXES.get(stem, default_box)
            jobs.append((image, cap, args.edit, tar, box))

    per_image: list[dict] = []
    grid_rows = []
    for image, cap, edit_key, tar, box in jobs:
        stem = image.stem
        row_key = stem if args.phase != "1b" else f"{stem}_{edit_key}"
        rec: dict = {
            "stem": stem,
            "image": str(image),
            "edit": edit_key,
            "prompt_tar": tar,
            "arms": {},
        }
        cells: list[tuple[str, Optional[Path]]] = [("source", image)]
        mask_png: Optional[Path] = None
        if any(a.mask or a.anchor_mask for a in arms):
            masks_dir = run_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            mask_png = write_face_mask(image, box, masks_dir / f"{row_key}.png")
            preview = masked_cond_preview(
                image, mask_png, masks_dir / f"{row_key}_cond.png"
            )
            rec["mask"] = str(mask_png.relative_to(run_dir))
            cells.append(("cond_masked", preview))
        for arm in arms:
            out_dir = run_dir / "renders" / row_key / arm.name
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = arm_argv(
                arm,
                image,
                cap,
                tar,
                out_dir,
                args,
                ck,
                mask_png=mask_png if (arm.mask or arm.anchor_mask) else None,
            )
            log_path = logs_dir / f"{row_key}_{arm.name}.log"
            logger.info("[%s/%s] running (log: %s)", row_key, arm.name, log_path.name)
            t0 = time.time()
            try:
                with log_path.open("w") as lf:
                    lf.write(" ".join(argv) + "\n\n")
                    lf.flush()
                    proc = subprocess.run(
                        argv,
                        cwd=REPO_ROOT,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
                logger.error(
                    "[%s/%s] TIMEOUT after %ds", row_key, arm.name, args.timeout
                )
            wall = time.time() - t0
            pngs = sorted(out_dir.glob("*.png"))
            out_png = pngs[-1] if (ok and pngs) else None
            # Region split is computed for EVERY arm (not just ec_mask) so the
            # gate can reference recon_base's outside-hole level.
            mses = mse_vs_source(out_png, image, mask_png) if out_png else None
            entry = {
                "ok": bool(out_png is not None),
                "wall_s": round(wall, 1),
                "argv": argv[1:],  # drop interpreter path
                "png": str(out_png.relative_to(run_dir)) if out_png else None,
                "mse_vs_source": mses["full"] if mses else None,
                "mse_outside": mses["outside"] if mses else None,
                "mse_inside": mses["inside"] if mses else None,
            }
            rec["arms"][arm.name] = entry
            cells.append((arm.name, out_png))
            logger.info(
                "[%s/%s] %s wall=%.0fs mse=%s out=%s in=%s",
                row_key,
                arm.name,
                "ok" if entry["ok"] else "FAILED",
                wall,
                f"{entry['mse_vs_source']:.5f}"
                if entry["mse_vs_source"] is not None
                else "-",
                f"{entry['mse_outside']:.5f}"
                if entry["mse_outside"] is not None
                else "-",
                f"{entry['mse_inside']:.5f}"
                if entry["mse_inside"] is not None
                else "-",
            )
        # Reconstruction gate: EC must not break the anchor. Only meaningful
        # when the recon arms ran (phases 0/1a — 1b is render-judged).
        has_recon = "recon_base" in rec["arms"]
        if has_recon:
            rb = rec["arms"].get("recon_base", {}).get("mse_vs_source")
            re_ = rec["arms"].get("recon_ec", {}).get("mse_vs_source")
            rec["recon_ratio"] = (re_ / rb) if (rb and re_ is not None) else None
            rec["recon_gate_pass"] = (
                rec["recon_ratio"] is not None and rec["recon_ratio"] <= 2.0
            )
        # Phase 1a gate: outside the hole the masked prior must hold the
        # source at recon level (≤ 2×), with zero b_offset tuning. Whether the
        # edit LANDS inside the hole stays a render call (grid + face crops).
        if mask_png is not None and has_recon:
            rb_out = rec["arms"].get("recon_base", {}).get("mse_outside")
            em_out = rec["arms"].get("ec_mask", {}).get("mse_outside")
            rec["mask_outside_ratio"] = (
                (em_out / rb_out) if (rb_out and em_out is not None) else None
            )
            rec["mask_gate_pass"] = (
                rec["mask_outside_ratio"] is not None
                and rec["mask_outside_ratio"] <= 2.0
            )
        per_image.append(rec)
        grid_rows.append((row_key, cells))

    grid_path = run_dir / "grid.png"
    make_grid(grid_rows, grid_path)

    recon_recs = [r for r in per_image if "recon_gate_pass" in r]
    gate_pass = all(r["recon_gate_pass"] for r in recon_recs) if recon_recs else None
    metrics = {
        "edit": args.edit,
        "ec_weight": args.ec_weight,
        "phase": args.phase,
        "arms": [a.name for a in arms],
        "per_image": per_image,
        "recon_gate_pass_all": gate_pass,
    }
    if args.phase == "1a":
        metrics["mask_gate_pass_all"] = all(
            r.get("mask_gate_pass", False) for r in per_image
        )
    write_result(
        run_dir,
        script=str(Path(__file__).relative_to(REPO_ROOT)),
        args=args,
        metrics=metrics,
        artifacts=["grid.png"],
    )
    logger.info("=" * 60)
    for r in per_image:
        line = f"{r['stem']}/{r['edit']}:"
        if "recon_gate_pass" in r:
            line += " recon_ratio=%s gate=%s" % (
                f"{r['recon_ratio']:.2f}" if r["recon_ratio"] is not None else "-",
                "PASS" if r["recon_gate_pass"] else "FAIL",
            )
        if "mask_outside_ratio" in r:
            line += "  mask_outside_ratio=%s mask_gate=%s" % (
                f"{r['mask_outside_ratio']:.2f}"
                if r["mask_outside_ratio"] is not None
                else "-",
                "PASS" if r.get("mask_gate_pass") else "FAIL",
            )
        logger.info(line)
    if gate_pass is not None:
        logger.info("Recon gate (all images): %s", "PASS" if gate_pass else "FAIL")
    if args.phase == "1a":
        logger.info(
            "Mask gate (all images): %s",
            "PASS" if metrics["mask_gate_pass_all"] else "FAIL",
        )
    logger.info("Human verdict artifact: %s", grid_path)
    out = {"run_dir": str(run_dir), "recon_gate_pass_all": gate_pass}
    if args.phase == "1a":
        out["mask_gate_pass_all"] = metrics["mask_gate_pass_all"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
