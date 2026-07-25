"""Phase 2 — does the subject descriptor retrieve identity at all?

`run_bench.py --phase 2` asks whether the cross-image subject adapter can serve
DirectEdit's geometry edits. That test composes three mechanisms (inversion
anchor, EC cond gate, prompt), so a null result there is ambiguous: it could be
the adapter, or the composition masking it.

This probe removes the ambiguity by dropping DirectEdit entirely. It is plain
generation, and it replays the adapter's own training task:

    cond   = image A of a character   (--easycontrol_image)
    prompt = the caption of image B   (a DIFFERENT image of the SAME character)

If the descriptor learned position-free identity retrieval, the EC arm should
look more like the character than the no-EC arm run at the same seed/prompt.
The no-EC arm is the control that matters — the prompt alone already carries
the character NAME as a tag, so "the render looks like the character" is NOT
evidence of retrieval on its own. Only the EC-vs-noEC delta is.

Arms per pair:
    noec        prompt only, no adapter          control — what the tags alone give
    ec_b<off>   same seed/prompt + cond = A      one arm per --b_offsets entry

Verdict is render-judged off the contact sheet (cond | real target | noec |
ec_*). Pairs are drawn from the miner's own manifest, so they are TRAIN pairs:
this measures an upper bound on retrieval, which is what we want — a train-set
failure is decisive, a train-set success would still need a held-out check.

Usage:
    uv run python project/directedit_ec/bench/run_subject_probe.py --n_pairs 3
    uv run python project/directedit_ec/bench/run_subject_probe.py --b_offsets 0,4
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.env import default_checkpoints  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

PAIRS_JSON = REPO_ROOT / "post_image_dataset" / "easycontrol" / "subject" / "pairs.json"
RESIZED_DIR = REPO_ROOT / "post_image_dataset" / "resized"
DEFAULT_EC_WEIGHT = (
    REPO_ROOT / "output" / "ckpt" / "anima_easycontrol_subject.safetensors"
)


def resolve(rel: str) -> Path:
    """'artist/stem' → the resized PNG (the miner's manifest is dir-relative)."""
    return RESIZED_DIR / f"{rel}.png"


def pick_pairs(n: int, seed: int, cross_artist_only: bool) -> list[dict]:
    manifest = json.loads(PAIRS_JSON.read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    if cross_artist_only:
        pairs = [
            p for p in pairs if p["target"].split("/")[0] != p["cond"].split("/")[0]
        ]
    # Keep only pairs whose PNG + caption both resolve (the manifest indexes the
    # cache, which can outlive a re-preprocess).
    ok = []
    for p in pairs:
        t, c = resolve(p["target"]), resolve(p["cond"])
        if t.is_file() and c.is_file() and t.with_suffix(".txt").is_file():
            ok.append(p)
    if not ok:
        raise SystemExit(f"no resolvable pairs in {PAIRS_JSON}")
    # One pair per character, so n pairs = n distinct identities.
    by_char: dict[str, dict] = {}
    for p in ok:
        by_char.setdefault(p["character"], p)
    chosen = sorted(by_char.values(), key=lambda p: p["target"])
    random.Random(seed).shuffle(chosen)
    return chosen[:n]


def arm_argv(
    name: str,
    prompt: str,
    cond: Optional[Path],
    b_offset: Optional[float],
    out_dir: Path,
    args,
    ck,
) -> list[str]:
    del name
    argv = [
        sys.executable,
        str(REPO_ROOT / "inference.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        "flash",
        "--prompt",
        prompt,
        "--negative_prompt",
        "worst quality, low quality, score_1, score_2, score_3, blurry, "
        "jpeg artifacts, sepia",
        "--image_size",
        str(args.height),
        str(args.width),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift",
        "3.0",
        "--sampler",
        "euler",
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--save_path",
        str(out_dir),
    ]
    if cond is not None:
        argv += [
            "--easycontrol_weight",
            str(args.ec_weight),
            "--easycontrol_image",
            str(cond),
            "--easycontrol_image_match_size",
        ]
        if b_offset:
            argv += ["--easycontrol_b_offset", str(b_offset)]
    return argv


def make_sheet(rows: list[tuple[str, list[tuple[str, Optional[Path]]]]], out: Path):
    from PIL import Image, ImageDraw

    thumb_h, label_h = 384, 22
    cols = max(len(r[1]) for r in rows)
    thumbs = []
    for _, cells in rows:
        row = []
        for label, p in cells:
            if p is not None and Path(p).is_file():
                im = Image.open(p).convert("RGB")
                w = int(im.size[0] * thumb_h / im.size[1])
                row.append((label, im.resize((w, thumb_h), Image.LANCZOS)))
            else:
                row.append((label, None))
        thumbs.append(row)
    cell_w = max(
        (im.size[0] for row in thumbs for _, im in row if im is not None),
        default=thumb_h,
    )
    canvas = Image.new(
        "RGB", (cell_w * cols, (thumb_h + label_h) * len(rows)), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(thumbs):
        for c, (label, im) in enumerate(row):
            x, y = c * cell_w, r * (thumb_h + label_h)
            if im is not None:
                canvas.paste(im, (x + (cell_w - im.size[0]) // 2, y + label_h))
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255))
    canvas.save(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n_pairs", type=int, default=3)
    p.add_argument("--ec_weight", default=str(DEFAULT_EC_WEIGHT))
    p.add_argument(
        "--b_offsets",
        default="0,4",
        help="Comma-separated b_cond offsets; one ec_b<off> arm each.",
    )
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=1216)
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument(
        "--any_artist",
        action="store_true",
        help="Allow same-artist pairs (default: cross-artist only, which "
        "starves the style shortcut).",
    )
    p.add_argument("--label", default="phase2-subject-probe")
    args = p.parse_args()

    if not Path(args.ec_weight).is_file():
        raise SystemExit(f"EC checkpoint not found: {args.ec_weight}")
    ck = default_checkpoints()
    offsets = [float(s) for s in args.b_offsets.split(",") if s.strip()]
    pairs = pick_pairs(args.n_pairs, args.seed, not args.any_artist)

    run_dir = make_run_dir(
        "directedit_ec",
        root=Path(__file__).resolve().parent / "results",
        label=args.label,
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.info("Run dir: %s", run_dir)
    logger.info("Pairs: %s", [p["character"] for p in pairs])

    arms: list[tuple[str, Optional[float]]] = [("noec", None)]
    arms += [(f"ec_b{o:g}", o) for o in offsets]

    per_pair = []
    sheet_rows = []
    for pair in pairs:
        cond = resolve(pair["cond"])
        target = resolve(pair["target"])
        prompt = target.with_suffix(".txt").read_text(encoding="utf-8").strip()
        key = pair["character"].replace(" ", "_").replace("/", "_")
        rec = {
            "character": pair["character"],
            "cond": str(cond),
            "target": str(target),
            "prompt": prompt,
            "arms": {},
        }
        cells: list[tuple[str, Optional[Path]]] = [
            ("cond (A)", cond),
            ("real target (B)", target),
        ]
        for name, off in arms:
            out_dir = run_dir / "renders" / key / name
            out_dir.mkdir(parents=True, exist_ok=True)
            argv = arm_argv(
                name,
                prompt,
                cond if name != "noec" else None,
                off,
                out_dir,
                args,
                ck,
            )
            log_path = logs_dir / f"{key}_{name}.log"
            logger.info("[%s/%s] running", key, name)
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
                logger.error("[%s/%s] TIMEOUT", key, name)
            pngs = sorted(q for q in out_dir.glob("*.png"))
            out_png = pngs[-1] if (ok and pngs) else None
            rec["arms"][name] = {
                "ok": out_png is not None,
                "wall_s": round(time.time() - t0, 1),
                "png": str(out_png.relative_to(run_dir)) if out_png else None,
            }
            cells.append((name, out_png))
            logger.info(
                "[%s/%s] %s wall=%.0fs",
                key,
                name,
                "ok" if out_png else "FAILED",
                time.time() - t0,
            )
        per_pair.append(rec)
        sheet_rows.append((key, cells))

    sheet = run_dir / "grid.png"
    make_sheet(sheet_rows, sheet)
    write_result(
        run_dir,
        script=str(Path(__file__).relative_to(REPO_ROOT)),
        args=args,
        metrics={
            "ec_weight": args.ec_weight,
            "arms": [a for a, _ in arms],
            "per_pair": per_pair,
            "note": "render-judged; train-set pairs = upper bound on retrieval",
        },
        artifacts=["grid.png"],
    )
    logger.info("Human verdict artifact: %s", sheet)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
