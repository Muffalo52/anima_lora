"""``make sr-*`` — ResShift super-resolution sidecar dispatch.

The SR sidecar lives OUTSIDE Anima's uv project: it has its own isolated venv
(``sr/.venv`` — modern Blackwell-capable torch, **no xformers**) because ResShift
pins an incompatible torch. These targets are thin shells that invoke that venv's
python; they never import Anima's torch. See ``sr/README.md`` for the env split.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SR = ROOT / "sr"
VENV_PY = SR / ".venv" / "bin" / "python"
RESSHIFT = ROOT / "ResShift"


def _venv_py() -> str:
    if not VENV_PY.exists():
        sys.exit(
            f"SR venv missing ({VENV_PY}).\n"
            "Create it first:  make sr-setup"
        )
    return str(VENV_PY)


def _run(cmd, cwd=ROOT):
    print("RUN:", " ".join(str(c) for c in cmd), f"(cwd={cwd})")
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode:
        sys.exit(r.returncode)


def cmd_sr_setup(extra):
    """Create sr/.venv, install Blackwell torch + ResShift deps, patch basicsr."""
    _run([str(SR / "scripts" / "setup_env.sh")], cwd=SR)


def cmd_sr_prep(extra):
    """Build the frozen synthetic-LR eval set from image_dataset/ (anima env OK)."""
    # build_eval_set only needs PIL/numpy — run under whatever python invoked us.
    _run([sys.executable, str(SR / "scripts" / "build_eval_set.py"), *extra])


def cmd_sr_phase0(extra):
    """Phase-0 sanity: released ResShift x4 on the eval set + metrics + montage."""
    _run([_venv_py(), str(SR / "scripts" / "run_phase0.py"), *extra])


def cmd_sr_rsd_train(extra):
    """RSD distillation: distill the v2 15-step teacher -> 1-step student on our art."""
    _run([_venv_py(), str(SR / "distill_rsd" / "train.py"), *extra])


def cmd_sr_rsd_dryrun(extra):
    """RSD VRAM feasibility dry-run (build all nets + 1 fake/gen step)."""
    _run([_venv_py(), str(SR / "distill_rsd" / "dry_run.py"), *extra])


def cmd_sr_rsd_infer(extra):
    """Single-step RSD student inference + MUSIQ: make sr-rsd-infer CKPT=… (passes --ckpt)."""
    ckpt = os.environ.get("CKPT", "")
    argv = (["--ckpt", ckpt] if ckpt else []) + list(extra)
    _run([_venv_py(), str(SR / "distill_rsd" / "infer.py"), *argv])


def cmd_sr_test(extra):
    """Tiled SR on a folder/image: make sr-test IN=<path> [OUT=… SCALE=4 VERSION=v3].

    Thin pass-through to ResShift's inference_resshift.py inside the SR venv.
    """
    in_path = os.environ.get("IN", "")
    if not in_path:
        sys.exit("set IN=<input image or dir>  (e.g. make sr-test IN=foo.png)")
    out = os.environ.get("OUT", str(SR / "data" / "results"))
    scale = os.environ.get("SCALE", "4")
    version = os.environ.get("VERSION", "v3")
    chop = os.environ.get("CHOP", "512")
    cmd = [
        _venv_py(), "inference_resshift.py",
        "-i", in_path, "-o", out,
        "--task", "realsr", "--scale", scale,
        "--version", version, "--chop_size", chop, *extra,
    ]
    _run(cmd, cwd=RESSHIFT)
