"""``make sr-*`` — ResShift super-resolution sidecar dispatch.

The SR sidecar lives OUTSIDE Anima's uv project: it has its own isolated venv
(``sr/.venv`` — same Blackwell torch family as root, **no xformers**). The split is
NOT a torch-incompatibility (that rationale is stale); it's to keep the SR deps out of
the main Anima lockfile. ResShift's source is vendored under ``sr/resshift/`` (no
external clone). These targets are thin shells that invoke that venv's python.
See ``sr/README.md`` for the env split.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SR = ROOT / "sr"
VENV_PY = SR / ".venv" / "bin" / "python"


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


def cmd_sr_build_hr_pool(extra):
    """Filter gelcrawl/retrieved + image_dataset into a sharp HR pool (anima env OK)."""
    # PIL/numpy only — run under whatever python invoked us (no SR venv needed).
    _run([sys.executable, str(SR / "scripts" / "build_hr_pool.py"), *extra])


def cmd_sr_train(extra):
    """ResShift ×2 finetune: warm-start the released x4 v2, train on our HR pool.

    Defaults --src to sr/data/hr_pool when it exists (build it with `make sr-build-hr-pool`);
    otherwise train.py falls back to image_dataset. Pass ARGS=\"--iters … --bs … --amp\".
    """
    _run([_venv_py(), str(SR / "train_x2" / "train.py"), *extra])


def cmd_sr_rsd_train(extra):
    """RSD distillation: distill the v2 15-step teacher -> 1-step student on our art.

    Defaults --src to the 4096-capped HR cache (sr/data/rsd_hr_cap4096) when it exists
    and the caller didn't pass their own --src — that cache carries native ~4096-scale
    detail at bounded decode cost (NOT the downsized rsd_hr_1024). Falls back to
    train.py's own image_dataset default if the cache is absent.

    VERSION=x2 distills our sr-train x2 finetune (sr/weights/resshift_x2_final.pth) instead
    of the released x4 teacher; output lands in output/sr/rsd_x2. VERSION=x4 is the default.
    """
    argv = list(extra)
    version = os.environ.get("VERSION", "")
    if version and not any(a == "--version" or a.startswith("--version=") for a in argv):
        argv = ["--version", version, *argv]
    if not any(a == "--src" or a.startswith("--src=") for a in argv):
        cache = SR / "data" / "rsd_hr_cap4096"
        if cache.is_dir():
            print(f"[sr-rsd-train] defaulting --src to {cache} (pass --src to override)")
            argv = ["--src", str(cache), *argv]
        else:
            print(f"[sr-rsd-train] {cache} absent — train.py will fall back to image_dataset")
    _run([_venv_py(), str(SR / "distill_rsd" / "train.py"), *argv])


def cmd_sr_rsd_dryrun(extra):
    """RSD VRAM feasibility dry-run (build all nets + 1 fake/gen step)."""
    _run([_venv_py(), str(SR / "distill_rsd" / "dry_run.py"), *extra])


def cmd_sr_rsd_infer(extra):
    """Single-step RSD student inference + MUSIQ. CKPT=… picks a ckpt; unset = most recent.

    VERSION=x2 loads the x2 config (sf=2) and defaults ckpt_dir/out_dir to output/sr/rsd_x2.
    """
    ckpt = os.environ.get("CKPT", "")
    version = os.environ.get("VERSION", "")
    argv = (["--version", version] if version else []) \
        + (["--ckpt", ckpt] if ckpt else []) + list(extra)
    _run([_venv_py(), str(SR / "distill_rsd" / "infer.py"), *argv])


def cmd_sr_test(extra):
    """Tiled SR (released x4 or local x2) on a folder/image: make sr-test IN=<path>
    [OUT=… VERSION=v3|x2 CHOP=512 CKPT=…].

    Thin pass-through to the vendored, basicsr-free sr/scripts/sr_infer.py. Output is
    rsd-infer-style: per-image PNGs + infer_summary.json (MUSIQ) + contact_sheet.png.
    OUT unset -> script default (output/sr/<version>/infer for x2, sr/data/results for
    released x4). ARGS="--no_musiq --no_sheet --sheet_max N" to trim the extras.
    """
    in_path = os.environ.get("IN", "")
    if not in_path:
        sys.exit("set IN=<input image or dir>  (e.g. make sr-test IN=foo.png)")
    version = os.environ.get("VERSION", "v3")
    chop = os.environ.get("CHOP", "512")
    cmd = [
        _venv_py(), str(SR / "scripts" / "sr_infer.py"),
        "-i", in_path, "--version", version, "--chop_size", chop,
    ]
    out = os.environ.get("OUT", "")
    if out:
        cmd += ["-o", out]
    ckpt = os.environ.get("CKPT", "")  # x2 only: pick a make-sr-train checkpoint
    if ckpt:
        cmd += ["--ckpt", ckpt]
    cmd += list(extra)
    _run(cmd)
