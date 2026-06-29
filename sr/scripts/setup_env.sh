#!/usr/bin/env bash
# Reproducible setup for the ResShift SR sidecar venv.
#
# The venv is isolated from Anima's uv project NOT because of torch (both run the
# same torch 2.12 + cu132 Blackwell build) but to keep the SR deps out of the main
# Anima lockfile. ResShift's own source is VENDORED under sr/resshift/ (committed),
# so there's no external clone to fetch or patch. Python is pinned to 3.13 to match
# the root .venv; xformers is intentionally absent (the vendored VQGAN already ships
# the query-chunked SDPA attention -- a Blackwell xformers build is a non-win, see
# README.md). basicsr is gone too: the vendored tree is basicsr-free.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> sr/

echo "==> creating sr/.venv (python 3.13, matching root .venv)"
uv venv .venv --clear --python 3.13

echo "==> installing torch/torchvision (cu132 index -> Blackwell sm_120 build, matches root)"
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu132

echo "==> installing SR deps (xformers + basicsr intentionally omitted)"
uv pip install --python .venv/bin/python -r requirements.txt

echo "==> verifying (torch + vendored ResShift import)"
.venv/bin/python - <<'PY'
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path("resshift").resolve()))
from models.unet import UNetModelSwin  # noqa: F401 (vendored)
from ldm.models.autoencoder import VQModelTorch  # noqa: F401 (vendored, patched attn)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("vendored ResShift import OK")
PY
echo "==> SR venv ready."
