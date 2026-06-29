#!/usr/bin/env bash
# Reproducible setup for the ResShift SR sidecar venv.
# Isolated from Anima's uv project: modern Blackwell-capable torch, NO xformers
# (ResShift falls back to vanilla/softmax attention). Run from sr/.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> sr/

echo "==> creating sr/.venv (python 3.11)"
uv venv .venv --python 3.11

echo "==> installing torch/torchvision (cu128 index; resolves to a Blackwell sm_120 build)"
uv pip install --python .venv/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> installing ResShift deps (xformers intentionally omitted)"
uv pip install --python .venv/bin/python -r requirements.txt

echo "==> patching basicsr (torchvision.transforms.functional_tensor was removed)"
DEG=$(.venv/bin/python - <<'PY'
import importlib.util as u
print(u.find_spec("basicsr").submodule_search_locations[0] + "/data/degradations.py")
PY
)
sed -i 's/from torchvision.transforms.functional_tensor import/from torchvision.transforms.functional import/' "$DEG"

echo "==> patching ResShift source (no-xformers attention)"
.venv/bin/python scripts/patch_resshift.py

echo "==> verifying"
.venv/bin/python - <<'PY'
import torch, basicsr
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "cap", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("basicsr", basicsr.__version__, "OK")
PY
echo "==> SR venv ready."
