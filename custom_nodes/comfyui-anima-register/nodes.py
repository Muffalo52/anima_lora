"""Anima register-token adapter — ComfyUI inference node (consolidated).

One node: takes a stock Anima ``MODEL`` + a register ``adapter.safetensors``
(trained by ``train.py --method register`` / ``networks/methods/register.py``)
and returns a patched ``MODEL`` a normal KSampler runs with the registers live.

The adapter (see ``register_apply.py``) is comfy-native — it patches the split
``q_proj``/``k_proj``/``v_proj`` of ``MiniTrainDIT`` (no fused ``qkv_proj``) and
reimplements the forward to inject/strip register tokens. It is (re)built and
applied at the first sampling step against the resident ``diffusion_model``, so
it survives ComfyUI's clone/reload/recompile.

Constraints:
* Do NOT stack with the Block Compile node — the register mechanism runs eager.
* Loads the canonical split ``.safetensors`` (config in the header metadata).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .register_apply import RegisterComfyAdapter

log = logging.getLogger("comfyui-anima-register")

HERE = Path(__file__).resolve().parent
# In-tree install: .../anima_lora/custom_nodes/comfyui-anima-register -> repo root.
REPO_ROOT = HERE.parents[1]
# Dirs scanned for the dropdown (kept tight so we only stat a handful of headers).
_SCAN_ROOTS = ("output/ckpt", "output/temp", "bench/headroom/results")
_NO_ADAPTERS = "<no register adapters found — use path_override>"


def _is_register_safetensors(path: Path) -> bool:
    """Cheap header-only check that a .safetensors is a register adapter."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        return meta.get("ss_network_spec") == "register" or "ss_num_registers" in meta
    except Exception:
        return False


def _list_register_adapters() -> list[str]:
    """Repo-relative paths of every register adapter under the scan roots."""
    found = []
    for rel in _SCAN_ROOTS:
        root = REPO_ROOT / rel
        if not root.is_dir():
            continue
        for p in root.rglob("*.safetensors"):
            if _is_register_safetensors(p):
                found.append(p.relative_to(REPO_ROOT).as_posix())
    return sorted(set(found))


def _load_adapter(path: str, strength: float) -> RegisterComfyAdapter:
    """Load a register ``.safetensors`` into a ready-to-apply adapter.

    The checkpoint carries the split QKV surface + ``ss_*`` config in the header
    metadata (written by ``AdapterNetworkBase.save_weights`` / register's
    ``metadata_fields``).
    """
    from safetensors import safe_open

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.is_file():
        raise FileNotFoundError(f"register adapter not found: {p}")

    state_dict, meta = {}, {}
    with safe_open(str(p), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    if "ss_num_registers" not in meta:
        raise ValueError(
            f"{p} is not a register adapter (no ss_num_registers metadata)"
        )

    config = {
        "num_registers": int(meta["ss_num_registers"]),
        "arm": meta.get("ss_arm", "B"),
        "qkv_mode": meta.get("ss_qkv_mode", "unfrozen"),
        "target_blocks": json.loads(meta["ss_target_blocks"]),
        "scale": float(meta.get("ss_scale", 1.0)),
        # pre-insert_block checkpoints trained with entry insertion — default 0
        "insert_block": int(meta.get("ss_insert_block", 0)),
    }
    log.info("loaded register adapter %s (%s)", p.name, config)
    return RegisterComfyAdapter(state_dict, config, strength=strength)


class AnimaRegisterAdapter:
    """Load + apply a register adapter onto a MODEL in one node."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _list_register_adapters() or [_NO_ADAPTERS]
        return {
            "required": {
                "model": ("MODEL",),
                "adapter_name": (choices,),
            },
            "optional": {
                # Non-empty wins over the dropdown — for a path outside the roots.
                "path_override": ("STRING", {"default": "", "multiline": False}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "anima"

    def apply(
        self, model, adapter_name: str, path_override: str = "", strength: float = 1.0
    ):
        sel = (path_override or "").strip() or adapter_name
        if sel == _NO_ADAPTERS:
            raise ValueError(
                f"no register adapter selected; scanned {_SCAN_ROOTS} under "
                f"{REPO_ROOT} — set path_override to an adapter.safetensors"
            )

        m = model.clone()

        def unet_wrapper(apply_model, args):
            dm = m.model.diffusion_model
            adapter = getattr(dm, "_anima_register_adapter", None)
            # (Re)build against the *current* resident diffusion_model if the hook
            # was stranded (clone/reload/recompile swapped the module instance).
            if adapter is None or adapter.dm is not dm:
                adapter = _load_adapter(sel, strength).apply(dm)
                dm._anima_register_adapter = adapter
            return apply_model(args["input"], args["timestep"], **args["c"])

        m.set_model_unet_function_wrapper(unet_wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaRegisterAdapter": AnimaRegisterAdapter}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaRegisterAdapter": "Anima Register Adapter"}
