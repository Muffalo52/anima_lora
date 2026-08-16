import tomllib
from pathlib import Path
from types import SimpleNamespace

from library.runtime.backend import is_rocm, resolve_attention_mode
from scripts import update

ROOT = Path(__file__).resolve().parents[1]


def _torch(hip):
    return SimpleNamespace(version=SimpleNamespace(hip=hip))


def test_rocm_detection_uses_hip_version():
    assert is_rocm(_torch("7.14.0"))
    assert not is_rocm(_torch(None))


def test_rocm_flash_falls_back_to_torch_sdpa():
    assert resolve_attention_mode("flash", _torch("7.14.0")) == "torch"


def test_rocm_keeps_explicit_non_flash_modes():
    assert resolve_attention_mode("flex", _torch("7.14.0")) == "flex"
    assert resolve_attention_mode("torch", _torch("7.14.0")) == "torch"


def test_cuda_keeps_flash():
    assert resolve_attention_mode("flash", _torch(None)) == "flash"


def test_update_reuses_saved_rocm_backend(monkeypatch, tmp_path):
    (tmp_path / ".anima_backend").write_text("rocm\n", encoding="ascii")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.delenv("ANIMA_BACKEND", raising=False)

    assert update._uv_sync_command() == [
        "uv",
        "sync",
        "--extra",
        "rocm-windows",
    ]


def test_update_environment_override_wins(monkeypatch, tmp_path):
    (tmp_path / ".anima_backend").write_text("rocm\n", encoding="ascii")
    monkeypatch.setattr(update, "ROOT", tmp_path)
    monkeypatch.setattr(update.sys, "platform", "win32")
    monkeypatch.setenv("ANIMA_BACKEND", "cuda")

    assert update._selected_windows_backend() == "cuda"


def test_windows_backend_dependencies_are_isolated():
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    extras = project["project"]["optional-dependencies"]
    cuda = "\n".join(extras["cuda-windows"])
    rocm = "\n".join(extras["rocm-windows"])

    assert "+cu132" in cuda
    assert "flash_attn" in cuda
    assert "+rocm7.14.0" in rocm
    assert "device-gfx1200" in rocm and "device-gfx1201" in rocm
    assert "flash" not in rocm
