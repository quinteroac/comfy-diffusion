"""Tests for US-001 ComfyUI runtime auto-download behavior."""

from __future__ import annotations

import importlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pytest

from comfy_diffusion import _runtime
from comfy_diffusion.runtime import check_runtime


def _build_minimal_comfyui_zip() -> bytes:
    archive_buffer = io.BytesIO()
    archive_root = f"ComfyUI-{_runtime.COMFYUI_PINNED_REF}"

    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr(f"{archive_root}/comfy/__init__.py", "")
        archive.writestr(
            f"{archive_root}/comfy/model_management.py",
            (
                "def get_torch_device():\n"
                "    return 'cpu'\n\n"
                "def get_total_memory(_device):\n"
                "    return 0\n\n"
                "def get_free_memory(_device):\n"
                "    return 0\n"
            ),
        )
        archive.writestr(
            f"{archive_root}/comfyui_version.py",
            "__version__ = '0.16.3'\n",
        )

    return archive_buffer.getvalue()


def test_check_runtime_bootstraps_from_absent_vendor_comfyui_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comfyui_root = tmp_path / "vendor" / "ComfyUI"
    zip_fixture = _build_minimal_comfyui_zip()
    download_calls = {"count": 0}
    original_sys_path = list(sys.path)

    def fake_urlretrieve(url: str, filename: str | Path, *_args: Any, **_kwargs: Any) -> None:
        download_calls["count"] += 1
        assert url == _runtime.COMFYUI_PINNED_ARCHIVE_URL
        Path(filename).write_bytes(zip_fixture)

    monkeypatch.setattr(_runtime, "_comfyui_root", lambda: comfyui_root)
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    for module_name in ("comfyui_version", "comfy.model_management", "comfy"):
        sys.modules.pop(module_name, None)

    payload = check_runtime()

    for module_name in ("comfyui_version", "comfy.model_management", "comfy"):
        sys.modules.pop(module_name, None)
    sys.path[:] = original_sys_path

    assert download_calls["count"] == 1
    assert (comfyui_root / "comfy").is_dir()
    assert "error" not in payload
    assert payload["comfyui_version"] == "0.16.3"
    assert payload["device"] == "cpu"
    assert payload["vram_total_mb"] == 0
    assert payload["vram_free_mb"] == 0
    assert isinstance(payload["python_version"], str)


def test_ensure_comfyui_available_skips_download_when_runtime_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comfyui_root = tmp_path / "vendor" / "ComfyUI"
    (comfyui_root / "comfy").mkdir(parents=True)
    (comfyui_root / "comfy" / "__init__.py").write_text("", encoding="utf-8")

    download_called = {"value": False}

    def fake_download(target: Path) -> None:
        download_called["value"] = True
        raise AssertionError(f"unexpected download for {target}")

    monkeypatch.setattr(_runtime, "_comfyui_root", lambda: comfyui_root)
    monkeypatch.setattr(_runtime, "_download_and_extract_pinned_comfyui", fake_download)

    result = _runtime.ensure_comfyui_available()

    assert result == comfyui_root
    assert download_called["value"] is False


def test_ensure_comfyui_available_redownloads_managed_runtime_without_pin_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comfyui_root = tmp_path / "comfy_diffusion" / "vendor" / "ComfyUI"
    (comfyui_root / "comfy").mkdir(parents=True)
    (comfyui_root / "comfy" / "__init__.py").write_text("", encoding="utf-8")
    downloads: list[Path] = []

    def fake_download(target: Path) -> None:
        downloads.append(target)
        (target / "comfy").mkdir(parents=True, exist_ok=True)
        (target / "comfy" / "__init__.py").write_text("", encoding="utf-8")
        (target / _runtime.COMFYUI_PINNED_REF_MARKER).write_text(
            f"{_runtime.COMFYUI_PINNED_REF}\n", encoding="utf-8"
        )

    monkeypatch.setattr(_runtime, "_comfyui_root", lambda: comfyui_root)
    monkeypatch.setattr(_runtime, "_is_download_managed_runtime", lambda _root: True)
    monkeypatch.setattr(_runtime, "_download_and_extract_pinned_comfyui", fake_download)

    result = _runtime.ensure_comfyui_available()

    assert result == comfyui_root
    assert downloads == [comfyui_root]


def test_ensure_comfyui_available_skips_managed_runtime_with_matching_pin_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comfyui_root = tmp_path / "comfy_diffusion" / "vendor" / "ComfyUI"
    (comfyui_root / "comfy").mkdir(parents=True)
    (comfyui_root / "comfy" / "__init__.py").write_text("", encoding="utf-8")
    (comfyui_root / _runtime.COMFYUI_PINNED_REF_MARKER).write_text(
        f"{_runtime.COMFYUI_PINNED_REF}\n", encoding="utf-8"
    )

    def fake_download(target: Path) -> None:
        raise AssertionError(f"unexpected download for {target}")

    monkeypatch.setattr(_runtime, "_comfyui_root", lambda: comfyui_root)
    monkeypatch.setattr(_runtime, "_is_download_managed_runtime", lambda _root: True)
    monkeypatch.setattr(_runtime, "_download_and_extract_pinned_comfyui", fake_download)

    result = _runtime.ensure_comfyui_available()

    assert result == comfyui_root


def test_ensure_comfyui_available_downloads_and_extracts_when_runtime_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    comfyui_root = tmp_path / "vendor" / "ComfyUI"

    observed_url: dict[str, str] = {}
    extracted_into: dict[str, Path] = {}

    def fake_urlretrieve(url: str, filename: str | Path, *_args: Any, **_kwargs: Any) -> None:
        observed_url["value"] = url
        Path(filename).write_bytes(b"fake-zip")

    class _FakeZipFile:
        def __init__(self, _archive_path: str | Path) -> None:
            pass

        def __enter__(self) -> _FakeZipFile:
            return self

        def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
            return None

        def extractall(self, target_dir: str | Path) -> None:
            extracted_path = Path(target_dir)
            extracted_into["value"] = extracted_path
            extracted_root = extracted_path / f"ComfyUI-{_runtime.COMFYUI_PINNED_REF}"
            (extracted_root / "comfy").mkdir(parents=True)
            (extracted_root / "comfy" / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(_runtime, "_comfyui_root", lambda: comfyui_root)
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr(zipfile, "ZipFile", _FakeZipFile)

    result = _runtime.ensure_comfyui_available()

    assert result == comfyui_root
    assert comfyui_root.is_dir()
    assert (comfyui_root / "comfy").is_dir()
    assert (comfyui_root / _runtime.COMFYUI_PINNED_REF_MARKER).read_text(
        encoding="utf-8"
    ).strip() == _runtime.COMFYUI_PINNED_REF
    assert observed_url["value"] == _runtime.COMFYUI_PINNED_ARCHIVE_URL
    assert _runtime.COMFYUI_PINNED_REF in observed_url["value"]
    assert "value" in extracted_into


def test_check_runtime_returns_healthy_payload_after_successful_auto_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_called = {"value": False}

    def fake_ensure_available() -> Path:
        ensure_called["value"] = True
        return Path("/tmp/fake-comfy")

    def fake_import_module(module_name: str) -> Any:
        if module_name == "comfyui_version":
            return type("ComfyVersion", (), {"__version__": "0.16.3"})()

        if module_name == "comfy.model_management":
            return type(
                "ModelManagement",
                (),
                {
                    "get_torch_device": staticmethod(lambda: "cpu"),
                    "get_total_memory": staticmethod(lambda _dev: 1),
                    "get_free_memory": staticmethod(lambda _dev: 1),
                },
            )()

        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(_runtime, "ensure_comfyui_available", fake_ensure_available)
    monkeypatch.setattr(_runtime, "ensure_comfyui_on_path", lambda: Path("/tmp/fake-comfy"))
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    payload = check_runtime()

    assert ensure_called["value"] is True
    assert payload == {
        "comfyui_version": "0.16.3",
        "device": "cpu",
        "vram_total_mb": 0,
        "vram_free_mb": 0,
        "python_version": payload["python_version"],
    }


def test_runtime_pinned_ref_constant_matches_git_submodule_pin() -> None:
    gitmodules_text = (Path(__file__).resolve().parents[1] / ".gitmodules").read_text(
        encoding="utf-8"
    )

    assert f"Pinned ComfyUI ref: {_runtime.COMFYUI_PINNED_REF}" in gitmodules_text


@pytest.mark.parametrize(
    ("failure", "expected_phrase"),
    [
        (ConnectionError("network timeout while downloading archive"), "network timeout"),
        (PermissionError("permission denied writing vendor directory"), "permission denied"),
        (OSError("no space left on device"), "no space left on device"),
    ],
)
def test_check_runtime_returns_error_dict_when_autodownload_bootstrap_fails(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_phrase: str,
) -> None:
    def fake_ensure_available() -> Path:
        raise failure

    monkeypatch.setattr(_runtime, "ensure_comfyui_available", fake_ensure_available)

    payload = check_runtime()

    assert "error" in payload
    assert expected_phrase in payload["error"]
    assert payload["comfyui_version"] is None
    assert payload["device"] is None
    assert payload["vram_total_mb"] is None
    assert payload["vram_free_mb"] is None
    assert isinstance(payload["python_version"], str)
