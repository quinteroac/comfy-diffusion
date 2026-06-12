"""Internal runtime bootstrap for comfy_diffusion.

Path insertion is intentionally lightweight and import-safe: this module must not
import torch or comfy internals just to make ComfyUI discoverable.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

COMFYUI_PINNED_REF = "822aca19836cd75c815631db23c3ad742d1f7d5e"
COMFYUI_PINNED_ARCHIVE_URL = (
    "https://github.com/Comfy-Org/ComfyUI/archive/"
    f"{COMFYUI_PINNED_REF}.zip"
)
COMFYUI_PINNED_REF_MARKER = ".comfy-diffusion-comfyui-ref"


def _comfyui_root() -> Path:
    """Return the absolute path to the vendored ComfyUI directory."""
    package_dir = Path(__file__).resolve().parent

    # Preferred layout: repo_root/vendor/ComfyUI (vendored git submodule).
    repo_vendor = package_dir.parent / "vendor" / "ComfyUI"
    if repo_vendor.exists():
        return repo_vendor

    # Back-compat layout (older iterations): comfy_diffusion/vendor/ComfyUI.
    package_vendor = package_dir / "vendor" / "ComfyUI"
    return package_vendor


def _has_comfyui_runtime(comfyui_root: Path) -> bool:
    """Return True if the ComfyUI runtime directory looks initialized."""
    return comfyui_root.is_dir() and (comfyui_root / "comfy").is_dir()


def _is_download_managed_runtime(comfyui_root: Path) -> bool:
    """Return True when this runtime lives inside the installed package tree."""
    package_dir = Path(__file__).resolve().parent
    return comfyui_root == package_dir / "vendor" / "ComfyUI"


def _has_pinned_comfyui_runtime(comfyui_root: Path) -> bool:
    """Return True if a downloaded ComfyUI runtime matches the pinned ref."""
    if not _has_comfyui_runtime(comfyui_root):
        return False
    if not _is_download_managed_runtime(comfyui_root):
        return True

    marker = comfyui_root / COMFYUI_PINNED_REF_MARKER
    try:
        return marker.read_text(encoding="utf-8").strip() == COMFYUI_PINNED_REF
    except OSError:
        return False


def _download_and_extract_pinned_comfyui(comfyui_root: Path) -> None:
    """Download and extract the pinned ComfyUI ref into vendor/ComfyUI."""
    vendor_dir = comfyui_root.parent
    vendor_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="comfyui-download-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        archive_path = tmp_dir / "comfyui.zip"

        urllib.request.urlretrieve(COMFYUI_PINNED_ARCHIVE_URL, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(tmp_dir)

        extracted_candidates = list(tmp_dir.glob("ComfyUI-*"))
        if not extracted_candidates:
            raise RuntimeError("Downloaded ComfyUI archive had unexpected structure.")

        extracted_root = extracted_candidates[0]
        if comfyui_root.exists():
            shutil.rmtree(comfyui_root)
        shutil.move(str(extracted_root), str(comfyui_root))

    if not _has_comfyui_runtime(comfyui_root):
        raise RuntimeError("ComfyUI runtime download completed but content is invalid.")
    (comfyui_root / COMFYUI_PINNED_REF_MARKER).write_text(
        f"{COMFYUI_PINNED_REF}\n", encoding="utf-8"
    )


def ensure_comfyui_available() -> Path:
    """Ensure vendored ComfyUI exists; download pinned ref if missing."""
    comfyui_root = _comfyui_root()

    if not _has_pinned_comfyui_runtime(comfyui_root):
        _download_and_extract_pinned_comfyui(comfyui_root)

    return comfyui_root


def _torch_has_accelerator() -> bool:
    """Return whether PyTorch can use a non-CPU device in this process."""
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False

    checks = [
        lambda: bool(getattr(torch.cuda, "is_available", lambda: False)()),
        lambda: bool(
            getattr(getattr(torch, "xpu", None), "is_available", lambda: False)()
        ),
        lambda: bool(
            getattr(
                getattr(getattr(torch, "backends", None), "mps", None),
                "is_available",
                lambda: False,
            )()
        ),
        lambda: bool(getattr(getattr(torch, "npu", None), "is_available", lambda: False)()),
        lambda: bool(getattr(getattr(torch, "mlu", None), "is_available", lambda: False)()),
    ]
    for check in checks:
        try:
            if check():
                return True
        except Exception:
            continue
    return False


def ensure_comfyui_on_path() -> Path:
    """Ensure vendored ComfyUI is available and importable; return the inserted path.

    Respects the ``COMFY_VRAM_MODE`` environment variable to configure VRAM
    management before ``comfy.model_management`` is first imported.  Accepted
    values (case-insensitive): ``low``, ``no``, ``high``, ``normal``.
    Must be set before any ``comfy.*`` import occurs in the process.
    """
    import os

    comfyui_root = ensure_comfyui_available()
    comfyui_root_str = str(comfyui_root)

    if comfyui_root_str not in sys.path:
        sys.path.insert(0, comfyui_root_str)

    # Apply VRAM mode env-var override before model_management is imported.
    vram_mode = os.environ.get("COMFY_VRAM_MODE", "").strip().lower()
    reserve_vram = os.environ.get("COMFY_RESERVE_VRAM", "").strip()
    should_force_cpu = not _torch_has_accelerator()
    if (
        vram_mode or reserve_vram or should_force_cpu
    ) and "comfy.model_management" not in sys.modules:
        try:
            import comfy.cli_args as _cli_args  # noqa: PLC0415
            _args = _cli_args.args
            if should_force_cpu and hasattr(_args, "cpu"):
                _args.cpu = True
            if vram_mode:
                # Reset all vram flags first.
                for _flag in ("lowvram", "novram", "highvram", "gpu_only"):
                    if hasattr(_args, _flag):
                        setattr(_args, _flag, False)
                if vram_mode == "low":
                    _args.lowvram = True
                elif vram_mode == "no":
                    _args.novram = True
                elif vram_mode == "high":
                    _args.highvram = True
                # "normal" — already reset above
            if reserve_vram:
                _args.reserve_vram = float(reserve_vram)
        except Exception:
            pass  # Best-effort; don't crash if cli_args isn't importable yet.

    return comfyui_root
