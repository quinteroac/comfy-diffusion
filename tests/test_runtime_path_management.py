"""Tests for US-004 runtime path bootstrap behavior."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python(code: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_root = str(_repo_root())
    env["PYTHONPATH"] = (
        repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"
    )

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_runtime_bootstrap_makes_comfy_internals_discoverable() -> None:
    result = _run_python(
        (
            "import importlib.util\n"
            "from comfy_diffusion._runtime import ensure_comfyui_on_path\n"
            "ensure_comfyui_on_path()\n"
            "spec = importlib.util.find_spec('comfy.model_management')\n"
            "assert spec is not None\n"
        ),
        cwd=_repo_root(),
    )
    assert result.returncode == 0


def test_path_insertion_is_minimal_and_not_duplicated() -> None:
    result = _run_python(
        (
            "from pathlib import Path\n"
            "import importlib\n"
            "import json\n"
            "import comfy_diffusion\n"
            "from comfy_diffusion._runtime import ensure_comfyui_on_path\n"
            "import sys\n"
            "package_root = Path(comfy_diffusion.__file__).resolve().parent.parent\n"
            "expected = str(package_root / 'vendor' / 'ComfyUI')\n"
            "ensure_comfyui_on_path()\n"
            "importlib.reload(comfy_diffusion)\n"
            "ensure_comfyui_on_path()\n"
            "matches = [p for p in sys.path if p == expected]\n"
            "vendor_entries = [p for p in sys.path if '/vendor/' in p.replace('\\\\\\\\', '/')]\n"
            "print(json.dumps({'expected': expected, 'matches': len(matches),"
            " 'vendor_entries': vendor_entries}))\n"
        ),
        cwd=_repo_root(),
    )

    payload = json.loads(result.stdout)
    assert Path(payload["expected"]).is_absolute()
    assert payload["matches"] == 1
    assert payload["vendor_entries"] == [payload["expected"]]


def test_import_works_from_any_working_directory(tmp_path: Path) -> None:
    result = _run_python(
        (
            "from pathlib import Path\n"
            "import json\n"
            "import comfy_diffusion\n"
            "from comfy_diffusion._runtime import ensure_comfyui_on_path\n"
            "import sys\n"
            "comfyui_path = str("
            "Path(comfy_diffusion.__file__).resolve().parent.parent / 'vendor' / 'ComfyUI')\n"
            "ensure_comfyui_on_path()\n"
            "print(json.dumps({'cwd': str(Path.cwd()), 'comfyui_path': comfyui_path,"
            " 'on_path': comfyui_path in sys.path}))\n"
        ),
        cwd=tmp_path,
    )

    payload = json.loads(result.stdout)
    assert payload["cwd"] == str(tmp_path.resolve())
    assert Path(payload["comfyui_path"]).is_absolute()
    assert payload["on_path"] is True
