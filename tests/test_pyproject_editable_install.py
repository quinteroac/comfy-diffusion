"""Tests for US-001 base installation and package metadata behavior."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_pyproject() -> dict[str, Any]:
    pyproject_path = _repo_root() / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _run_command(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd or _repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )


def _venv_python_path(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _dependency_name(spec: str) -> str:
    base = spec.split(";", maxsplit=1)[0].strip()
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        base = base.split(separator, maxsplit=1)[0].strip()
    return base.split("[", maxsplit=1)[0].strip().lower().replace("_", "-")


def test_pyproject_declares_required_project_metadata() -> None:
    pyproject = _read_pyproject()
    project = pyproject["project"]

    assert project["name"] == "comfy-diffusion"
    assert project["version"]
    assert project["description"]
    assert project["requires-python"] == ">=3.12"
    assert project["license"]["text"] == "GPL-3.0-only"
    assert project["authors"]
    assert project["authors"][0]["name"]
    assert project["urls"]["Homepage"]
    assert project["urls"]["Repository"]
    assert project["urls"]["Issues"]


def test_core_dependencies_are_declared_and_torch_is_optional_only() -> None:
    pyproject = _read_pyproject()
    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]

    assert dependencies
    assert "torch" not in {_dependency_name(dependency) for dependency in dependencies}
    assert optional["cpu"] == ["torch", "torchvision"]
    assert optional["cuda"] == ["torch", "torchvision"]


def test_torch_optional_dependencies_are_not_version_pinned() -> None:
    pyproject = _read_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    for extra in ("cpu", "cuda"):
        for dependency in optional[extra]:
            assert dependency in {"torch", "torchvision"}
            assert "==" not in dependency
            assert ">=" not in dependency
            assert "<=" not in dependency


def test_uv_sources_route_torch_to_cpu_and_cuda_indexes_by_extra() -> None:
    pyproject = _read_pyproject()
    sources = pyproject["tool"]["uv"]["sources"]

    torch_sources = sources["torch"]
    markers = {source["index"]: source["marker"] for source in torch_sources}

    assert markers["pytorch-cpu"] == "extra == 'cpu'"
    assert markers["pytorch-cuda"] == "extra != 'cpu' and extra == 'cuda'"


def test_video_optional_dependencies_are_declared_in_video_extra() -> None:
    pyproject = _read_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    assert "video" in optional
    assert optional["video"] == [
        "av>=16.0.0",
        "imageio>=2.37.2",
        "opencv-python>=4.13.0.92",
    ]


def test_video_dependencies_are_not_in_core_dependencies() -> None:
    pyproject = _read_pyproject()
    dependencies = pyproject["project"]["dependencies"]

    assert all(not dependency.startswith("opencv-python") for dependency in dependencies)
    assert all(not dependency.startswith("imageio") for dependency in dependencies)


def test_audio_optional_dependency_is_declared_in_audio_extra() -> None:
    pyproject = _read_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    assert "audio" in optional
    assert optional["audio"] == ["torchaudio"]


def test_all_extra_is_union_of_cpu_cuda_video_and_audio_dependencies() -> None:
    pyproject = _read_pyproject()
    optional = pyproject["project"]["optional-dependencies"]

    expected_union = set()
    for extra in ("cuda", "video", "audio", "comfyui"):
        expected_union.update(optional[extra])

    assert "all" in optional
    assert set(optional["all"]) == expected_union


def test_comfyui_is_not_declared_as_pip_dependency() -> None:
    pyproject = _read_pyproject()
    project = pyproject["project"]
    dependencies = project["dependencies"]
    optional = project["optional-dependencies"]

    package_names = {_dependency_name(dep) for dep in dependencies}
    for deps in optional.values():
        package_names.update(_dependency_name(dep) for dep in deps)

    assert "comfyui" not in package_names
    assert "comfy-ui" not in package_names


def test_uv_base_install_succeeds_and_runtime_is_usable_without_torch(
    tmp_path: Path,
) -> None:
    submodule_path = _repo_root() / "vendor" / "ComfyUI"
    assert submodule_path.is_dir()

    venv_dir = tmp_path / ".venv"
    _run_command("uv", "venv", str(venv_dir))

    venv_python = _venv_python_path(venv_dir)
    assert venv_python.is_file()

    _run_command("uv", "pip", "install", "--python", str(venv_python), ".")
    import_result = _run_command(
        str(venv_python),
        "-c",
        (
            "import importlib.util; "
            "import comfy_diffusion; "
            "assert importlib.util.find_spec('torch') is None; "
            "result = comfy_diffusion.check_runtime(); "
            "assert isinstance(result, dict); "
            "required = {'comfyui_version', 'device', 'vram_total_mb', "
            "'vram_free_mb', 'python_version'}; "
            "assert required.issubset(result.keys())"
        ),
        cwd=tmp_path,
    )
    assert import_result.returncode == 0


def test_uv_video_extra_install_succeeds_and_runtime_is_usable(tmp_path: Path) -> None:
    submodule_path = _repo_root() / "vendor" / "ComfyUI"
    assert submodule_path.is_dir()

    venv_dir = tmp_path / ".venv"
    _run_command("uv", "venv", str(venv_dir))

    venv_python = _venv_python_path(venv_dir)
    assert venv_python.is_file()

    _run_command("uv", "pip", "install", "--python", str(venv_python), ".[video]")
    import_result = _run_command(
        str(venv_python),
        "-c",
        (
            "import importlib.util; "
            "import comfy_diffusion; "
            "assert importlib.util.find_spec('cv2') is not None; "
            "result = comfy_diffusion.check_runtime(); "
            "assert isinstance(result, dict); "
            "required = {'comfyui_version', 'device', 'vram_total_mb', "
            "'vram_free_mb', 'python_version'}; "
            "assert required.issubset(result.keys())"
        ),
        cwd=tmp_path,
    )
    assert import_result.returncode == 0
