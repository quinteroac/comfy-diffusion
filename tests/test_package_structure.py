"""Tests for US-001 package structure and exports."""

from __future__ import annotations

import tomllib
from pathlib import Path

import comfy_diffusion
from comfy_diffusion import check_runtime


def test_package_root_exists_with_init() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "comfy_diffusion"
    assert package_dir.is_dir()
    assert (package_dir / "__init__.py").is_file()


def test_check_runtime_is_public_symbol() -> None:
    assert callable(check_runtime)
    assert comfy_diffusion.check_runtime is check_runtime
    assert "check_runtime" in comfy_diffusion.__all__


def test_raw_node_api_is_explicit_submodule_only() -> None:
    assert "list_nodes" not in comfy_diffusion.__all__
    assert not hasattr(comfy_diffusion, "list_nodes")


def test_package_uses_src_less_layout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "comfy_diffusion").is_dir()
    assert not (repo_root / "src" / "comfy_diffusion").exists()


def test_only_comfy_diffusion_package_is_discovered() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "comfy_diffusion*"
    ]


def test_comfy_diffusion_is_the_only_console_script() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {
        "comfy-diffusion": "comfy_diffusion.cli.main:app"
    }


def test_removed_application_layers_are_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for path in ("cli", "server", "mcp", "frontend", "parallax"):
        assert not (repo_root / path).exists()
