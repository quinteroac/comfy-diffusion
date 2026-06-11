"""Tests for US-004 README runtime-first quick start documentation."""

from __future__ import annotations

from pathlib import Path


def _readme_text() -> str:
    return (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")


def _quick_start_section(readme_text: str) -> str:
    start = readme_text.index("## Quick Start")
    end = readme_text.find("\n## ", start + 1)
    if end == -1:
        return readme_text[start:]
    return readme_text[start:end]


def test_readme_quick_start_shows_check_runtime_as_first_call() -> None:
    section = _quick_start_section(_readme_text())

    check_runtime_call = section.index("runtime = check_runtime()")
    model_loading_call = section.index("ModelManager(")

    assert check_runtime_call < model_loading_call


def test_readme_quick_start_explains_automatic_comfyui_download() -> None:
    section = _quick_start_section(_readme_text())

    assert "automatic download" in section
    assert "pinned ComfyUI ref" in section


def test_readme_quick_start_shows_error_dict_handling() -> None:
    section = _quick_start_section(_readme_text())

    assert 'if "error" in runtime:' in section
    assert "returns an error dict" in section
