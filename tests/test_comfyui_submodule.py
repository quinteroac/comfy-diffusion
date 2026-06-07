"""Tests for US-002 ComfyUI submodule pinning."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

SUBMODULE_NAME = "vendor/ComfyUI"
SUBMODULE_URL = "https://github.com/Comfy-Org/ComfyUI.git"
PINNED_TAG = "v0.24.1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd or _repo_root(),
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_submodule_is_gitlink_and_pinned_to_stable_tag() -> None:
    index_entry = _run_git("ls-files", "--stage", "--", SUBMODULE_NAME)
    mode, sha, stage_and_path = index_entry.split(maxsplit=2)

    assert mode == "160000"
    assert len(sha) == 40
    assert stage_and_path.endswith(SUBMODULE_NAME)
    assert re.fullmatch(r"v\d+\.\d+\.\d+", PINNED_TAG)

    checked_out_tag = _run_git(
        "-C", SUBMODULE_NAME, "describe", "--tags", "--exact-match", "HEAD"
    )
    assert checked_out_tag == PINNED_TAG


def test_gitmodules_references_comfyui_repository() -> None:
    path_value = _run_git(
        "config", "-f", ".gitmodules", "--get", f"submodule.{SUBMODULE_NAME}.path"
    )
    url_value = _run_git(
        "config", "-f", ".gitmodules", "--get", f"submodule.{SUBMODULE_NAME}.url"
    )

    assert path_value == SUBMODULE_NAME
    assert url_value == SUBMODULE_URL


def test_submodule_update_init_restores_pinned_tag_without_manual_steps() -> None:
    submodule_path = _repo_root() / SUBMODULE_NAME
    _run_git("submodule", "deinit", "-f", "--", SUBMODULE_NAME)
    shutil.rmtree(submodule_path, ignore_errors=True)

    _run_git("submodule", "update", "--init", "--checkout", "--", SUBMODULE_NAME)
    assert submodule_path.is_dir()

    checked_out_tag = _run_git(
        "-C", SUBMODULE_NAME, "describe", "--tags", "--exact-match", "HEAD"
    )
    assert checked_out_tag == PINNED_TAG


def test_pinned_tag_is_documented() -> None:
    gitmodules_text = (_repo_root() / ".gitmodules").read_text(encoding="utf-8")
    assert f"Pinned ComfyUI release tag: {PINNED_TAG}" in gitmodules_text
