"""Tests for US-002 ComfyUI submodule pinning."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

SUBMODULE_NAME = "vendor/ComfyUI"
SUBMODULE_URL = "https://github.com/Comfy-Org/ComfyUI.git"
PINNED_REF = "fb991e2c1e7476809d566a4620c2132e05a466dd"


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


def test_submodule_is_gitlink_and_checked_out_at_pinned_ref() -> None:
    index_entry = _run_git("ls-files", "--stage", "--", SUBMODULE_NAME)
    mode, sha, stage_and_path = index_entry.split(maxsplit=2)

    assert mode == "160000"
    assert len(sha) == 40
    assert stage_and_path.endswith(SUBMODULE_NAME)
    assert re.fullmatch(r"(v\d+\.\d+\.\d+|[0-9a-f]{40})", PINNED_REF)

    checked_out_sha = _run_git("-C", SUBMODULE_NAME, "rev-parse", "HEAD")
    assert checked_out_sha == PINNED_REF


def test_gitmodules_references_comfyui_repository() -> None:
    path_value = _run_git(
        "config", "-f", ".gitmodules", "--get", f"submodule.{SUBMODULE_NAME}.path"
    )
    url_value = _run_git(
        "config", "-f", ".gitmodules", "--get", f"submodule.{SUBMODULE_NAME}.url"
    )

    assert path_value == SUBMODULE_NAME
    assert url_value == SUBMODULE_URL


def test_submodule_head_contains_pinned_ref() -> None:
    submodule_path = _repo_root() / SUBMODULE_NAME
    assert submodule_path.is_dir()

    is_ancestor = subprocess.run(
        ["git", "-C", SUBMODULE_NAME, "merge-base", "--is-ancestor", PINNED_REF, "HEAD"],
        cwd=_repo_root(),
        check=False,
    )
    assert is_ancestor.returncode == 0


def test_pinned_ref_is_documented() -> None:
    gitmodules_text = (_repo_root() / ".gitmodules").read_text(encoding="utf-8")
    assert f"Pinned ComfyUI ref: {PINNED_REF}" in gitmodules_text
