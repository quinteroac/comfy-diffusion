"""Tests for custom node Git installer helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import comfy_diffusion.custom_nodes as custom_nodes


def test_install_custom_node_clones_fresh_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], Path | None]] = []

    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append((args, cwd))
        if args[:2] == ["git", "clone"]:
            target = Path(args[3])
            (target / ".git").mkdir(parents=True)
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    result = custom_nodes.install_custom_node(
        "https://github.com/acme/demo-node.git",
        custom_nodes_dir=tmp_path,
    )

    assert result.installed is True
    assert result.updated is False
    assert result.name == "demo-node"
    assert result.commit == "abc123"
    assert calls[0][0] == [
        "git",
        "clone",
        "https://github.com/acme/demo-node.git",
        str(result.path),
    ]


def test_install_custom_node_updates_existing_clean_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "demo-node"
    (target / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return "def456"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    result = custom_nodes.install_custom_node(
        "https://github.com/acme/demo-node.git",
        custom_nodes_dir=tmp_path,
    )

    assert result.updated is True
    assert ["git", "fetch", "--all", "--tags"] in calls
    assert ["git", "pull", "--ff-only"] in calls


def test_install_custom_node_ref_checks_out_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(args)
        if args[:2] == ["git", "clone"]:
            (Path(args[3]) / ".git").mkdir(parents=True)
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    custom_nodes.install_custom_node(
        "https://github.com/acme/demo-node.git",
        ref="v1.2.3",
        custom_nodes_dir=tmp_path,
    )

    assert ["git", "checkout", "v1.2.3"] in calls


def test_install_custom_node_dirty_existing_repo_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "demo-node"
    (target / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(args)
        if args == ["git", "status", "--porcelain"]:
            return " M file.py"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    with pytest.raises(RuntimeError, match="local changes"):
        custom_nodes.install_custom_node(
            "https://github.com/acme/demo-node.git",
            custom_nodes_dir=tmp_path,
        )

    assert ["git", "fetch", "--all", "--tags"] not in calls


def test_install_custom_node_installs_requirements_with_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(args)
        if args[:2] == ["git", "clone"]:
            target = Path(args[3])
            (target / ".git").mkdir(parents=True)
            (target / "requirements.txt").write_text("demo-dep\n", encoding="utf-8")
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    result = custom_nodes.install_custom_node(
        "https://github.com/acme/demo-node.git",
        custom_nodes_dir=tmp_path,
        install_deps=True,
    )

    assert result.dependencies_installed is True
    assert ["uv", "pip", "install", "-r", str(result.requirements_path)] in calls


def test_install_custom_node_reports_requirements_without_installing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        calls.append(args)
        if args[:2] == ["git", "clone"]:
            target = Path(args[3])
            (target / ".git").mkdir(parents=True)
            (target / "requirements.txt").write_text("demo-dep\n", encoding="utf-8")
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    result = custom_nodes.install_custom_node(
        "https://github.com/acme/demo-node.git",
        custom_nodes_dir=tmp_path,
    )

    assert result.requirements_path is not None
    assert result.dependencies_installed is False
    assert not any(args[:3] == ["uv", "pip", "install"] for args in calls)


def test_install_custom_node_uses_explicit_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(args: list[str], *, cwd: Path | None = None) -> str:
        if args[:2] == ["git", "clone"]:
            (Path(args[3]) / ".git").mkdir(parents=True)
        if args == ["git", "rev-parse", "HEAD"]:
            return "abc123"
        return ""

    monkeypatch.setattr(custom_nodes, "_run_command", fake_run)

    result = custom_nodes.install_custom_node(
        "https://github.com/acme/demo-node.git",
        name="custom-name",
        custom_nodes_dir=tmp_path,
    )

    assert result.name == "custom-name"
    assert result.path == tmp_path / "custom-name"


def test_install_custom_node_rejects_path_like_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid custom node install name"):
        custom_nodes.install_custom_node(
            "https://github.com/acme/demo-node.git",
            name="../outside",
            custom_nodes_dir=tmp_path,
        )


def test_list_installed_custom_nodes(tmp_path: Path) -> None:
    node_dir = tmp_path / "demo-node"
    node_dir.mkdir()
    (node_dir / "requirements.txt").write_text("demo-dep\n", encoding="utf-8")
    (tmp_path / "not-a-dir").write_text("", encoding="utf-8")

    installed = custom_nodes.list_installed_custom_nodes(custom_nodes_dir=tmp_path)

    assert installed[0].name == "demo-node"
    assert installed[0].has_requirements is True
