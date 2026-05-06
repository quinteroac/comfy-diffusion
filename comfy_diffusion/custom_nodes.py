"""Install and inspect trusted ComfyUI custom node repositories."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomNodeInstallResult:
    """Result from installing or updating a custom node repository."""

    repo_url: str
    name: str
    path: Path
    ref: str | None
    commit: str
    installed: bool
    updated: bool
    requirements_path: Path | None
    dependencies_installed: bool
    dependency_command: tuple[str, ...] | None


@dataclass(frozen=True)
class CustomNodeInstallInfo:
    """A custom node repository installed in the comfy-diffusion cache."""

    name: str
    path: Path
    has_requirements: bool


def default_custom_nodes_dir() -> Path:
    """Return the default custom node install directory."""
    return Path.home() / ".cache" / "comfy-diffusion" / "custom_nodes"


def _derive_repo_name(repo_url: str) -> str:
    normalized = repo_url.rstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    if not name:
        raise ValueError("Could not derive custom node name from repo URL.")
    return name


def _validate_install_name(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or path.name != name or name in {"", ".", ".."}:
        raise ValueError(f"Invalid custom node install name: {name}")
    return name


def _resolve_custom_nodes_dir(custom_nodes_dir: str | Path | None) -> Path:
    directory = (
        Path(custom_nodes_dir)
        if custom_nodes_dir is not None
        else default_custom_nodes_dir()
    )
    return directory.expanduser().resolve()


def _run_command(args: list[str], *, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        detail = stderr or stdout or f"exit code {exc.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(args)}. {detail}") from exc
    return completed.stdout.strip()


def _ensure_clean_git_tree(target: Path) -> None:
    status = _run_command(["git", "status", "--porcelain"], cwd=target)
    if status:
        raise RuntimeError(f"Custom node repository has local changes: {target}")


def _current_commit(target: Path) -> str:
    return _run_command(["git", "rev-parse", "HEAD"], cwd=target)


def install_custom_node(
    repo_url: str,
    *,
    ref: str | None = None,
    name: str | None = None,
    custom_nodes_dir: str | Path | None = None,
    install_deps: bool = False,
) -> CustomNodeInstallResult:
    """Clone or update a trusted ComfyUI custom node repository."""
    install_root = _resolve_custom_nodes_dir(custom_nodes_dir)
    install_name = _validate_install_name(name or _derive_repo_name(repo_url))
    target = install_root / install_name

    installed = False
    updated = False
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(f"Custom node target exists but is not a git repository: {target}")
        _ensure_clean_git_tree(target)
        _run_command(["git", "fetch", "--all", "--tags"], cwd=target)
        if ref is not None:
            _run_command(["git", "checkout", ref], cwd=target)
        else:
            _run_command(["git", "pull", "--ff-only"], cwd=target)
        updated = True
    else:
        install_root.mkdir(parents=True, exist_ok=True)
        _run_command(["git", "clone", repo_url, str(target)])
        if ref is not None:
            _run_command(["git", "checkout", ref], cwd=target)
        installed = True

    requirements_path = target / "requirements.txt"
    has_requirements = requirements_path.is_file()
    dependency_command: tuple[str, ...] | None = None
    dependencies_installed = False
    if has_requirements:
        dependency_command = ("uv", "pip", "install", "-r", str(requirements_path))
        if install_deps:
            _run_command(list(dependency_command), cwd=target)
            dependencies_installed = True

    return CustomNodeInstallResult(
        repo_url=repo_url,
        name=install_name,
        path=target,
        ref=ref,
        commit=_current_commit(target),
        installed=installed,
        updated=updated,
        requirements_path=requirements_path if has_requirements else None,
        dependencies_installed=dependencies_installed,
        dependency_command=dependency_command,
    )


def list_installed_custom_nodes(
    custom_nodes_dir: str | Path | None = None,
) -> list[CustomNodeInstallInfo]:
    """List custom node repositories installed in the configured directory."""
    install_root = _resolve_custom_nodes_dir(custom_nodes_dir)
    if not install_root.is_dir():
        return []

    installed: list[CustomNodeInstallInfo] = []
    for path in sorted(install_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        installed.append(
            CustomNodeInstallInfo(
                name=path.name,
                path=path,
                has_requirements=(path / "requirements.txt").is_file(),
            )
        )
    return installed
