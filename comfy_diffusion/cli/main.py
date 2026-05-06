"""Small operational CLI for the comfy-diffusion Python package."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any

import click
import typer

from comfy_diffusion._runtime import COMFYUI_PINNED_TAG, _comfyui_root
from comfy_diffusion.downloader import (
    CivitAIModelEntry,
    HFModelEntry,
    ModelEntry,
    URLModelEntry,
    download_models,
)
from comfy_diffusion.runtime import check_runtime

app = typer.Typer(
    name="comfy-diffusion",
    help="Operational tools for the comfy-diffusion Python package.",
    no_args_is_help=True,
)
runtime_app = typer.Typer(help="Inspect the local ComfyUI runtime.", no_args_is_help=True)
models_app = typer.Typer(help="Inspect and download model files.", no_args_is_help=True)
nodes_app = typer.Typer(help="Inspect raw ComfyUI nodes.", no_args_is_help=True)

app.add_typer(runtime_app, name="runtime")
app.add_typer(models_app, name="models")
app.add_typer(nodes_app, name="nodes")

_MODEL_DIRS = (
    "checkpoints",
    "unet",
    "diffusion_models",
    "text_encoders",
    "clip",
    "vae",
    "loras",
    "embeddings",
    "upscale_models",
    "upscale",
    "audio_encoders",
    "llm",
    "clip_vision",
)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_models_dir() -> Path:
    return Path.home() / ".cache" / "comfy-diffusion" / "models"


def _custom_node_paths(custom_node: list[Path] | None) -> list[Path]:
    return custom_node or []


@runtime_app.command("check")
def runtime_check(
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Print runtime diagnostics from ``comfy_diffusion.check_runtime()``."""
    result = check_runtime()
    if json_output:
        _echo_json(result)
        return

    if "error" in result:
        typer.echo(f"error: {result['error']}")
    for key in ("comfyui_version", "device", "vram_total_mb", "vram_free_mb", "python_version"):
        typer.echo(f"{key}: {result.get(key)}")


@runtime_app.command("paths")
def runtime_paths(
    models_dir: Annotated[
        Path | None,
        typer.Option("--models-dir", help="Models directory to resolve."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Print package, runtime, and model paths."""
    resolved_models_dir = (models_dir or _default_models_dir()).expanduser().resolve()
    payload = {
        "package_root": str(_package_root()),
        "comfyui_root": str(_comfyui_root()),
        "models_dir": str(resolved_models_dir),
        "comfyui_pinned_tag": COMFYUI_PINNED_TAG,
    }
    if json_output:
        _echo_json(payload)
        return
    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


def _require_str(raw: dict[str, Any], key: str, *, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise typer.BadParameter(f"models[{index}].{key} must be a non-empty string")
    return value


def _optional_sha256(raw: dict[str, Any]) -> str | None:
    value = raw.get("sha256")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise typer.BadParameter("sha256 must be a non-empty string when provided")
    return value


def _parse_manifest(path: Path) -> list[ModelEntry]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise typer.BadParameter(f"could not read manifest: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"manifest is not valid JSON: {exc}") from exc

    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise typer.BadParameter("manifest root must be an object with a models list")

    entries: list[ModelEntry] = []
    for index, raw_entry in enumerate(models):
        if not isinstance(raw_entry, dict):
            raise typer.BadParameter(f"models[{index}] must be an object")
        entry_type = raw_entry.get("type")
        dest = _require_str(raw_entry, "dest", index=index)
        sha256 = _optional_sha256(raw_entry)

        if entry_type == "hf":
            entries.append(
                HFModelEntry(
                    repo_id=_require_str(raw_entry, "repo_id", index=index),
                    filename=_require_str(raw_entry, "filename", index=index),
                    dest=dest,
                    sha256=sha256,
                )
            )
        elif entry_type == "civitai":
            model_id = raw_entry.get("model_id")
            if not isinstance(model_id, int):
                raise typer.BadParameter(f"models[{index}].model_id must be an integer")
            version_id = raw_entry.get("version_id")
            if version_id is not None and not isinstance(version_id, int):
                raise typer.BadParameter(f"models[{index}].version_id must be an integer")
            entries.append(
                CivitAIModelEntry(
                    model_id=model_id,
                    version_id=version_id,
                    dest=dest,
                    sha256=sha256,
                )
            )
        elif entry_type == "url":
            entries.append(
                URLModelEntry(
                    url=_require_str(raw_entry, "url", index=index),
                    dest=dest,
                    sha256=sha256,
                )
            )
        else:
            raise typer.BadParameter(
                f"models[{index}].type must be one of: hf, civitai, url"
            )
    return entries


@models_app.command("download")
def models_download(
    manifest: Annotated[
        Path,
        typer.Option("--manifest", exists=True, dir_okay=False, help="Model manifest JSON file."),
    ],
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir", help="Base directory for relative model destinations."),
    ],
    quiet: Annotated[
        bool,
        typer.Option("--quiet", help="Suppress download progress output."),
    ] = False,
) -> None:
    """Download models from a JSON manifest."""
    entries = _parse_manifest(manifest)
    download_models(entries, models_dir=models_dir, quiet=quiet)
    typer.echo(f"Downloaded {len(entries)} model file(s).")


@models_app.command("list")
def models_list(
    models_dir: Annotated[
        Path,
        typer.Option("--models-dir", exists=True, file_okay=False, help="Models directory."),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """List known model directories and files without network access."""
    root = models_dir.expanduser().resolve()
    payload: dict[str, list[str]] = {}
    for dirname in _MODEL_DIRS:
        directory = root / dirname
        if directory.is_dir():
            payload[dirname] = sorted(
                str(path.relative_to(directory))
                for path in directory.rglob("*")
                if path.is_file()
            )
        else:
            payload[dirname] = []

    if json_output:
        _echo_json({"models_dir": str(root), "directories": payload})
        return

    typer.echo(f"models_dir: {root}")
    for dirname, files in payload.items():
        typer.echo(f"{dirname}: {len(files)} file(s)")
        for file_path in files:
            typer.echo(f"  {file_path}")


@nodes_app.command("list")
def nodes_list(
    include_api: Annotated[
        bool,
        typer.Option("--include-api", help="Include built-in ComfyUI API nodes."),
    ] = False,
    custom_node: Annotated[
        list[Path] | None,
        typer.Option(
            "--custom-node",
            help="Explicit custom node .py file or package directory to load.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """List raw ComfyUI nodes, optionally including explicit custom node paths."""
    from comfy_diffusion.nodes import list_nodes

    try:
        nodes = list_nodes(
            include_api=include_api,
            custom_node_paths=_custom_node_paths(custom_node),
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if json_output:
        _echo_json(
            {
                "include_api": include_api,
                "custom_nodes": [str(path) for path in _custom_node_paths(custom_node)],
                "nodes": {node_id: _jsonable(info) for node_id, info in nodes.items()},
            }
        )
        return

    for node_id, info in nodes.items():
        suffix = " [api]" if info.is_api_node else ""
        if info.is_custom_node:
            suffix += " [custom]"
        typer.echo(f"{node_id}{suffix}: {info.class_name}")


@nodes_app.command("show")
def nodes_show(
    node_id: Annotated[str, typer.Argument(help="ComfyUI node id to inspect.")],
    include_api: Annotated[
        bool,
        typer.Option("--include-api", help="Include built-in ComfyUI API nodes."),
    ] = False,
    custom_node: Annotated[
        list[Path] | None,
        typer.Option(
            "--custom-node",
            help="Explicit custom node .py file or package directory to load.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Show metadata for one raw ComfyUI node."""
    from comfy_diffusion.nodes import get_node_info

    try:
        info = get_node_info(
            node_id,
            include_api=include_api,
            custom_node_paths=_custom_node_paths(custom_node),
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    payload = _jsonable(info)
    if json_output:
        _echo_json(payload)
        return

    for key, value in payload.items():
        typer.echo(f"{key}: {value}")


@nodes_app.command("install")
def nodes_install(
    repo_url: Annotated[str, typer.Argument(help="Git repository URL to install.")],
    ref: Annotated[
        str | None,
        typer.Option("--ref", help="Branch, tag, or commit to checkout after fetching."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Install directory name. Defaults to repo name."),
    ] = None,
    custom_nodes_dir: Annotated[
        Path | None,
        typer.Option("--custom-nodes-dir", help="Directory for installed custom nodes."),
    ] = None,
    install_deps: Annotated[
        bool,
        typer.Option("--install-deps", help="Install requirements.txt with uv pip."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """Install or update a trusted custom node Git repository."""
    from comfy_diffusion.custom_nodes import install_custom_node

    try:
        result = install_custom_node(
            repo_url,
            ref=ref,
            name=name,
            custom_nodes_dir=custom_nodes_dir,
            install_deps=install_deps,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    payload = _jsonable(result)
    if json_output:
        _echo_json(payload)
        return

    typer.echo(f"name: {result.name}")
    typer.echo(f"path: {result.path}")
    typer.echo(f"commit: {result.commit}")
    typer.echo(f"installed: {result.installed}")
    typer.echo(f"updated: {result.updated}")
    if result.requirements_path is not None:
        typer.echo(f"requirements: {result.requirements_path}")
        if result.dependencies_installed:
            typer.echo("dependencies: installed")
        elif result.dependency_command is not None:
            typer.echo(f"dependencies: run {' '.join(result.dependency_command)}")


@nodes_app.command("installed")
def nodes_installed(
    custom_nodes_dir: Annotated[
        Path | None,
        typer.Option("--custom-nodes-dir", help="Directory for installed custom nodes."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print machine-readable JSON."),
    ] = False,
) -> None:
    """List custom node repositories installed by comfy-diffusion."""
    from comfy_diffusion.custom_nodes import list_installed_custom_nodes

    installed = list_installed_custom_nodes(custom_nodes_dir=custom_nodes_dir)
    if json_output:
        _echo_json({"custom_nodes": [_jsonable(item) for item in installed]})
        return

    for item in installed:
        suffix = " [requirements]" if item.has_requirements else ""
        typer.echo(f"{item.name}{suffix}: {item.path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
