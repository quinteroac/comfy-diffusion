"""Tests for the compact comfy-diffusion CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from comfy_diffusion.cli import main as cli_main
from comfy_diffusion.downloader import URLModelEntry
from comfy_diffusion.nodes import NodeInfo

runner = CliRunner()


def test_runtime_check_prints_json(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        cli_main,
        "check_runtime",
        lambda: {
            "comfyui_version": "0.18.3",
            "device": "cpu",
            "vram_total_mb": 0,
            "vram_free_mb": 0,
            "python_version": "3.12.0",
        },
    )

    result = runner.invoke(cli_main.app, ["runtime", "check", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["device"] == "cpu"


def test_runtime_paths_prints_expected_keys(tmp_path: Path) -> None:
    result = runner.invoke(
        cli_main.app,
        ["runtime", "paths", "--models-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["package_root"].endswith("comfy_diffusion")
    assert payload["models_dir"] == str(tmp_path.resolve())
    assert payload["comfyui_pinned_tag"]
    assert payload["comfyui_root"]


def test_models_list_reports_known_directories(tmp_path: Path) -> None:
    model_file = tmp_path / "checkpoints" / "demo.safetensors"
    model_file.parent.mkdir()
    model_file.write_bytes(b"demo")

    result = runner.invoke(
        cli_main.app,
        ["models", "list", "--models-dir", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["directories"]["checkpoints"] == ["demo.safetensors"]
    assert "vae" in payload["directories"]


def test_models_download_parses_manifest_and_calls_downloader(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "type": "url",
                        "url": "https://example.com/model.safetensors",
                        "dest": "checkpoints",
                        "sha256": "a" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    download_mock = MagicMock()
    monkeypatch.setattr(cli_main, "download_models", download_mock)

    result = runner.invoke(
        cli_main.app,
        [
            "models",
            "download",
            "--manifest",
            str(manifest),
            "--models-dir",
            str(tmp_path / "models"),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    entries = download_mock.call_args.args[0]
    assert entries == [
        URLModelEntry(
            url="https://example.com/model.safetensors",
            dest="checkpoints",
            sha256="a" * 64,
        )
    ]
    assert download_mock.call_args.kwargs["quiet"] is True


def test_models_download_rejects_invalid_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models": [{"type": "url", "dest": "checkpoints"}]}))

    result = runner.invoke(
        cli_main.app,
        [
            "models",
            "download",
            "--manifest",
            str(manifest),
            "--models-dir",
            str(tmp_path / "models"),
        ],
    )

    assert result.exit_code != 0
    assert "url must be a non-empty string" in result.output


def test_models_download_requires_models_dir(tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"models": []}), encoding="utf-8")

    result = runner.invoke(
        cli_main.app,
        ["models", "download", "--manifest", str(manifest)],
    )

    assert result.exit_code != 0
    assert "models-dir" in result.output


def test_nodes_list_prints_json(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "comfy_diffusion.nodes.list_nodes",
        lambda include_api=False, custom_node_paths=None: {
            "VAEDecode": NodeInfo(
                node_id="VAEDecode",
                class_name="VAEDecode",
                display_name=None,
                module="nodes",
                category="latent",
                input_types={"required": {}},
                return_types=("IMAGE",),
                function_name="decode",
                is_api_node=False,
                is_custom_node=False,
                source_path=None,
            )
        },
    )

    result = runner.invoke(cli_main.app, ["nodes", "list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"]["VAEDecode"]["return_types"] == ["IMAGE"]


def test_nodes_show_prints_json(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "comfy_diffusion.nodes.get_node_info",
        lambda node_id, include_api=False, custom_node_paths=None: NodeInfo(
            node_id=node_id,
            class_name="VAEDecode",
            display_name=None,
            module="nodes",
            category="latent",
            input_types={"required": {}},
            return_types=("IMAGE",),
            function_name="decode",
            is_api_node=False,
            is_custom_node=False,
            source_path=None,
        ),
    )

    result = runner.invoke(cli_main.app, ["nodes", "show", "VAEDecode", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["node_id"] == "VAEDecode"


def test_nodes_list_accepts_include_api_without_loading_custom_nodes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[bool] = []

    def _fake_list_nodes(
        include_api: bool = False,
        custom_node_paths: list[Path] | None = None,
    ) -> dict[str, NodeInfo]:
        calls.append(include_api)
        return {}

    monkeypatch.setattr("comfy_diffusion.nodes.list_nodes", _fake_list_nodes)

    result = runner.invoke(cli_main.app, ["nodes", "list", "--include-api", "--json"])

    assert result.exit_code == 0
    assert calls == [True]


def test_nodes_install_prints_json(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from comfy_diffusion.custom_nodes import CustomNodeInstallResult

    monkeypatch.setattr(
        "comfy_diffusion.custom_nodes.install_custom_node",
        lambda repo_url, **kwargs: CustomNodeInstallResult(
            repo_url=repo_url,
            name="demo-node",
            path=tmp_path / "demo-node",
            ref=kwargs["ref"],
            commit="abc123",
            installed=True,
            updated=False,
            requirements_path=None,
            dependencies_installed=False,
            dependency_command=None,
        ),
    )

    result = runner.invoke(
        cli_main.app,
        ["nodes", "install", "https://github.com/acme/demo-node.git", "--ref", "v1", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["name"] == "demo-node"
    assert payload["ref"] == "v1"


def test_nodes_installed_prints_json(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from comfy_diffusion.custom_nodes import CustomNodeInstallInfo

    monkeypatch.setattr(
        "comfy_diffusion.custom_nodes.list_installed_custom_nodes",
        lambda custom_nodes_dir=None: [
            CustomNodeInstallInfo(
                name="demo-node",
                path=tmp_path / "demo-node",
                has_requirements=True,
            )
        ],
    )

    result = runner.invoke(cli_main.app, ["nodes", "installed", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["custom_nodes"][0]["has_requirements"] is True


def test_nodes_list_passes_custom_node_paths(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    custom_node = tmp_path / "custom.py"
    calls: list[list[Path] | None] = []

    def _fake_list_nodes(
        include_api: bool = False,
        custom_node_paths: list[Path] | None = None,
    ) -> dict[str, NodeInfo]:
        calls.append(custom_node_paths)
        return {}

    monkeypatch.setattr("comfy_diffusion.nodes.list_nodes", _fake_list_nodes)

    result = runner.invoke(
        cli_main.app,
        ["nodes", "list", "--custom-node", str(custom_node), "--json"],
    )

    assert result.exit_code == 0
    assert calls == [[custom_node]]


def test_nodes_show_passes_custom_node_paths(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    custom_node = tmp_path / "custom.py"
    calls: list[list[Path] | None] = []

    def _fake_get_node_info(
        node_id: str,
        include_api: bool = False,
        custom_node_paths: list[Path] | None = None,
    ) -> NodeInfo:
        calls.append(custom_node_paths)
        return NodeInfo(
            node_id=node_id,
            class_name="Custom",
            display_name=None,
            module="custom_nodes.custom",
            category="custom",
            input_types={},
            return_types=("IMAGE",),
            function_name="execute",
            is_api_node=False,
            is_custom_node=True,
            source_path=str(custom_node),
        )

    monkeypatch.setattr("comfy_diffusion.nodes.get_node_info", _fake_get_node_info)

    result = runner.invoke(
        cli_main.app,
        ["nodes", "show", "Custom", "--custom-node", str(custom_node), "--json"],
    )

    assert result.exit_code == 0
    assert calls == [[custom_node]]
