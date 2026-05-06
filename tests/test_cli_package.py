"""Tests for the compact comfy-diffusion CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from comfy_diffusion.cli import main as cli_main
from comfy_diffusion.downloader import URLModelEntry

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
