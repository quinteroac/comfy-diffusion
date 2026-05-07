"""Tests for US-001 LTXV audio VAE loading behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from comfy_diffusion.models import ModelManager


def _install_fake_ltxv_audio_vae_loader_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dict: dict[str, Any],
    metadata: dict[str, Any] | None,
    audio_vae_object: Any,
    resolved_checkpoint_path: str,
) -> dict[str, list[Any]]:
    calls: dict[str, list[Any]] = {
        "add_model_folder_path": [],
        "get_full_path_or_raise": [],
        "load_torch_file": [],
        "state_dict_prefix_replace": [],
        "VAE": [],
        "throw_exception_if_invalid": [],
    }

    folder_paths_module = ModuleType("folder_paths")

    def add_model_folder_path(
        folder_name: str, full_folder_path: str, is_default: bool = False
    ) -> None:
        calls["add_model_folder_path"].append((folder_name, full_folder_path, is_default))

    def get_full_path_or_raise(folder_name: str, filename: str) -> str:
        calls["get_full_path_or_raise"].append((folder_name, filename))
        if folder_name != "checkpoints":
            raise AssertionError(f"unexpected folder name: {folder_name}")
        return resolved_checkpoint_path

    setattr(folder_paths_module, "add_model_folder_path", add_model_folder_path)
    setattr(folder_paths_module, "get_full_path_or_raise", get_full_path_or_raise)

    comfy_utils_module = ModuleType("comfy.utils")

    def load_torch_file(path: str, *, return_metadata: bool) -> tuple[dict[str, Any], Any]:
        calls["load_torch_file"].append((path, return_metadata))
        return state_dict, metadata

    setattr(comfy_utils_module, "load_torch_file", load_torch_file)
    remapped_state_dict = {"autoencoder.encoder.weight": object(), "vocoder.weight": object()}

    def state_dict_prefix_replace(
        sd: dict[str, Any],
        replace_prefix: dict[str, str],
        *,
        filter_keys: bool,
    ) -> dict[str, Any]:
        calls["state_dict_prefix_replace"].append((sd, replace_prefix, filter_keys))
        return remapped_state_dict

    setattr(comfy_utils_module, "state_dict_prefix_replace", state_dict_prefix_replace)

    comfy_sd_module = ModuleType("comfy.sd")

    def throw_exception_if_invalid() -> None:
        calls["throw_exception_if_invalid"].append(())

    def make_vae(*, sd: dict[str, Any], metadata: dict[str, Any] | None) -> Any:
        calls["VAE"].append((sd, metadata))
        audio_vae_object.throw_exception_if_invalid.side_effect = throw_exception_if_invalid
        return audio_vae_object

    setattr(comfy_sd_module, "VAE", make_vae)

    comfy_module = ModuleType("comfy")

    setattr(comfy_module, "utils", comfy_utils_module)
    setattr(comfy_module, "sd", comfy_sd_module)

    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)

    return calls


def test_load_ltxv_audio_vae_calls_loader_and_returns_raw_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    embeddings_dir = models_dir / "embeddings"
    checkpoints_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)

    audio_vae_file = tmp_path / "weights" / "ltxv_audio_vae.safetensors"
    audio_vae_file.parent.mkdir(parents=True)
    audio_vae_file.write_text("stub ltxv audio vae")

    fake_state_dict = {"audio_vae.encoder.weight": object()}
    fake_metadata = {"format": "safetensors"}
    fake_audio_vae = MagicMock()

    calls = _install_fake_ltxv_audio_vae_loader_modules(
        monkeypatch,
        state_dict=fake_state_dict,
        metadata=fake_metadata,
        audio_vae_object=fake_audio_vae,
        resolved_checkpoint_path="/unused/for/absolute/path.safetensors",
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_ltxv_audio_vae(audio_vae_file.parent / "." / audio_vae_file.name)

    assert result is fake_audio_vae
    assert calls["get_full_path_or_raise"] == []
    assert calls["load_torch_file"] == [(str(audio_vae_file.resolve()), True)]
    assert calls["state_dict_prefix_replace"] == [
        (
            fake_state_dict,
            {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
            True,
        )
    ]
    assert len(calls["VAE"]) == 1
    assert calls["VAE"][0][0] is not fake_state_dict
    assert set(calls["VAE"][0][0]) == {"autoencoder.encoder.weight", "vocoder.weight"}
    assert calls["VAE"][0][1] == fake_metadata
    assert calls["throw_exception_if_invalid"] == [()]
    assert calls["add_model_folder_path"] == [
        ("checkpoints", str(checkpoints_dir), True),
        ("embeddings", str(embeddings_dir), True),
        ("diffusion_models", str(models_dir / "unet"), True),
        ("diffusion_models", str(models_dir / "diffusion_models"), False),
        ("text_encoders", str(models_dir / "text_encoders"), True),
        ("text_encoders", str(models_dir / "clip"), False),
        ("vae", str(models_dir / "vae"), True),
        ("llm", str(models_dir / "llm"), True),
        ("upscale_models", str(models_dir / "upscale_models"), True),
        ("latent_upscale_models", str(models_dir / "upscale"), True),
        ("audio_encoders", str(models_dir / "audio_encoders"), True),
    ]


def test_load_ltxv_audio_vae_is_callable_from_models_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    expected_resolved = str((models_dir / "checkpoints" / "audio_vae.safetensors").resolve())
    fake_audio_vae = MagicMock()
    calls = _install_fake_ltxv_audio_vae_loader_modules(
        monkeypatch,
        state_dict={"weights": object()},
        metadata={"meta": "ok"},
        audio_vae_object=fake_audio_vae,
        resolved_checkpoint_path=expected_resolved,
    )

    manager = ModelManager(models_dir=models_dir)
    result = manager.load_ltxv_audio_vae("audio_vae.safetensors")

    assert result is fake_audio_vae
    assert calls["get_full_path_or_raise"] == [("checkpoints", "audio_vae.safetensors")]
    assert calls["load_torch_file"] == [(expected_resolved, True)]


def test_load_ltxv_audio_vae_raises_file_not_found_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = tmp_path / "models"
    (models_dir / "checkpoints").mkdir(parents=True)
    (models_dir / "embeddings").mkdir(parents=True)

    calls = _install_fake_ltxv_audio_vae_loader_modules(
        monkeypatch,
        state_dict={},
        metadata=None,
        audio_vae_object=MagicMock(),
        resolved_checkpoint_path="/unused/path.safetensors",
    )

    missing_audio_vae_path = tmp_path / "weights" / "missing_audio_vae.safetensors"
    expected_path = str(missing_audio_vae_path.resolve())

    with pytest.raises(FileNotFoundError, match="ltxv audio vae file not found") as exc_info:
        ModelManager(models_dir=models_dir).load_ltxv_audio_vae(str(missing_audio_vae_path))

    assert expected_path in str(exc_info.value)
    assert calls["get_full_path_or_raise"] == []
    assert calls["load_torch_file"] == []
    assert calls["VAE"] == []
