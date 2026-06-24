"""Tests for the local Krea2 Turbo text-to-image pipeline."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "krea2" / "turbo.py"
)

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_REBALANCE_PATCH = "comfy_diffusion.conditioning.rebalance_krea2_conditioning"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_EMPTY_LATENT_PATCH = "comfy_diffusion.latent.empty_latent_image"
_SAMPLE_PATCH = "comfy_diffusion.sampling.sample"
_VAE_DECODE_PATCH = "comfy_diffusion.vae.vae_decode"

_OK_RUNTIME = {"python_version": "3.12.0"}
_ERR_RUNTIME = {"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"}


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file()


def test_pipeline_parses_without_syntax_errors() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert isinstance(ast.parse(source, filename=str(_PIPELINE_FILE)), ast.Module)


def test_no_top_level_comfy_or_torch_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for line_number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(("import comfy.", "from comfy.", "import torch", "from torch")):
            assert line.startswith("    "), f"Top-level runtime import at line {line_number}"


def test_dunder_all_values() -> None:
    from comfy_diffusion.pipelines.image.krea2 import turbo

    assert set(turbo.__all__) == {"manifest", "run"}


def test_manifest_returns_official_krea2_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.krea2.turbo import manifest

    entries = manifest()

    assert len(entries) == 3
    assert all(isinstance(entry, HFModelEntry) for entry in entries)
    assert {entry.repo_id for entry in entries} == {"Comfy-Org/Krea-2"}
    assert [entry.filename for entry in entries] == [
        "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "vae/qwen_image_vae.safetensors",
    ]
    assert [str(entry.dest) for entry in entries] == [
        "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        "vae/qwen_image_vae.safetensors",
    ]


def test_run_signature_defaults() -> None:
    from comfy_diffusion.pipelines.image.krea2.turbo import run

    signature = inspect.signature(run)

    assert signature.parameters["width"].default == 1024
    assert signature.parameters["height"].default == 1024
    assert signature.parameters["steps"].default == 8
    assert signature.parameters["cfg"].default == 1.0
    assert signature.parameters["sampler_name"].default == "euler"
    assert signature.parameters["scheduler"].default == "simple"
    assert signature.parameters["denoise"].default == 1.0
    assert signature.parameters["seed"].default == 0
    assert signature.parameters["rebalance_multiplier"].default == 4.0


def test_run_raises_when_runtime_check_fails(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.krea2 import turbo

    with patch(_RUNTIME_PATCH, return_value=_ERR_RUNTIME):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            turbo.run(models_dir=tmp_path, prompt="test")


def test_run_wires_local_krea2_pipeline(tmp_path: Path) -> None:
    from comfy_diffusion.conditioning import KREA2_REBALANCE_DEFAULT_WEIGHTS
    from comfy_diffusion.pipelines.image.krea2 import turbo

    mm = MagicMock()
    model = MagicMock(name="model")
    clip = MagicMock(name="clip")
    vae = MagicMock(name="vae")
    mm.load_unet.return_value = model
    mm.load_clip.return_value = clip
    mm.load_vae.return_value = vae

    positive_raw = MagicMock(name="positive_raw")
    positive = MagicMock(name="positive")
    negative = MagicMock(name="negative")
    latent = {"samples": MagicMock(name="latent")}
    latent_out = MagicMock(name="latent_out")
    image = MagicMock(name="image")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm) as model_manager,
        patch(_ENCODE_PATCH, return_value=positive_raw) as encode_prompt,
        patch(_REBALANCE_PATCH, return_value=positive) as rebalance,
        patch(_ZERO_OUT_PATCH, return_value=negative) as zero_out,
        patch(_EMPTY_LATENT_PATCH, return_value=latent) as empty_latent,
        patch(_SAMPLE_PATCH, return_value=latent_out) as sample,
        patch(_VAE_DECODE_PATCH, return_value=image) as vae_decode,
    ):
        result = turbo.run(
            models_dir=tmp_path,
            prompt="cinematic graphite city",
            width=1280,
            height=768,
            steps=10,
            cfg=1.25,
            sampler_name="euler_ancestral",
            scheduler="beta",
            denoise=0.9,
            seed=123,
        )

    assert result == [image]
    model_manager.assert_called_once_with(tmp_path)
    mm.load_unet.assert_called_once_with(
        tmp_path / "diffusion_models" / "krea2_turbo_fp8_scaled.safetensors"
    )
    mm.load_clip.assert_called_once_with(
        tmp_path / "text_encoders" / "qwen3vl_4b_fp8_scaled.safetensors",
        clip_type="krea2",
    )
    mm.load_vae.assert_called_once_with(tmp_path / "vae" / "qwen_image_vae.safetensors")
    encode_prompt.assert_called_once_with(clip, "cinematic graphite city")
    rebalance.assert_called_once_with(
        positive_raw,
        multiplier=4.0,
        per_layer_weights=KREA2_REBALANCE_DEFAULT_WEIGHTS,
    )
    zero_out.assert_called_once_with(positive)
    empty_latent.assert_called_once_with(1280, 768, batch_size=1)
    sample.assert_called_once_with(
        model,
        positive,
        negative,
        latent,
        10,
        1.25,
        "euler_ancestral",
        "beta",
        123,
        denoise=0.9,
    )
    vae_decode.assert_called_once_with(vae, latent_out)


def test_run_uses_custom_filenames_and_rebalance_weights(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.krea2 import turbo

    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    custom_weights = "1,1,1,1,1,1,1,1,1,1,1,1"

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=MagicMock()),
        patch(_REBALANCE_PATCH, return_value=MagicMock()) as rebalance,
        patch(_ZERO_OUT_PATCH, return_value=MagicMock()),
        patch(_EMPTY_LATENT_PATCH, return_value={"samples": MagicMock()}),
        patch(_SAMPLE_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_PATCH, return_value=MagicMock()),
    ):
        turbo.run(
            models_dir=tmp_path,
            prompt="test",
            rebalance_multiplier=2.5,
            rebalance_per_layer_weights=custom_weights,
            unet_filename="custom_unet.safetensors",
            clip_filename="custom_clip.safetensors",
            vae_filename="custom_vae.safetensors",
        )

    mm.load_unet.assert_has_calls([call(tmp_path / "custom_unet.safetensors")])
    mm.load_clip.assert_called_once_with(tmp_path / "custom_clip.safetensors", clip_type="krea2")
    mm.load_vae.assert_called_once_with(tmp_path / "custom_vae.safetensors")
    assert rebalance.call_args.kwargs == {
        "multiplier": 2.5,
        "per_layer_weights": custom_weights,
    }
