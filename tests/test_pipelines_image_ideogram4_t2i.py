"""Tests for the local Ideogram 4 text-to-image pipeline."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT / "comfy_diffusion" / "pipelines" / "image" / "ideogram4" / "t2i.py"
)

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_ZERO_OUT_PATCH = "comfy_diffusion.conditioning.conditioning_zero_out"
_EMPTY_FLUX2_LATENT_PATCH = "comfy_diffusion.latent.empty_flux2_latent_image"
_CFG_OVERRIDE_PATCH = "comfy_diffusion.sampling.cfg_override"
_RANDOM_NOISE_PATCH = "comfy_diffusion.sampling.random_noise"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_IDEOGRAM4_SCHEDULER_PATCH = "comfy_diffusion.sampling.ideogram4_scheduler"
_DUAL_MODEL_GUIDER_PATCH = "comfy_diffusion.sampling.dual_model_guider"
_SAMPLE_CUSTOM_PATCH = "comfy_diffusion.sampling.sample_custom"
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
    from comfy_diffusion.pipelines.image.ideogram4 import t2i

    assert set(t2i.__all__) == {"manifest", "run"}


def test_manifest_returns_fp8_ideogram4_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.image.ideogram4.t2i import manifest

    entries = manifest()

    assert len(entries) == 4
    assert all(isinstance(entry, HFModelEntry) for entry in entries)
    assert {entry.repo_id for entry in entries} == {"Comfy-Org/Ideogram-4"}
    assert [entry.filename for entry in entries] == [
        "diffusion_models/ideogram4_fp8_scaled.safetensors",
        "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        "vae/flux2-vae.safetensors",
    ]
    assert [str(entry.dest) for entry in entries] == [
        "diffusion_models/ideogram4_fp8_scaled.safetensors",
        "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        "vae/flux2-vae.safetensors",
    ]


def test_run_signature_defaults() -> None:
    from comfy_diffusion.pipelines.image.ideogram4.t2i import run

    signature = inspect.signature(run)

    assert signature.parameters["width"].default == 1024
    assert signature.parameters["height"].default == 1024
    assert signature.parameters["steps"].default == 20
    assert signature.parameters["cfg"].default == 7.0
    assert signature.parameters["cfg_override_value"].default == 3.0
    assert signature.parameters["cfg_override_start"].default == 0.7
    assert signature.parameters["cfg_override_end"].default == 1.0
    assert signature.parameters["seed"].default == 0
    assert signature.parameters["mu"].default == 0.0
    assert signature.parameters["std"].default == 1.75
    assert signature.parameters["sampler_name"].default == "euler"


def test_run_raises_when_runtime_check_fails(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.ideogram4 import t2i

    with patch(_RUNTIME_PATCH, return_value=_ERR_RUNTIME):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            t2i.run(models_dir=tmp_path, prompt="test")


def test_run_wires_local_ideogram4_pipeline(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.image.ideogram4 import t2i

    mm = MagicMock()
    model = MagicMock(name="model")
    model_negative = MagicMock(name="model_negative")
    clip = MagicMock(name="clip")
    vae = MagicMock(name="vae")
    mm.load_unet.side_effect = [model, model_negative]
    mm.load_clip.return_value = clip
    mm.load_vae.return_value = vae

    positive = MagicMock(name="positive")
    negative = MagicMock(name="negative")
    cfg_overridden_model = MagicMock(name="cfg_overridden_model")
    latent = {"samples": MagicMock(name="latent")}
    noise = MagicMock(name="noise")
    sampler = MagicMock(name="sampler")
    sigmas = MagicMock(name="sigmas")
    guider = MagicMock(name="guider")
    latent_out = MagicMock(name="latent_out")
    image = MagicMock(name="image")

    with (
        patch(_RUNTIME_PATCH, return_value=_OK_RUNTIME),
        patch(_MM_PATCH, return_value=mm) as model_manager,
        patch(_ENCODE_PATCH, return_value=positive) as encode_prompt,
        patch(_ZERO_OUT_PATCH, return_value=negative) as zero_out,
        patch(_CFG_OVERRIDE_PATCH, return_value=cfg_overridden_model) as cfg_override,
        patch(_EMPTY_FLUX2_LATENT_PATCH, return_value=latent) as empty_latent,
        patch(_RANDOM_NOISE_PATCH, return_value=noise) as random_noise,
        patch(_GET_SAMPLER_PATCH, return_value=sampler) as get_sampler,
        patch(_IDEOGRAM4_SCHEDULER_PATCH, return_value=sigmas) as scheduler,
        patch(_DUAL_MODEL_GUIDER_PATCH, return_value=guider) as dual_guider,
        patch(_SAMPLE_CUSTOM_PATCH, return_value=(latent_out, MagicMock())) as sample_custom,
        patch(_VAE_DECODE_PATCH, return_value=image) as vae_decode,
    ):
        result = t2i.run(
            models_dir=tmp_path,
            prompt="A poster that says LOCAL",
            width=1536,
            height=1024,
            steps=32,
            cfg=6.5,
            seed=123,
            mu=0.25,
            std=1.5,
            sampler_name="euler_cfg_pp",
        )

    assert result == [image]
    model_manager.assert_called_once_with(tmp_path)
    mm.load_unet.assert_has_calls(
        [
            call(tmp_path / "diffusion_models" / "ideogram4_fp8_scaled.safetensors"),
            call(
                tmp_path
                / "diffusion_models"
                / "ideogram4_unconditional_fp8_scaled.safetensors"
            ),
        ]
    )
    mm.load_clip.assert_called_once_with(
        tmp_path / "text_encoders" / "qwen3vl_8b_fp8_scaled.safetensors",
        clip_type="ideogram4",
    )
    mm.load_vae.assert_called_once_with(tmp_path / "vae" / "flux2-vae.safetensors")
    encode_prompt.assert_called_once_with(clip, "A poster that says LOCAL")
    zero_out.assert_called_once_with(positive)
    cfg_override.assert_called_once_with(model, 3.0, 0.7, 1.0)
    empty_latent.assert_called_once_with(1536, 1024, batch_size=1)
    random_noise.assert_called_once_with(123)
    get_sampler.assert_called_once_with("euler_cfg_pp")
    scheduler.assert_called_once_with(32, 1536, 1024, 0.25, 1.5)
    dual_guider.assert_called_once_with(
        cfg_overridden_model,
        positive,
        6.5,
        model_negative=model_negative,
        negative=negative,
    )
    sample_custom.assert_called_once_with(noise, guider, sampler, sigmas, latent)
    vae_decode.assert_called_once_with(vae, latent_out)
