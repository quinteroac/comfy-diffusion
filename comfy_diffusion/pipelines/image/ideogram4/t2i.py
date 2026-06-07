"""Ideogram 4 local text-to-image pipeline.

This module exposes a direct Python pipeline for the local Ideogram 4 ComfyUI
runtime path.  It uses the local model, text encoder, scheduler, asymmetric CFG
guider, and Flux2 VAE from vendored ComfyUI; it does not call Ideogram hosted
APIs or ComfyUI API nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

_HF_REPO_IDEOGRAM4 = "Comfy-Org/Ideogram-4"

_UNET_DEST = Path("diffusion_models") / "ideogram4_fp8_scaled.safetensors"
_UNCOND_UNET_DEST = (
    Path("diffusion_models") / "ideogram4_unconditional_fp8_scaled.safetensors"
)
_CLIP_DEST = Path("text_encoders") / "qwen3vl_8b_fp8_scaled.safetensors"
_VAE_DEST = Path("vae") / "flux2-vae.safetensors"

_DEFAULT_WIDTH = 1024
_DEFAULT_HEIGHT = 1024
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 7.0
_DEFAULT_CFG_OVERRIDE = 3.0
_DEFAULT_CFG_OVERRIDE_START = 0.7
_DEFAULT_CFG_OVERRIDE_END = 1.0
_DEFAULT_SEED = 0
_DEFAULT_MU = 0.0
_DEFAULT_STD = 1.75
_DEFAULT_SAMPLER = "euler"


def manifest() -> list[ModelEntry]:
    """Return the local FP8 Ideogram 4 model files required by this pipeline."""
    return [
        HFModelEntry(
            repo_id=_HF_REPO_IDEOGRAM4,
            filename="diffusion_models/ideogram4_fp8_scaled.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_IDEOGRAM4,
            filename="diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
            dest=_UNCOND_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_IDEOGRAM4,
            filename="text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_IDEOGRAM4,
            filename="vae/flux2-vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    prompt: str,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    steps: int = _DEFAULT_STEPS,
    cfg: float = _DEFAULT_CFG,
    cfg_override_value: float | None = _DEFAULT_CFG_OVERRIDE,
    cfg_override_start: float = _DEFAULT_CFG_OVERRIDE_START,
    cfg_override_end: float = _DEFAULT_CFG_OVERRIDE_END,
    seed: int = _DEFAULT_SEED,
    mu: float = _DEFAULT_MU,
    std: float = _DEFAULT_STD,
    sampler_name: str = _DEFAULT_SAMPLER,
    unet_filename: str | Path | None = None,
    uncond_unet_filename: str | Path | None = None,
    clip_filename: str | Path | None = None,
    vae_filename: str | Path | None = None,
) -> list[Any]:
    """Run local Ideogram 4 text-to-image generation.

    ``prompt`` is passed directly to the Ideogram 4 text encoder.  Callers may
    provide plain text or a pre-structured JSON string; prompt upsampling is not
    performed by this pipeline.
    """
    from comfy_diffusion.conditioning import conditioning_zero_out, encode_prompt
    from comfy_diffusion.latent import empty_flux2_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        cfg_override,
        dual_model_guider,
        get_sampler,
        ideogram4_scheduler,
        random_noise,
        sample_custom,
    )
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_path = models_dir / (unet_filename or _UNET_DEST)
    uncond_unet_path = models_dir / (uncond_unet_filename or _UNCOND_UNET_DEST)
    clip_path = models_dir / (clip_filename or _CLIP_DEST)
    vae_path = models_dir / (vae_filename or _VAE_DEST)

    model = mm.load_unet(unet_path)
    model_negative = mm.load_unet(uncond_unet_path)
    clip = mm.load_clip(clip_path, clip_type="ideogram4")
    vae = mm.load_vae(vae_path)

    positive = encode_prompt(clip, prompt)
    negative = conditioning_zero_out(positive)
    if cfg_override_value is not None:
        model = cfg_override(
            model,
            cfg_override_value,
            cfg_override_start,
            cfg_override_end,
        )

    latent = empty_flux2_latent_image(width, height, batch_size=1)
    noise = random_noise(seed)
    sampler = get_sampler(sampler_name)
    sigmas = ideogram4_scheduler(steps, width, height, mu, std)
    guider = dual_model_guider(
        model,
        positive,
        cfg,
        model_negative=model_negative,
        negative=negative,
    )

    latent_out, _ = sample_custom(noise, guider, sampler, sigmas, latent)
    image = vae_decode(vae, latent_out)
    return [image]
