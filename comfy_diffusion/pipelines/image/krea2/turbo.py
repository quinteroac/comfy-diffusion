"""Krea2 Turbo local text-to-image pipeline.

This module exposes a direct Python pipeline for the native ComfyUI Krea2
runtime path.  It uses separate diffusion model, text encoder, and VAE files,
loads the Qwen3-VL text encoder with ``clip_type="krea2"``, and applies the
Krea2 conditioning rebalance before sampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

_HF_REPO = "Comfy-Org/Krea-2"

_UNET_DEST = Path("diffusion_models") / "krea2_turbo_fp8_scaled.safetensors"
_CLIP_DEST = Path("text_encoders") / "qwen3vl_4b_fp8_scaled.safetensors"
_VAE_DEST = Path("vae") / "qwen_image_vae.safetensors"

_DEFAULT_WIDTH = 1024
_DEFAULT_HEIGHT = 1024
_DEFAULT_STEPS = 8
_DEFAULT_CFG = 1.0
_DEFAULT_SAMPLER = "euler"
_DEFAULT_SCHEDULER = "simple"
_DEFAULT_DENOISE = 1.0
_DEFAULT_SEED = 0


def manifest() -> list[ModelEntry]:
    """Return the model files required by the Krea2 Turbo pipeline."""
    return [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="diffusion_models/krea2_turbo_fp8_scaled.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
            dest=_CLIP_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="vae/qwen_image_vae.safetensors",
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
    sampler_name: str = _DEFAULT_SAMPLER,
    scheduler: str = _DEFAULT_SCHEDULER,
    denoise: float = _DEFAULT_DENOISE,
    seed: int = _DEFAULT_SEED,
    rebalance_multiplier: float = 4.0,
    rebalance_per_layer_weights: Any = None,
    unet_filename: str | Path | None = None,
    clip_filename: str | Path | None = None,
    vae_filename: str | Path | None = None,
) -> list[Any]:
    """Run local Krea2 Turbo text-to-image generation."""
    from comfy_diffusion.conditioning import (
        KREA2_REBALANCE_DEFAULT_WEIGHTS,
        conditioning_zero_out,
        encode_prompt,
        rebalance_krea2_conditioning,
    )
    from comfy_diffusion.latent import empty_latent_image
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample
    from comfy_diffusion.vae import vae_decode

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_path = models_dir / (unet_filename or _UNET_DEST)
    clip_path = models_dir / (clip_filename or _CLIP_DEST)
    vae_path = models_dir / (vae_filename or _VAE_DEST)

    model = mm.load_unet(unet_path)
    clip = mm.load_clip(clip_path, clip_type="krea2")
    vae = mm.load_vae(vae_path)

    positive = encode_prompt(clip, prompt)
    positive = rebalance_krea2_conditioning(
        positive,
        multiplier=rebalance_multiplier,
        per_layer_weights=(
            KREA2_REBALANCE_DEFAULT_WEIGHTS
            if rebalance_per_layer_weights is None
            else rebalance_per_layer_weights
        ),
    )
    negative = conditioning_zero_out(positive)

    latent = empty_latent_image(width, height, batch_size=1)
    latent_out = sample(
        model,
        positive,
        negative,
        latent,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
        denoise=denoise,
    )
    image = vae_decode(vae, latent_out)
    return [image]
