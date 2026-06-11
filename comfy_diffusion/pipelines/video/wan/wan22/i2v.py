"""WAN 2.2 image-to-video pipeline (14B dual-model, no-LoRA default path).

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the full inference pipeline end-to-end: model loading,
  ``WanImageToVideo`` conditioning, dual two-pass ``KSamplerAdvanced`` with
  ``ModelSamplingSD3`` (shift=5) patched on both models.

This pipeline mirrors the ``video_wan2_2_14B_i2v.json`` reference workflow
with all four ``ComfySwitchNode`` switches at their default value ``False``
(no LoRA branch active).  In that configuration:

- The high-noise UNet is used for **pass 1** (add_noise=True, steps 0→steps//2,
  return_with_leftover_noise=True).
- The low-noise UNet is used for **pass 2** (add_noise=False,
  steps//2→steps, return_with_leftover_noise=False).

Workflow data flow
------------------
1. Load high-noise UNet (no LoRA — switch=False selects direct model)
2. Load low-noise UNet (no LoRA — switch=False selects direct model)
3. Load text encoder (UMT5-XXL) + VAE (WAN 2.1)
4. Encode positive and negative text prompts
5. WanImageToVideo — builds I2V conditioning and initial latent from the
   start image
6. Apply ModelSamplingSD3(shift=5) to both UNets
7. KSamplerAdvanced pass 1 — high-noise model: add_noise=True,
   start_at_step=0, end_at_step=steps//2, return_with_leftover_noise=True
8. KSamplerAdvanced pass 2 — low-noise model: add_noise=False,
   start_at_step=steps//2, end_at_step=steps, return_with_leftover_noise=False
9. VAEDecode → PIL frames

Pattern
-------
Every pipeline module in ``comfy_diffusion/pipelines/`` follows this contract:

1. Define a ``manifest() -> list[ModelEntry]`` function that declares all
   required model files as typed ``ModelEntry`` dataclass instances.
2. Define a ``run(...)`` function that implements the full pipeline using the
   ``comfy_diffusion`` public API (lazy imports only — no top-level
   ``comfy.*`` or ``comfy_diffusion.*`` imports).
3. Keep the manifest as the single source of truth for model file paths —
   ``run()`` should derive its default paths from ``manifest()`` so the two
   stay in sync automatically.

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest, run
    from PIL import Image

    # 1. Download models (idempotent — skips files already present).
    download_models(manifest(), models_dir="/path/to/models")

    # 2. Run inference.
    start_image = Image.open("input.jpg")
    frames = run(
        start_image,
        "a dragon flying through clouds",
        models_dir="/path/to/models",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

# ---------------------------------------------------------------------------
# HuggingFace repository for WAN 2.2
# ---------------------------------------------------------------------------

_HF_REPO_WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

# Relative destination paths (resolved against models_dir by download_models).
_UNET_HIGH_DEST = Path("diffusion_models") / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
_UNET_LOW_DEST = Path("diffusion_models") / "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
_LORA_HIGH_DEST = (
    Path("loras") / "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
)
_LORA_LOW_DEST = (
    Path("loras") / "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
)
_TEXT_ENCODER_DEST = Path("text_encoders") / "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_VAE_DEST = Path("vae") / "wan_2.1_vae.safetensors"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def manifest() -> list[ModelEntry]:
    """Return the list of model files required by the WAN 2.2 I2V pipeline.

    Returns exactly 6 :class:`~comfy_diffusion.downloader.HFModelEntry` items:
    two UNets (high-noise and low-noise), two LoRAs (matching the UNets),
    one text encoder (UMT5-XXL), and one VAE (WAN 2.1).

    All entries are listed including LoRA files — download them so the caller
    can opt into the LoRA branch without re-downloading.  The default
    ``run()`` execution path does not apply LoRAs (switch=False).

    Pass the result directly to :func:`~comfy_diffusion.downloader.download_models`::

        download_models(manifest(), models_dir="/path/to/models")
    """
    return [
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            dest=_UNET_HIGH_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
            dest=_UNET_LOW_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
            dest=_LORA_HIGH_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
            dest=_LORA_LOW_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/vae/wan_2.1_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def run(
    image: Any,
    prompt: str,
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
        "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
    width: int = 640,
    height: int = 640,
    length: int = 81,
    *,
    models_dir: str | Path,
    seed: int = 0,
    steps: int = 20,
    cfg: float = 3.5,
    unet_high_filename: str | None = None,
    unet_low_filename: str | None = None,
    text_encoder_filename: str | None = None,
    vae_filename: str | None = None,
) -> list[Any]:
    """Run the WAN 2.2 image-to-video pipeline end-to-end.

    Dual two-pass ``KSamplerAdvanced`` flow mirroring the
    ``video_wan2_2_14B_i2v.json`` reference workflow with all
    ``ComfySwitchNode`` switches at their default value ``False`` (no LoRA).

    The high-noise UNet handles pass 1 (steps 0→steps//2) and the low-noise
    UNet handles pass 2 (steps//2→steps), each with ``ModelSamplingSD3``
    (shift=5) applied.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image to animate.  Converted to a float32 BHWC tensor
        internally before passing to ``WanImageToVideo`` conditioning.
    prompt : str
        Positive text prompt describing the desired video content.
    negative_prompt : str, optional
        Negative text prompt.  Defaults to the Chinese-language negative prompt
        from the reference workflow.
    width : int, optional
        Output frame width in pixels.  Default ``640``.
    height : int, optional
        Output frame height in pixels.  Default ``640``.
    length : int, optional
        Number of video frames to generate.  Default ``81``.
    models_dir : str | Path
        Root directory where model weights are stored.
    seed : int, optional
        Random seed for reproducibility.  Default ``0``.
    steps : int, optional
        Total number of denoising steps across both passes.  Default ``20``.
    cfg : float, optional
        CFG (classifier-free guidance) scale.  Default ``3.5``.
    unet_high_filename : str | None, optional
        Override the default high-noise UNet filename.
    unet_low_filename : str | None, optional
        Override the default low-noise UNet filename.
    text_encoder_filename : str | None, optional
        Override the default text-encoder filename.
    vae_filename : str | None, optional
        Override the default VAE filename.

    Returns
    -------
    list[PIL.Image.Image]
        Decoded video frames as PIL images, one per frame.
    """
    # Lazy imports — ComfyUI must not be imported at module top level.
    from comfy_diffusion.conditioning import encode_prompt, wan_image_to_video
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.models import ModelManager, model_sampling_sd3
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import sample_advanced
    from comfy_diffusion.vae import vae_decode_batch

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    # Resolve model paths (allow caller overrides, fall back to manifest paths).
    unet_high_path = (
        Path(unet_high_filename) if unet_high_filename else models_dir / _UNET_HIGH_DEST
    )
    unet_low_path = (
        Path(unet_low_filename) if unet_low_filename else models_dir / _UNET_LOW_DEST
    )
    te_path = (
        Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
    )
    vae_path = Path(vae_filename) if vae_filename else models_dir / _VAE_DEST

    # Load models.
    # Switch=False: direct UNet output (no LoRA) → ModelSamplingSD3.
    # Switch116(False): HIGH noise UNet → pass 2 (KSampler85).
    # Switch117(False): LOW noise UNet → pass 1 (KSampler86).
    model_high = mm.load_unet(unet_high_path)
    model_low = mm.load_unet(unet_low_path)
    clip = mm.load_clip(te_path, clip_type="wan")
    vae = mm.load_vae(vae_path)

    # Apply ModelSamplingSD3 patch (shift=5) as in the reference workflow.
    model_high = model_sampling_sd3(model_high, shift=5.0)
    model_low = model_sampling_sd3(model_low, shift=5.0)

    # Text conditioning.
    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    # Convert PIL image to BHWC float32 tensor for WanImageToVideo.
    start_image = image_to_tensor(image)

    # WanImageToVideo — builds I2V conditioning and initial latent.
    positive, negative, latent = wan_image_to_video(
        positive,
        negative,
        vae,
        width=width,
        height=height,
        length=length,
        start_image=start_image,
    )

    # Compute step split: high-noise handles [0, split_step), low-noise handles
    # [split_step, steps).
    split_step = steps // 2

    # Pass 1 — high-noise model: add_noise=True, return_with_leftover_noise=True.
    latent = sample_advanced(
        model_high,
        positive,
        negative,
        latent,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="simple",
        noise_seed=seed,
        add_noise=True,
        start_at_step=0,
        end_at_step=split_step,
        return_with_leftover_noise=True,
    )

    # Pass 2 — low-noise model: add_noise=False, return_with_leftover_noise=False.
    latent = sample_advanced(
        model_low,
        positive,
        negative,
        latent,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="simple",
        noise_seed=seed,
        add_noise=False,
        start_at_step=split_step,
        end_at_step=steps,
        return_with_leftover_noise=False,
    )

    # Decode to frames.
    return vae_decode_batch(vae, latent)
