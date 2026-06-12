"""WAN 2.2 Bernini reference-guided video editing pipeline.

This pipeline adapts the supplied ComfyUI Bernini workflow into the
``comfy-diffusion`` module pattern: ``manifest()`` declares the required model
files and ``run()`` composes the public runtime helpers directly.  It supports
video editing, reference-to-video, and multi-reference conditioning.

Workflow data flow
------------------
1. Load Bernini high-noise and low-noise WAN 2.2 UNets
2. Apply the LightX2V model-only LoRA to both experts
3. Load WAN text encoder and VAE
4. Encode positive and negative prompts
5. Attach source-video and optional reference-image context with Bernini conditioning
6. Sample with ``SamplerCustom`` using split sigmas: high-noise first, low-noise second
7. Decode the latent video to frames
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

_HF_REPO_BERNINI = "Kijai/WanVideo_comfy_fp8_scaled"
_HF_REPO_LIGHTX2V = "Kijai/WanVideo_comfy"
_HF_REPO_WAN22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
_HF_REPO_NSFW_WAN_UMT5 = "NSFW-API/NSFW-Wan-UMT5-XXL"

_UNET_HIGH_DEST = (
    Path("diffusion_models") / "Wan22_Bernini_HIGH_fp8_e4m3fn_scaled.safetensors"
)
_UNET_LOW_DEST = (
    Path("diffusion_models") / "Wan22_Bernini_LOW_fp8_e4m3fn_scaled.safetensors"
)
_LORA_DEST = (
    Path("loras")
    / "wan22"
    / "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
)
_TEXT_ENCODER_DEST = Path("text_encoders") / "nsfw_wan_umt5-xxl_fp8_scaled.safetensors"
_VAE_DEST = Path("vae") / "wan_2.1_vae.safetensors"

_DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
_DEFAULT_PROMPT = (
    "Replace the character for the character in image 0. "
    "Keep camera motion, lighting, and background unchanged."
)


def manifest() -> list[ModelEntry]:
    """Return the model files required by the Bernini WAN 2.2 pipeline."""
    return [
        HFModelEntry(
            repo_id=_HF_REPO_BERNINI,
            filename="Bernini/Wan22_Bernini_HIGH_fp8_e4m3fn_scaled.safetensors",
            dest=_UNET_HIGH_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_BERNINI,
            filename="Bernini/Wan22_Bernini_LOW_fp8_e4m3fn_scaled.safetensors",
            dest=_UNET_LOW_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_LIGHTX2V,
            filename="Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            dest=_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_NSFW_WAN_UMT5,
            filename="nsfw_wan_umt5-xxl_fp8_scaled.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_WAN22,
            filename="split_files/vae/wan_2.1_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]


def _resolve_path(models_dir: Path, override: str | Path | None, default: Path) -> Path:
    if override is None:
        return models_dir / default
    return Path(override)


def _coerce_reference_images(reference_images: Any, image_to_tensor: Any) -> list[Any] | None:
    if reference_images is None:
        return None

    images = (
        list(reference_images)
        if isinstance(reference_images, (list, tuple))
        else [reference_images]
    )
    coerced = [image_to_tensor(image) if hasattr(image, "convert") else image for image in images]
    return coerced or None


def _reference_images_to_autogrow(reference_images: list[Any] | None) -> dict[str, Any] | None:
    if not reference_images:
        return None
    return {f"reference_image_{index}": image for index, image in enumerate(reference_images)}


def _unwrap_bernini_output(output: Any) -> tuple[Any, Any, dict[str, Any]]:
    result = getattr(output, "result", output)
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise RuntimeError("BerniniConditioning returned an unexpected output shape")
    return result[0], result[1], result[2]


def _run_bernini_conditioning(
    positive: Any,
    negative: Any,
    vae: Any,
    *,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    source_video: Any | None,
    reference_images: list[Any] | None,
    ref_max_size: int,
) -> tuple[Any, Any, dict[str, Any]]:
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    try:
        from comfy_extras.nodes_bernini import BerniniConditioning
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Vendored ComfyUI does not include BerniniConditioning; "
            "update the ComfyUI pin to a Bernini-capable nightly build."
        ) from exc

    output = BerniniConditioning.execute(
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        source_video=source_video,
        reference_images=_reference_images_to_autogrow(reference_images),
        ref_max_size=ref_max_size,
    )
    return _unwrap_bernini_output(output)


def run(
    source_video: Any | None = None,
    reference_image: Any | None = None,
    prompt: str = _DEFAULT_PROMPT,
    negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    *,
    models_dir: str | Path,
    seed: int = 3,
    steps: int = 8,
    split_step: int = 4,
    cfg: float = 1.0,
    high_lora_strength: float = 3.0,
    low_lora_strength: float = 1.5,
    sampler_name: str = "res_multistep",
    scheduler: str = "simple",
    ref_max_size: int = 848,
    batch_size: int = 1,
    unet_high_filename: str | Path | None = None,
    unet_low_filename: str | Path | None = None,
    lora_filename: str | Path | None = None,
    text_encoder_filename: str | Path | None = None,
    vae_filename: str | Path | None = None,
) -> list[Any]:
    """Run Bernini reference-guided video editing.

    ``source_video`` may be omitted for reference-to-video generation, a path,
    or a BHWC tensor such as the output from
    :func:`comfy_diffusion.video.load_video`. ``reference_image`` may be
    omitted when a source video is present, a single PIL/BHWC image, or a list
    of PIL/BHWC images for multi-reference conditioning.
    """
    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.lora import apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        basic_scheduler,
        get_sampler,
        sample_custom_simple,
        split_sigmas,
    )
    from comfy_diffusion.vae import vae_decode_batch
    from comfy_diffusion.video import load_video

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    reference_images = _coerce_reference_images(reference_image, image_to_tensor)
    if source_video is None and reference_images is None:
        raise ValueError("source_video or reference_image must be provided")
    if isinstance(source_video, (str, Path)):
        source_video = load_video(source_video)

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_high_path = _resolve_path(models_dir, unet_high_filename, _UNET_HIGH_DEST)
    unet_low_path = _resolve_path(models_dir, unet_low_filename, _UNET_LOW_DEST)
    lora_path = _resolve_path(models_dir, lora_filename, _LORA_DEST)
    text_encoder_path = _resolve_path(models_dir, text_encoder_filename, _TEXT_ENCODER_DEST)
    vae_path = _resolve_path(models_dir, vae_filename, _VAE_DEST)

    model_high = mm.load_unet(unet_high_path)
    model_low = mm.load_unet(unet_low_path)
    clip = mm.load_clip(text_encoder_path, clip_type="wan")
    vae = mm.load_vae(vae_path)

    model_high, _ = apply_lora(model_high, clip, lora_path, high_lora_strength, 0.0)
    model_low, _ = apply_lora(model_low, clip, lora_path, low_lora_strength, 0.0)

    positive, negative = encode_prompt(clip, prompt, negative_prompt)

    positive, negative, latent = _run_bernini_conditioning(
        positive,
        negative,
        vae,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
        source_video=source_video,
        reference_images=reference_images,
        ref_max_size=ref_max_size,
    )

    sampler = get_sampler(sampler_name)
    sigmas = basic_scheduler(model_low, scheduler, steps, denoise=1.0)
    high_sigmas, low_sigmas = split_sigmas(sigmas, split_step)

    latent = sample_custom_simple(
        model_high,
        add_noise=True,
        noise_seed=seed,
        cfg=cfg,
        positive=positive,
        negative=negative,
        sampler=sampler,
        sigmas=high_sigmas,
        latent_image=latent,
    )
    latent = sample_custom_simple(
        model_low,
        add_noise=False,
        noise_seed=0,
        cfg=cfg,
        positive=positive,
        negative=negative,
        sampler=sampler,
        sigmas=low_sigmas,
        latent_image=latent,
    )

    return vae_decode_batch(vae, latent)
