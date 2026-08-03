"""MiniMax H3 reference-to-video pipeline.

This is the library equivalent of
``comfyui_official_workflows/video/minimax/video_minimax_h3_r2v.json``.
MiniMax H3 generates a video and a synchronized audio track from a prompt and
one or more reference images.  Reference images are addressed in the prompt as
``<Picture 1>``, ``<Picture 2>``, etc.

The reference workflow uses two images, a 1344x768 canvas, 124 frames at 24
fps, the ``res_multistep`` sampler, and a 20-step ``simple`` schedule.  The
defaults below mirror those settings while allowing callers to supply their
own prompt and images.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import ModelEntry, URLModelEntry

__all__ = ["manifest", "run"]

_UNET_DEST = Path("diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
_CLIP_DEST = Path("text_encoders/minimax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
_AUDIO_VAE_DEST = Path("vae/minimax/minimax_h3_audio_vae_fp32.safetensors")
_VIDEO_VAE_DEST = Path("vae/minimax/minimax_h3_video_vae_fp16.safetensors")

_HF_BASE = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"


def manifest() -> list[ModelEntry]:
    """Return the four model files used by the official H3 R2V workflow.

    Destinations mirror ComfyUI's model layout, with each model family inside
    its own ``minimax/`` directory (for example
    ``/mnt/models/comfyui/vae/minimax``).
    """
    return [
        URLModelEntry(
            url=f"{_HF_BASE}/diffusion_models/{_UNET_DEST.name}",
            dest=_UNET_DEST,
        ),
        URLModelEntry(
            url=f"{_HF_BASE}/text_encoders/{_CLIP_DEST.name}",
            dest=_CLIP_DEST,
        ),
        URLModelEntry(
            url=f"{_HF_BASE}/vae/{_AUDIO_VAE_DEST.name}",
            dest=_AUDIO_VAE_DEST,
        ),
        URLModelEntry(
            url=f"{_HF_BASE}/vae/{_VIDEO_VAE_DEST.name}",
            dest=_VIDEO_VAE_DEST,
        ),
    ]


def _model_path(models_dir: Path, override: str | Path | None, default: Path) -> Path:
    """Resolve an explicit path or the default path under ``models_dir``."""
    return Path(override) if override is not None else models_dir / default


def run(
    prompt: str,
    reference_images: Sequence[Any],
    *,
    models_dir: str | Path,
    width: int = 1344,
    height: int = 768,
    length: int = 124,
    steps: int = 20,
    seed: int = 0,
    ref_image_size: str = "match",
    unet_filename: str | Path | None = None,
    text_encoder_filename: str | Path | None = None,
    audio_vae_filename: str | Path | None = None,
    video_vae_filename: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate MiniMax H3 video and synchronized audio.

    Parameters
    ----------
    prompt:
        Prompt containing ``<Picture i>`` tags for the supplied references.
    reference_images:
        Non-empty sequence of ComfyUI BHWC image tensors.  PIL images can be
        converted with :func:`comfy_diffusion.image.image_to_tensor`.
    models_dir:
        Model root.  With the supplied weights this is ``/mnt/models/comfyui``.
    width, height, length:
        Output dimensions and frame count. H3 snaps the frame count to its
        ``17k + 5`` grid; the default is 124 frames at 24 fps (~5 seconds).

    Returns ``{"frames": list[PIL.Image.Image], "audio": {"waveform": ...,
    "sample_rate": 32000}}``. When ``output_path`` is supplied, the returned
    dictionary also contains ``video_path`` and the file already includes the
    generated audio track.
    """
    if not reference_images:
        raise ValueError("reference_images must contain at least one image")
    if ref_image_size not in {"match", "max"}:
        raise ValueError("ref_image_size must be 'match' or 'max'")

    # All ComfyUI and torch imports remain deferred until inference time.
    from comfy_diffusion._runtime import ensure_comfyui_on_path
    from comfy_diffusion.audio import vae_decode_audio
    from comfy_diffusion.image import image_to_tensor
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        basic_guider,
        basic_scheduler,
        get_sampler,
        random_noise,
        sample_custom,
    )
    from comfy_diffusion.vae import vae_decode_batch
    from comfy_diffusion.video import save_video_with_audio

    runtime = check_runtime()
    if runtime.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {runtime['error']}")

    models_root = Path(models_dir)
    manager = ModelManager(models_root)
    model = manager.load_unet(_model_path(models_root, unet_filename, _UNET_DEST))
    clip = manager.load_clip(
        _model_path(models_root, text_encoder_filename, _CLIP_DEST),
        clip_type="minimax",
    )
    video_vae = manager.load_vae(
        _model_path(models_root, video_vae_filename, _VIDEO_VAE_DEST)
    )
    audio_vae = manager.load_vae(
        _model_path(models_root, audio_vae_filename, _AUDIO_VAE_DEST)
    )

    # MiniMaxH3ReferenceToVideo performs reference resizing, VAE encoding, and
    # Qwen3-VL multimodal token construction exactly as the official node does.
    ensure_comfyui_on_path()
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

    ref_images = {
        f"ref_image_{index}": (
            image if hasattr(image, "shape") else image_to_tensor(image)
        )
        for index, image in enumerate(reference_images)
    }
    conditioning = MiniMaxH3ReferenceToVideo.execute(
        clip=clip,
        vae=video_vae,
        audio_vae=audio_vae,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        ref_image_size=ref_image_size,
        ref_images=ref_images,
    )
    result = getattr(conditioning, "result", conditioning)
    positive, latent = result[0], result[1]

    # The official workflow uses BasicGuider (no separate negative prompt).
    guider = basic_guider(model, positive)
    noise = random_noise(seed)
    sampler = get_sampler("res_multistep")
    sigmas = basic_scheduler(model, "simple", steps)
    _, sampled = sample_custom(noise, guider, sampler, sigmas, latent)

    audio_samples = sampled["samples"].unbind()[-1]
    audio = vae_decode_audio(audio_vae, {"samples": audio_samples})
    # Normalize the result to ComfyUI's [B, channels, samples] convention.
    # Depending on the VAE wrapper version, the shared helper can return the
    # compatible [B, samples, channels] layout instead.
    if audio.ndim == 3 and audio.shape[1] > 8 and audio.shape[-1] <= 8:
        audio = audio.movedim(-1, 1)
    frames = vae_decode_batch(video_vae, {"samples": sampled["samples"]})
    result = {
        "frames": frames,
        "audio": {
            "waveform": audio,
            "sample_rate": int(getattr(audio_vae, "audio_sample_rate", 32000)),
        },
    }
    if output_path is not None:
        save_video_with_audio(frames, result["audio"], output_path, fps=24)
        result["video_path"] = str(Path(output_path))
    return result
