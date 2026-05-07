"""LTX-Video 2.3 motion-track IC-LoRA pipeline.

This pipeline mirrors the official Lightricks
``LTX-2.3_ICLoRA_Motion_Track_Distilled.json`` workflow. It expects a rendered
motion-track control video: a reference video whose frames contain the colored
spline overlays that describe sparse object or region trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

_HF_REPO_CKPT = "Lightricks/LTX-2.3"
_HF_REPO_TE = "Comfy-Org/ltx-2"
_HF_REPO_IC_LORA = "Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control"

_UNET_DEST = Path("checkpoints") / "ltx-2.3-22b-dev.safetensors"
_TEXT_ENCODER_DEST = Path("text_encoders") / "gemma_3_12B_it.safetensors"
_DISTILLED_LORA_DEST = Path("loras") / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
_IC_LORA_DEST = Path("loras") / "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors"

_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"


def manifest() -> list[ModelEntry]:
    """Return all model files required by the motion-track IC-LoRA pipeline."""
    return [
        HFModelEntry(
            repo_id=_HF_REPO_CKPT,
            filename="ltx-2.3-22b-dev.safetensors",
            dest=_UNET_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_TE,
            filename="split_files/text_encoders/gemma_3_12B_it.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_CKPT,
            filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            dest=_DISTILLED_LORA_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO_IC_LORA,
            filename="ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
            dest=_IC_LORA_DEST,
        ),
    ]


def run(
    *,
    models_dir: str | Path,
    motion_track_video: str | Path | Any,
    prompt: str,
    negative_prompt: str = "pc game, console game, video game, cartoon, childish, ugly",
    width: int = 960,
    height: int = 544,
    length: int = 121,
    fps: int = 24,
    cfg: float = 1.0,
    seed: int = 0,
    distilled_lora_strength: float = 0.5,
    ic_lora_strength: float = 1.0,
    guide_strength: float = 1.0,
    unet_filename: str | None = None,
    vae_filename: str | None = None,
    audio_vae_filename: str | None = None,
    text_encoder_filename: str | None = None,
    distilled_lora_filename: str | None = None,
    ic_lora_filename: str | None = None,
    use_tiled_guide_encode: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 64,
) -> dict[str, Any]:
    """Run LTX-2.3 motion-track controlled video generation.

    Parameters
    ----------
    motion_track_video:
        Control video as a path or a BHWC torch tensor. The frames should show
        the colored trajectory splines expected by the LTX motion-track IC-LoRA.
    """
    from comfy_diffusion.audio import (
        ltxv_audio_vae_decode,
        ltxv_concat_av_latent,
        ltxv_empty_latent_audio,
        ltxv_separate_av_latent,
    )
    from comfy_diffusion.conditioning import (
        encode_prompt,
        ltx_add_video_ic_lora_guide,
        ltxv_conditioning,
        ltxv_crop_guides,
    )
    from comfy_diffusion.latent import ltxv_empty_latent_video
    from comfy_diffusion.lora import apply_ic_lora_model_only, apply_lora
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        cfg_guider,
        get_sampler,
        manual_sigmas,
        random_noise,
        sample_custom,
    )
    from comfy_diffusion.vae import vae_decode_batch_tiled
    from comfy_diffusion.video import load_video

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(
            f"ComfyUI runtime not available: {check_result['error']}"
        )

    models_dir = Path(models_dir)
    mm = ModelManager(models_dir)

    unet_path = Path(unet_filename) if unet_filename else models_dir / _UNET_DEST
    te_path = (
        Path(text_encoder_filename) if text_encoder_filename else models_dir / _TEXT_ENCODER_DEST
    )
    vae_path = Path(vae_filename) if vae_filename else unet_path
    audio_vae_path = Path(audio_vae_filename) if audio_vae_filename else vae_path
    distilled_lora_path = (
        Path(distilled_lora_filename)
        if distilled_lora_filename
        else models_dir / _DISTILLED_LORA_DEST
    )
    ic_lora_path = (
        Path(ic_lora_filename) if ic_lora_filename else models_dir / _IC_LORA_DEST
    )

    ckpt = mm.load_checkpoint_from_path(unet_path)
    model = ckpt.model
    vae = ckpt.vae if vae_filename is None else mm.load_vae(vae_path)
    audio_vae = mm.load_ltxv_audio_vae(audio_vae_path)
    clip = mm.load_ltxav_text_encoder(te_path, unet_path)

    model, _ = apply_lora(model, clip, distilled_lora_path, distilled_lora_strength, 0.0)
    model, latent_downscale_factor = apply_ic_lora_model_only(
        model,
        ic_lora_path,
        ic_lora_strength,
    )

    if isinstance(motion_track_video, (str, Path)):
        guide_video = load_video(motion_track_video)
    else:
        guide_video = motion_track_video

    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative = ltxv_conditioning(positive, negative, frame_rate=fps)

    video_latent = ltxv_empty_latent_video(
        width=width,
        height=height,
        length=length,
        fps=fps,
    )
    positive, negative, video_latent = ltx_add_video_ic_lora_guide(
        positive,
        negative,
        vae,
        video_latent,
        guide_video,
        frame_idx=0,
        strength=guide_strength,
        latent_downscale_factor=latent_downscale_factor,
        crop="disabled",
        use_tiled_encode=use_tiled_guide_encode,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    audio_latent = ltxv_empty_latent_audio(audio_vae, frames_number=length, frame_rate=fps)
    av_latent = ltxv_concat_av_latent(video_latent, audio_latent)

    guider = cfg_guider(model, positive, negative, cfg)
    noise = random_noise(seed)
    sigmas = manual_sigmas(_SIGMAS)
    sampler = get_sampler("euler_ancestral_cfg_pp")
    sampled, _ = sample_custom(noise, guider, sampler, sigmas, av_latent)

    video_latent_out, audio_latent_out = ltxv_separate_av_latent(sampled)
    positive, negative, video_latent_out = ltxv_crop_guides(
        positive,
        negative,
        video_latent_out,
    )
    frames = vae_decode_batch_tiled(vae, video_latent_out)
    audio = ltxv_audio_vae_decode(audio_vae, audio_latent_out)

    return {"frames": frames, "audio": audio}
