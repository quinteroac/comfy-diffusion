"""VOID video object and interaction deletion pipeline.

Each pipeline module exports ``manifest()`` and ``run()``.

- ``manifest()`` returns a ``list[ModelEntry]`` describing every model file the
  pipeline needs.  Pass it directly to ``download_models()`` to fetch all
  weights before the first inference run.

- ``run()`` executes the two-pass VOID inpainting workflow using ComfyUI's
  native VOID nodes: quadmask preprocessing, CogVideoX inpaint conditioning,
  the VOID DDIM sampler, and optional optical-flow warped noise for pass 2.

Usage
-----
::

    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.void import manifest, run

    download_models(manifest(), models_dir="/path/to/models")
    result = run(
        models_dir="/path/to/models",
        video="/path/to/source.mp4",
        mask="/path/to/quadmask.mp4",
        prompt="the same scene with the selected object removed",
    )
    frames = result["frames"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from comfy_diffusion.downloader import HFModelEntry, ModelEntry

__all__ = ["manifest", "run"]

_HF_REPO = "Comfy-Org/void-model"
_TEXT_ENCODER_REPO = "comfyanonymous/flux_text_encoders"

_PASS1_DEST = Path("diffusion_models") / "void_pass1.safetensors"
_PASS2_DEST = Path("diffusion_models") / "void_pass2.safetensors"
_OPTICAL_FLOW_DEST = (
    Path("optical_flow") / "raft_large_C_T_SKHT_V2-ff5fadd5.safetensors"
)
_TEXT_ENCODER_DEST = Path("text_encoders") / "t5xxl_fp16.safetensors"
_VAE_DEST = Path("vae") / "cogvideox_vae.safetensors"
_SAM3_CHECKPOINT_DEST = Path("checkpoints") / "sam3.1_multiplex_fp16.safetensors"
_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def manifest(*, include_sam3: bool = False) -> list[ModelEntry]:
    """Return the list of model files required by the VOID pipeline.

    The files match ComfyUI's official ``utility_void_video_inpainting``
    workflow:

    - ``diffusion_models/void_pass1.safetensors``
    - ``diffusion_models/void_pass2.safetensors``
    - ``optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.safetensors``
    - ``text_encoders/t5xxl_fp16.safetensors``
    - ``vae/cogvideox_vae.safetensors``

    Set ``include_sam3=True`` to also include the SAM3.1 checkpoint used when
    ``run(..., mask_prompt="...")`` generates the video mask automatically.
    """
    entries: list[ModelEntry] = [
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="diffusion_models/void_pass1.safetensors",
            dest=_PASS1_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="diffusion_models/void_pass2.safetensors",
            dest=_PASS2_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.safetensors",
            dest=_OPTICAL_FLOW_DEST,
        ),
        HFModelEntry(
            repo_id=_TEXT_ENCODER_REPO,
            filename="t5xxl_fp16.safetensors",
            dest=_TEXT_ENCODER_DEST,
        ),
        HFModelEntry(
            repo_id=_HF_REPO,
            filename="vae/cogvideox_vae.safetensors",
            dest=_VAE_DEST,
        ),
    ]
    if include_sam3:
        entries.append(
            HFModelEntry(
                repo_id="Comfy-Org/sam3.1",
                filename="checkpoints/sam3.1_multiplex_fp16.safetensors",
                dest=_SAM3_CHECKPOINT_DEST,
            )
        )
    return entries


def _resolve_path(models_dir: Path, override: str | None, default: Path) -> Path:
    return Path(override) if override else models_dir / default


def _mask_to_tensor(mask: Any) -> Any:
    if hasattr(mask, "convert"):
        import torch

        gray = mask.convert("L")
        width, height = gray.size
        pixels = gray.load()
        if pixels is None:
            raise ValueError("unable to access mask pixels")
        rows = [[pixels[x, y] / 255.0 for x in range(width)] for y in range(height)]
        return torch.tensor([rows], dtype=torch.float32)
    shape = getattr(mask, "shape", None)
    if shape is not None and not callable(shape) and len(shape) == 4:
        return mask[..., 0]
    return mask


def _slice_sequence(sequence: Any, start: int, length: int) -> Any:
    end = start + length
    if isinstance(sequence, list):
        return sequence[start:end]
    shape = getattr(sequence, "shape", None)
    if hasattr(sequence, "__getitem__") and shape is not None and not callable(shape):
        return sequence[start:end]
    if start != 0:
        raise TypeError("start_frame_index requires a sliceable video/mask input")
    return sequence


def _load_mask(mask: Any, *, start_frame_index: int = 0, length: int | None = None) -> Any:
    if isinstance(mask, str | Path):
        path = Path(mask)
        if path.suffix.lower() in _VIDEO_SUFFIXES:
            from comfy_diffusion.video import load_video

            loaded = _mask_to_tensor(load_video(path))
            return _slice_sequence(loaded, start_frame_index, length) if length else loaded

        from PIL import Image

        with Image.open(path) as source:
            return _mask_to_tensor(source.copy())
    loaded = _mask_to_tensor(mask)
    return _slice_sequence(loaded, start_frame_index, length) if length else loaded


def _load_video_input(video: Any) -> Any:
    if isinstance(video, str | Path):
        from comfy_diffusion.video import load_video

        return load_video(video)
    if isinstance(video, list) and video and hasattr(video[0], "convert"):
        from comfy_diffusion.image import images_to_tensor

        return images_to_tensor(video)
    return video


def _resolve_length(
    *,
    requested_length: int,
    duration_seconds: int | float | None,
    fps: float | None,
    video: Any,
) -> int:
    if requested_length < 1:
        raise ValueError("length must be at least 1")
    if duration_seconds is None:
        return requested_length
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than 0")

    resolved_fps = fps
    if resolved_fps is None and isinstance(video, str | Path):
        from comfy_diffusion.video import get_video_metadata

        resolved_fps = float(get_video_metadata(video).get("fps") or 0.0)
    if resolved_fps is None or resolved_fps <= 0:
        raise ValueError(
            "fps must be provided when duration_seconds is used with non-path video input"
        )
    return max(1, int(duration_seconds * resolved_fps))


def _resolve_mask(
    *,
    mm: Any,
    models_root: Path,
    source_video: Any,
    mask: Any | None,
    mask_prompt: str | None,
    sam3_checkpoint_filename: str | None,
    sam3_threshold: float,
    sam3_refine_iterations: int,
    start_frame_index: int,
    length: int,
) -> Any:
    if mask is not None and mask_prompt is not None:
        raise ValueError("pass either mask or mask_prompt, not both")
    if mask is None and mask_prompt is None:
        raise ValueError("either mask or mask_prompt is required")
    if mask is not None:
        return _load_mask(mask, start_frame_index=start_frame_index, length=length)

    from comfy_diffusion.conditioning import encode_prompt
    from comfy_diffusion.segmentation import sam3_detect

    checkpoint_name = sam3_checkpoint_filename or _SAM3_CHECKPOINT_DEST.name
    checkpoint_path = models_root / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint file not found: {checkpoint_path}")

    sam3 = mm.load_checkpoint(checkpoint_name)
    if sam3.clip is None:
        raise RuntimeError("SAM3 checkpoint did not load a CLIP text encoder")

    sam3_conditioning = encode_prompt(sam3.clip, mask_prompt or "")
    sam3_mask, _ = sam3_detect(
        sam3.model,
        source_video,
        conditioning=sam3_conditioning,
        threshold=sam3_threshold,
        refine_iterations=sam3_refine_iterations,
        individual_masks=False,
    )
    return sam3_mask


def run(
    *,
    models_dir: str | Path,
    video: Any,
    prompt: str,
    mask: Any | None = None,
    mask_prompt: str | None = None,
    negative_prompt: str = "",
    width: int = 672,
    height: int = 384,
    length: int = 45,
    steps: int = 30,
    cfg: float = 6.0,
    seed: int = 43,
    scheduler: str = "simple",
    batch_size: int = 1,
    start_frame_index: int = 0,
    duration_seconds: int | float | None = None,
    fps: float | None = None,
    mask_dilate_width: int = 0,
    refine: bool = True,
    pass1_filename: str | None = None,
    pass2_filename: str | None = None,
    optical_flow_filename: str | None = None,
    text_encoder_filename: str | None = None,
    vae_filename: str | None = None,
    sam3_checkpoint_filename: str | None = None,
    sam3_threshold: float = 0.5,
    sam3_refine_iterations: int = 2,
    tile_size: int = 512,
    tile_overlap: int = 64,
) -> dict[str, Any]:
    """Run the VOID object/interactions deletion pipeline.

    Parameters
    ----------
    models_dir : str | Path
        Root directory where model weights are stored.
    video : str | Path | Any
        Source video as a path, BHWC tensor, or list of PIL frames.
    prompt : str
        Positive text prompt.
    mask : str | Path | Any, optional
        VOID quadmask source as a video path, image path, PIL image, BHW mask
        tensor, or BHWC video/image tensor. Pixel values are quantized by
        ``VOIDQuadmaskPreprocess``.
    mask_prompt : str, optional
        SAM3 text prompt used to generate a video mask automatically from the
        input video, e.g. ``"person"``. Requires the SAM3 checkpoint from
        ``manifest(include_sam3=True)``.
    negative_prompt : str, optional
        Negative text prompt.  Default ``""``.
    width, height, length : int, optional
        VOID working resolution and frame count.  Defaults match the vendor
        node defaults.  Lengths that produce odd CogVideoX latent length are
        rounded down internally by ComfyUI.
    start_frame_index : int, optional
        First source frame to process.  Default ``0``.
    duration_seconds : int | float, optional
        Clip duration in seconds.  When set, ``length`` is computed as
        ``duration_seconds * fps``, matching the official workflow.
    fps : float, optional
        Source-video FPS used with ``duration_seconds`` for non-path video
        inputs.  For path inputs, FPS is read from video metadata when omitted.
    steps : int, optional
        Number of scheduler steps for each pass.  Default ``30``.
    cfg : float, optional
        CFG scale.  Default ``6.0``.
    seed : int, optional
        Random seed for pass 1.  Default ``43``.
    scheduler : str, optional
        ComfyUI scheduler name passed to ``BasicScheduler``.  Default ``"simple"``.
    batch_size : int, optional
        Batch size for the latent.  Default ``1``.
    mask_dilate_width : int, optional
        Dilation radius for the primary mask region.  Default ``0``.
    refine : bool, optional
        When ``True``, run pass 2 using optical-flow warped noise.  Default ``True``.
    sam3_threshold : float, optional
        Detection threshold for automatic ``mask_prompt`` masks. Default ``0.5``.
    sam3_refine_iterations : int, optional
        SAM decoder refinement passes for automatic masks. Default ``2``.

    Returns
    -------
    dict[str, Any]
        ``{"frames": list[PIL.Image.Image], "pass1_frames": list[PIL.Image.Image]}``
        when ``refine`` is true, otherwise both keys point at the pass-1 output.
    """
    from comfy_diffusion.conditioning import (
        encode_prompt,
        void_inpaint_conditioning,
        void_quadmask_preprocess,
    )
    from comfy_diffusion.models import ModelManager
    from comfy_diffusion.runtime import check_runtime
    from comfy_diffusion.sampling import (
        basic_scheduler,
        cfg_guider,
        random_noise,
        sample_custom,
        void_sampler,
        void_warped_noise_source,
    )
    from comfy_diffusion.vae import vae_decode_batch
    from comfy_diffusion.video import void_warped_noise

    check_result = check_runtime()
    if check_result.get("error"):
        raise RuntimeError(f"ComfyUI runtime not available: {check_result['error']}")

    models_root = Path(models_dir)
    mm = ModelManager(models_root)
    if start_frame_index < 0:
        raise ValueError("start_frame_index must be non-negative")

    effective_length = _resolve_length(
        requested_length=length,
        duration_seconds=duration_seconds,
        fps=fps,
        video=video,
    )

    pass1_path = _resolve_path(models_root, pass1_filename, _PASS1_DEST)
    pass2_path = _resolve_path(models_root, pass2_filename, _PASS2_DEST)
    optical_flow_path = _resolve_path(
        models_root,
        optical_flow_filename,
        _OPTICAL_FLOW_DEST,
    )
    text_encoder_path = _resolve_path(models_root, text_encoder_filename, _TEXT_ENCODER_DEST)
    vae_path = _resolve_path(models_root, vae_filename, _VAE_DEST)

    source_video = _slice_sequence(
        _load_video_input(video),
        start_frame_index,
        effective_length,
    )
    mask_source = _resolve_mask(
        mm=mm,
        models_root=models_root,
        source_video=source_video,
        mask=mask,
        mask_prompt=mask_prompt,
        sam3_checkpoint_filename=sam3_checkpoint_filename,
        sam3_threshold=sam3_threshold,
        sam3_refine_iterations=sam3_refine_iterations,
        start_frame_index=start_frame_index,
        length=effective_length,
    )
    quadmask = void_quadmask_preprocess(mask_source, dilate_width=mask_dilate_width)

    clip = mm.load_clip(text_encoder_path, clip_type="cogvideox")
    vae = mm.load_vae(vae_path)
    model_pass1 = mm.load_unet(pass1_path)

    positive, negative = encode_prompt(clip, prompt, negative_prompt)
    positive, negative, latent = void_inpaint_conditioning(
        positive=positive,
        negative=negative,
        vae=vae,
        video=source_video,
        quadmask=quadmask,
        width=width,
        height=height,
        length=effective_length,
        batch_size=batch_size,
    )

    sampler = void_sampler()
    sigmas1 = basic_scheduler(model_pass1, scheduler, steps, denoise=1.0)
    guider1 = cfg_guider(model_pass1, positive, negative, cfg)
    sampled1, _ = sample_custom(random_noise(seed), guider1, sampler, sigmas1, latent)
    pass1_frames = vae_decode_batch(vae, sampled1)

    if not refine:
        return {"frames": pass1_frames, "pass1_frames": pass1_frames}

    from comfy_diffusion.image import images_to_tensor

    optical_flow = mm.load_optical_flow(optical_flow_path)
    model_pass2 = mm.load_unet(pass2_path)
    warped_noise = void_warped_noise(
        optical_flow=optical_flow,
        video=images_to_tensor(pass1_frames),
        width=width,
        height=height,
        length=effective_length,
        batch_size=batch_size,
    )
    sigmas2 = basic_scheduler(model_pass2, scheduler, steps, denoise=1.0)
    guider2 = cfg_guider(model_pass2, positive, negative, cfg)
    sampled2, _ = sample_custom(
        void_warped_noise_source(warped_noise),
        guider2,
        sampler,
        sigmas2,
        latent,
    )
    frames = vae_decode_batch(vae, sampled2)
    return {"frames": frames, "pass1_frames": pass1_frames}
