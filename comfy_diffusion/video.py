"""Video loading and saving helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _get_video_backend() -> tuple[str, Any]:
    """Return available video backend module (cv2 first, then imageio)."""
    try:
        import cv2

        return "cv2", cv2
    except ModuleNotFoundError:
        pass

    try:
        import imageio.v2 as imageio

        return "imageio", imageio
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Video I/O requires optional dependencies. Install with "
            "`comfy-diffusion[video]` to enable `load_video`, `save_video`, "
            "and `get_video_metadata`."
        ) from error


def _get_torch_module() -> Any | None:
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def _normalize_to_rgb_uint8(frame: Any) -> Any:
    """Normalize a single frame to HWC uint8 RGB."""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3:
        raise ValueError("each frame must be HWC, HW, or PIL image")

    channels = array.shape[2]
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    elif channels != 3:
        raise ValueError("frame channel count must be 1, 3, or 4")

    if np.issubdtype(array.dtype, np.floating):
        max_value = float(array.max(initial=0.0))
        if max_value <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def _coerce_frames_to_rgb_uint8(frames: Any) -> list[Any]:
    """Coerce supported frame inputs into a non-empty list of RGB uint8 arrays."""
    numpy_module: Any | None = None
    try:
        import numpy as np

        numpy_module = np
    except ModuleNotFoundError:
        pass

    if hasattr(frames, "detach") and hasattr(frames, "cpu"):
        tensor = frames.detach().cpu()
        if getattr(tensor, "ndim", None) == 3:
            tensor = tensor.unsqueeze(0)
        if getattr(tensor, "ndim", None) != 4:
            raise ValueError("tensor frames must be BHWC or HWC")

        array = tensor.numpy()
        return [_normalize_to_rgb_uint8(frame) for frame in array]

    if numpy_module is not None and isinstance(frames, numpy_module.ndarray):
        if frames.ndim == 3:
            return [_normalize_to_rgb_uint8(frames)]
        if frames.ndim == 4:
            return [_normalize_to_rgb_uint8(frame) for frame in frames]
        raise ValueError("numpy frames must be BHWC or HWC")

    if not isinstance(frames, Iterable) or isinstance(frames, (str, bytes)):
        raise TypeError("frames must be a BHWC tensor, numpy array, or iterable of frames")

    normalized_frames = [_normalize_to_rgb_uint8(frame) for frame in frames]
    if not normalized_frames:
        raise ValueError("frames is empty")
    return normalized_frames


def _read_frames_cv2(cv2: Any, path: Path) -> list[Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video file: {path}")

    frames: list[Any] = []
    try:
        while True:
            has_frame, frame = capture.read()
            if not has_frame:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return frames


def _read_frames_imageio(imageio: Any, path: Path) -> list[Any]:
    reader = imageio.get_reader(str(path))
    try:
        frames = [_normalize_to_rgb_uint8(frame) for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise ValueError(f"video contains no readable frames: {path}")
    return frames


def _frames_to_output(frames: list[Any]) -> Any:
    """Return BHWC float tensor when torch is available; otherwise PIL image list."""
    torch = _get_torch_module()
    if torch is not None:
        import numpy as np

        stacked = np.stack(frames)
        return torch.from_numpy(stacked).to(torch.float32) / 255.0

    from PIL import Image

    return [Image.fromarray(frame, mode="RGB") for frame in frames]


def load_video(path: str | Path) -> Any:
    """Load a video file into a BHWC tensor, or PIL frames when torch is unavailable."""
    video_path = Path(path)
    backend_name, backend = _get_video_backend()
    if backend_name == "cv2":
        frames = _read_frames_cv2(backend, video_path)
    else:
        frames = _read_frames_imageio(backend, video_path)
    return _frames_to_output(frames)


def save_video(frames: Any, path: str | Path, fps: float) -> None:
    """Save frames to a video file."""
    output_path = Path(path)
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    backend_name, backend = _get_video_backend()
    normalized_frames = _coerce_frames_to_rgb_uint8(frames)

    if backend_name == "cv2":
        _EXTENSION_TO_FOURCC = {".webm": "VP80", ".mp4": "mp4v"}
        fourcc_str = _EXTENSION_TO_FOURCC.get(output_path.suffix.lower(), "mp4v")
        height, width = normalized_frames[0].shape[:2]
        writer = backend.VideoWriter(
            str(output_path),
            backend.VideoWriter_fourcc(*fourcc_str),
            float(fps),
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"unable to open video writer for path: {output_path}")

        try:
            for frame in normalized_frames:
                if frame.shape[:2] != (height, width):
                    raise ValueError("all frames must have identical width and height")
                writer.write(backend.cvtColor(frame, backend.COLOR_RGB2BGR))
        finally:
            writer.release()
        return

    writer = backend.get_writer(str(output_path), fps=float(fps))
    try:
        for frame in normalized_frames:
            writer.append_data(frame)
    finally:
        writer.close()


def save_video_with_audio(
    frames: Any,
    audio: dict[str, Any],
    path: str | Path,
    fps: float,
) -> None:
    """Save frames and a ComfyUI AUDIO dict as one muxed video file.

    This uses the existing video writer for the video stream and the ``ffmpeg``
    executable for final audio muxing. It is intended for generated AV models
    such as MiniMax H3 and LTXV. ``ffmpeg`` must be available on ``PATH``.
    """
    import shutil
    import subprocess
    import tempfile
    import wave

    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("save_video_with_audio requires ffmpeg on PATH")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    waveform = np.asarray(waveform)
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim != 2:
        raise ValueError("audio waveform must have shape [channels, samples]")
    if waveform.shape[0] > 8 and waveform.shape[1] <= 8:
        waveform = waveform.T

    channels, samples = waveform.shape
    waveform = np.clip(waveform, -1.0, 1.0)
    interleaved = (waveform.T * 32767.0).astype(np.int16).tobytes()
    sample_rate = int(audio["sample_rate"])

    with tempfile.TemporaryDirectory(prefix="comfy-diffusion-av-") as tmp:
        tmp_dir = Path(tmp)
        video_only = tmp_dir / "video.mp4"
        audio_only = tmp_dir / "audio.wav"
        save_video(frames, video_only, fps)
        with wave.open(str(audio_only), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(interleaved)

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_only),
                "-i",
                str(audio_only),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )


def get_video_metadata(video_path: str | Path) -> dict[str, int | float]:
    """Return frame count, fps, width, and height for a video."""
    path = Path(video_path)
    backend_name, backend = _get_video_backend()

    if backend_name == "cv2":
        capture = backend.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"unable to open video file: {path}")

        try:
            frame_count = int(capture.get(backend.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(backend.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(backend.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(backend.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()

        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
        }

    reader = backend.get_reader(str(path))
    try:
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps", 0.0))

        size = metadata.get("size")
        if size is None:
            frame0 = _normalize_to_rgb_uint8(reader.get_data(0))
            height, width = frame0.shape[:2]
        else:
            width, height = int(size[0]), int(size[1])

        frame_count = int(metadata.get("nframes", 0) or 0)
        if frame_count <= 0:
            frame_count = reader.count_frames()
    finally:
        reader.close()

    return {
        "frame_count": int(frame_count),
        "fps": fps,
        "width": int(width),
        "height": int(height),
    }


def _get_get_video_components_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_video import GetVideoComponents

    return GetVideoComponents


def get_video_components(video: Any) -> tuple[Any, Any]:
    """Decompose a video into its frame images and audio track.

    Lazily imports and calls ``comfy_extras.nodes_video.GetVideoComponents``.

    Returns ``(images_tensor, audio)`` matching the node's output order.
    """
    get_video_components_type = _get_get_video_components_type()
    result = get_video_components_type.execute(video)
    raw = getattr(result, "result", result)
    return raw[0], raw[1]


def ltxv_img_to_video_inplace(
    vae: Any,
    image: Any,
    latent: dict[str, Any],
    strength: float = 1.0,
    bypass: bool = False,
) -> dict[str, Any]:
    """Inject an image frame into a latent for LTX2/LTX3 image-to-video pipelines.

    Mirrors ``LTXVImgToVideoInplace.execute()`` from ComfyUI.
    """
    if bypass:
        return latent

    import comfy.utils  # type: ignore[import-untyped]
    import torch

    samples = latent["samples"]
    _, height_scale_factor, width_scale_factor = vae.downscale_index_formula

    batch, _, latent_frames, latent_height, latent_width = samples.shape
    width = latent_width * width_scale_factor
    height = latent_height * height_scale_factor

    if image.shape[1] != height or image.shape[2] != width:
        pixels = comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    else:
        pixels = image

    encode_pixels = pixels[:, :, :, :3]
    t = vae.encode(encode_pixels)

    samples[:, :, : t.shape[2]] = t

    conditioning_latent_frames_mask = torch.ones(
        (batch, 1, latent_frames, 1, 1),
        dtype=torch.float32,
        device=samples.device,
    )
    conditioning_latent_frames_mask[:, :, : t.shape[2]] = 1.0 - strength

    return {"samples": samples, "noise_mask": conditioning_latent_frames_mask}


def ltxv_chunk_feed_forward(
    model: Any,
    min_chunk_size: int = 4,
    chunk_threshold: int = 4096,
) -> Any:
    """Return a model clone with chunked feed-forward enabled.

    Mirrors ``LTXVChunkFeedForward.execute()`` from KJNodes (comfyui-kjnodes).
    Splits feed-forward layers into memory-efficient chunks during sampling.
    ``min_chunk_size`` controls the minimum number of tokens per chunk;
    ``chunk_threshold`` is the token-count above which chunking is triggered.

    When ``chunk_threshold == 0`` the model is returned unchanged.
    """
    if chunk_threshold == 0:
        return model

    import torch

    model_clone = model.clone()
    diffusion_model = model_clone.get_model_object("diffusion_model")

    def _make_chunked_ff(orig_forward: Any) -> Any:
        def _chunked(x: Any, **kwargs: Any) -> Any:
            B, T, C = x.shape
            if T <= chunk_threshold:
                return orig_forward(x, **kwargs)
            chunk_size = max(min_chunk_size, (T + chunk_threshold - 1) // chunk_threshold * min_chunk_size)
            chunks = x.split(chunk_size, dim=1)
            out_chunks = [orig_forward(c, **kwargs) for c in chunks]
            return torch.cat(out_chunks, dim=1)

        return _chunked

    for idx, block in enumerate(diffusion_model.transformer_blocks):
        ff = getattr(block, "ff", None)
        if ff is not None:
            orig = ff.forward
            block.ff.forward = _make_chunked_ff(orig)
            model_clone.add_object_patch(
                f"diffusion_model.transformer_blocks.{idx}.ff.forward",
                block.ff.forward,
            )

    return model_clone


def ltx2_nag(
    model: Any,
    nag_scale: float,
    nag_alpha: float,
    nag_tau: float,
    nag_cond_video: Any = None,
    nag_cond_audio: Any = None,
    inplace: bool = True,
) -> Any:
    """Patch a model clone with LTX2 Normalized Attention Guidance (NAG).

    Mirrors ``LTX2_NAG.execute()`` from KJNodes (comfyui-kjnodes).

    When ``nag_scale == 0``, the unmodified model is returned immediately.
    ``nag_cond_video`` and ``nag_cond_audio`` are optional CONDITIONING tensors
    that anchor the negative guidance direction.

    Returns a patched model clone.
    """
    if nag_scale == 0:
        return model

    import types

    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.ldm.modules.attention as comfy_attention
    import comfy.model_management as mm

    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    dtype = model.model.manual_cast_dtype
    if dtype is None:
        dtype = model.model.diffusion_model.dtype

    model_clone = model.clone()
    diffusion_model = model_clone.get_model_object("diffusion_model")
    img_dim = diffusion_model.inner_dim
    audio_dim = diffusion_model.audio_inner_dim

    def _compute_attention(self: Any, query: Any, context: Any, transformer_options: dict = {}) -> Any:  # noqa: B006
        k = self.k_norm(self.to_k(context)).to(query.dtype)
        v = self.to_v(context).to(query.dtype)
        x = comfy_attention.optimized_attention(query, k, v, heads=self.heads, transformer_options=transformer_options).flatten(2)
        del k, v
        return x

    def _normalized_attention_guidance(self: Any, x_positive: Any, x_negative: Any) -> Any:
        if inplace:
            nag_guidance = x_negative.mul_(nag_scale - 1).neg_().add_(x_positive, alpha=nag_scale)
        else:
            nag_guidance = x_positive * nag_scale - x_negative * (nag_scale - 1)
        del x_negative

        norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
        norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)

        scale = norm_guidance / norm_positive
        torch.nan_to_num_(scale, nan=10.0)
        mask = scale > nag_tau
        del scale

        adjustment = (norm_positive * nag_tau) / (norm_guidance + 1e-7)
        del norm_positive, norm_guidance

        nag_guidance.mul_(torch.where(mask, adjustment, 1.0))
        del mask, adjustment

        if inplace:
            nag_guidance.sub_(x_positive).mul_(nag_alpha).add_(x_positive)
        else:
            nag_guidance = nag_guidance * nag_alpha + x_positive * (1 - nag_alpha)
        del x_positive
        return nag_guidance

    def _make_forward(nag_ctx: Any) -> Any:
        def wrapped_attention(self_module: Any, x: Any, context: Any, mask: Any = None, transformer_options: dict = {}, **kwargs: Any) -> Any:  # noqa: B006
            if context.shape[0] == 1:
                q_pos = self_module.q_norm(self_module.to_q(x))
                x_positive = _compute_attention(self_module, q_pos, context, transformer_options)
                x_negative = _compute_attention(self_module, q_pos, nag_ctx, transformer_options)
                del q_pos
                x_out = _normalized_attention_guidance(self_module, x_positive, x_negative)
            else:
                x_pos, x_neg = torch.chunk(x, 2, dim=0)
                context_pos, context_neg = torch.chunk(context, 2, dim=0)
                q_pos = self_module.q_norm(self_module.to_q(x_pos))
                del x_pos
                x_positive = _compute_attention(self_module, q_pos, context_pos, transformer_options)
                x_negative = _compute_attention(self_module, q_pos, nag_ctx, transformer_options)
                del q_pos, context_pos
                x_pos_out = _normalized_attention_guidance(self_module, x_positive, x_negative)

                q_neg = self_module.q_norm(self_module.to_q(x_neg))
                k_neg = self_module.k_norm(self_module.to_k(context_neg))
                v_neg = self_module.to_v(context_neg)
                x_neg_out = comfy_attention.optimized_attention(q_neg, k_neg, v_neg, heads=self_module.heads, transformer_options=transformer_options)
                x_out = torch.cat([x_pos_out, x_neg_out], dim=0)

            return self_module.to_out(x_out)

        return wrapped_attention

    if nag_cond_video is not None:
        diffusion_model.caption_projection.to(device)
        context_video = nag_cond_video[0][0].to(device, dtype)
        v_context, _ = torch.split(context_video, int(context_video.shape[-1] / 2), len(context_video.shape) - 1)
        context_video = diffusion_model.caption_projection(v_context)
        diffusion_model.caption_projection.to(offload_device)
        context_video = context_video.view(1, -1, img_dim)
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            forward_fn = types.MethodType(_make_forward(context_video), block.attn2)
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", forward_fn)

    if nag_cond_audio is not None and diffusion_model.audio_caption_projection is not None:
        diffusion_model.audio_caption_projection.to(device)
        context_audio = nag_cond_audio[0][0].to(device, dtype)
        _, a_context = torch.split(context_audio, int(context_audio.shape[-1] / 2), len(context_audio.shape) - 1)
        context_audio = diffusion_model.audio_caption_projection(a_context)
        diffusion_model.audio_caption_projection.to(offload_device)
        context_audio = context_audio.view(1, -1, audio_dim)
        for idx, block in enumerate(diffusion_model.transformer_blocks):
            forward_fn = types.MethodType(_make_forward(context_audio), block.audio_attn2)
            model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.audio_attn2.forward", forward_fn)

    return model_clone


def ltxv_img_to_video_inplace_kj(
    vae: Any,
    latent: dict[str, Any],
    image: Any,
    index: int = 0,
    strength: float = 1.0,
) -> dict[str, Any]:
    """Inject a single image frame into a latent for LTX2 image-to-video pipelines.

    Mirrors the single-image case of ``LTXVImgToVideoInplaceKJ.execute()`` from
    KJNodes (comfyui-kjnodes).  Unlike ``ltxv_img_to_video_inplace``, this
    variant supports inserting the image at an arbitrary latent frame ``index``
    (in *pixel* space; negative indices count from the end).

    Returns a new latent dict with updated ``samples`` and ``noise_mask``.
    """
    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.utils

    samples = latent["samples"].clone()
    scale_factors = vae.downscale_index_formula
    time_scale_factor, height_scale_factor, width_scale_factor = scale_factors

    batch, _, latent_frames, latent_height, latent_width = samples.shape
    width = latent_width * width_scale_factor
    height = latent_height * height_scale_factor

    if latent.get("noise_mask") is not None:
        conditioning_latent_frames_mask = latent["noise_mask"].clone()
    else:
        conditioning_latent_frames_mask = torch.ones(
            (batch, 1, latent_frames, 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )

    if image.shape[1] != height or image.shape[2] != width:
        pixels = comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    else:
        pixels = image

    encode_pixels = pixels[:, :, :, :3]
    t = vae.encode(encode_pixels)

    # Resolve negative index and convert pixel index to latent index
    pixel_frame_count = (latent_frames - 1) * time_scale_factor + 1
    resolved_index = index if index >= 0 else pixel_frame_count + index
    latent_idx = max(0, min(resolved_index // time_scale_factor, latent_frames - 1))

    end_index = min(latent_idx + t.shape[2], latent_frames)
    samples[:, :, latent_idx:end_index] = t[:, :, : end_index - latent_idx]
    conditioning_latent_frames_mask[:, :, latent_idx:end_index] = 1.0 - strength

    return {"samples": samples, "noise_mask": conditioning_latent_frames_mask}


def ltx2_sampling_preview_override(
    model: Any,
    preview_rate: int = 8,
    latent_upscale_model: Any = None,
    vae: Any = None,
) -> Any:
    """Add an LTX2 sampling preview override wrapper to a model clone.

    Mirrors ``LTX2SamplingPreviewOverride.execute()`` from KJNodes
    (comfyui-kjnodes).  During sampling the preview is generated every
    ``preview_rate`` steps using either a latent RGB approximation or, when a
    ``latent_upscale_model`` / ``vae`` is supplied, a higher-quality decoder.

    Returns a patched model clone.
    """
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.patcher_extension

    model_clone = model.clone()

    taeltx = False
    if vae is not None and getattr(getattr(vae, "first_stage_model", None), "__class__", None) is not None:
        if vae.first_stage_model.__class__.__name__ == "TAEHV":
            taeltx = True
            latent_upscale_model = None

    class _PreviewWrapper:
        def __init__(self) -> None:
            self._latent_upscale_model = latent_upscale_model
            self._vae = vae
            self._preview_rate = preview_rate
            self._taeltx = taeltx

        def __call__(
            self,
            executor: Any,
            noise: Any,
            latent_image: Any,
            sampler: Any,
            sigmas: Any,
            denoise_mask: Any,
            callback: Any,
            disable_pbar: bool,
            seed: Any,
            **kwargs: Any,
        ) -> Any:
            import comfy.latent_preview as latent_preview
            import comfy.utils

            guider = executor.class_obj
            x0_output: dict[str, Any] = {}
            total_steps = sigmas.shape[-1] - 1
            pbar = comfy.utils.ProgressBar(total_steps)
            step_counter = [0]

            previewer = latent_preview.get_previewer(
                guider.model_patcher.load_device,
                guider.model_patcher.model.latent_format,
            )

            original_callback = callback

            def _combined_callback(step: int, x0: Any, x: Any, total: int) -> None:
                x0_output["x0"] = x0
                preview_bytes = None
                if previewer and step_counter[0] % self._preview_rate == 0:
                    preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
                step_counter[0] += 1
                pbar.update_absolute(step_counter[0], total_steps, preview_bytes)
                if original_callback is not None:
                    original_callback(step, x0, x, total)

            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                _combined_callback,
                disable_pbar,
                seed,
                **kwargs,
            )

    model_clone.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "sampling_preview",
        _PreviewWrapper(),
    )
    return model_clone


def create_video(images: Any, audio: Any, fps: float) -> Any:
    """Create a VIDEO object from a batch of images and an audio track.

    Wraps ``CreateVideo.execute()`` from ``comfy_extras.nodes_video``.

    Returns a VIDEO object that can be passed to ``get_video_components`` or
    ``SaveVideo``.
    """
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_video import CreateVideo

    result = CreateVideo.execute(images=images, fps=float(fps), audio=audio)
    raw = getattr(result, "result", result)
    return raw[0]


def _get_void_warped_noise_type() -> Any:
    """Resolve ComfyUI VOIDWarpedNoise node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_void import VOIDWarpedNoise

    return VOIDWarpedNoise


def void_warped_noise(
    optical_flow: Any,
    video: Any,
    width: int = 672,
    height: int = 384,
    length: int = 45,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Generate optical-flow warped noise for VOID pass-2 refinement."""
    node_type = _get_void_warped_noise_type()
    output = node_type.execute(
        optical_flow=optical_flow,
        video=video,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
    )
    result = getattr(output, "result", output)
    return result[0]


def apply_cfg_norm(model: Any, strength: float = 1.0) -> Any:
    """Normalise the CFG prediction to the magnitude of the conditional output.

    Mirrors ``CFGNorm.execute()`` from ``comfy_extras/nodes_cfg.py``.
    Registers a post-CFG sampler hook that scales the denoised prediction so
    its norm matches the conditional denoised signal, clamped to [0, 1].

    Args:
        model: ComfyUI model object.
        strength: Scaling multiplier applied on top of the norm ratio
            (default ``1.0`` keeps the standard magnitude-matching behaviour).

    Returns:
        Cloned model with the CFG-norm post-processing hook installed.
    """
    import torch

    m = model.clone()

    def cfg_norm(args: dict) -> Any:
        cond_p = args["cond_denoised"]
        pred_text = args["denoised"]
        norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
        norm_pred_text = torch.norm(pred_text, dim=1, keepdim=True)
        scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
        return pred_text * scale * strength

    m.set_model_sampler_post_cfg_function(cfg_norm)
    return m


__all__ = [
    "load_video",
    "save_video",
    "get_video_metadata",
    "get_video_components",
    "ltxv_img_to_video_inplace",
    "ltxv_chunk_feed_forward",
    "ltx2_nag",
    "ltxv_img_to_video_inplace_kj",
    "ltx2_sampling_preview_override",
    "create_video",
    "void_warped_noise",
    "apply_cfg_norm",
]
