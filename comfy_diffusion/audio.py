"""Audio helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Protocol, cast


class _LtxvAudioVaeEncoder(Protocol):
    sample_rate: int

    def encode(self, audio: Any) -> Any: ...


class _LtxvAudioVaeDecoder(Protocol):
    output_sample_rate: int

    def decode(self, latent: Any) -> Any: ...


class _LtxvAudioVae(Protocol):
    sample_rate: int
    latent_channels: int
    latent_frequency_bins: int

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int: ...


class _AceStep15Clip(Protocol):
    def tokenize(
        self,
        tags: str,
        *,
        lyrics: str,
        bpm: int,
        duration: float,
        timesignature: int,
        language: str,
        keyscale: str,
        seed: int,
        generate_audio_codes: bool,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
    ) -> Any: ...

    def encode_from_tokens_scheduled(self, tokens: Any) -> Any: ...


def _get_ltxv_empty_latent_audio_type() -> Any:
    """Resolve ComfyUI LTXVEmptyLatentAudio node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio

    return LTXVEmptyLatentAudio


def _get_ace_step_15_latent_audio_dependencies() -> tuple[Any, Any]:
    """Resolve torch and ComfyUI model management at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import torch

    return torch, comfy.model_management


def _get_concat_av_latent_dependencies() -> tuple[Any, Any]:
    """Resolve torch and comfy.nested_tensor at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.nested_tensor
    import torch

    return torch, comfy.nested_tensor


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def _inference_mode_context() -> Any:
    """Return torch.no_grad() when torch is available, else a no-op context.

    Uses no_grad rather than inference_mode to avoid creating inference tensors,
    which break ComfyUI's model patcher when it tries to register parameters
    during device moves (e.g. freeing GPU memory between pipeline stages).
    """
    try:
        import torch
    except ModuleNotFoundError:
        return nullcontext()
    return torch.no_grad()


def ltxv_audio_vae_encode(vae: _LtxvAudioVaeEncoder, audio: Any) -> dict[str, Any]:
    """Encode raw audio with an LTXV audio VAE."""
    with _inference_mode_context():
        audio_latents = vae.encode(audio)
        return {"samples": audio_latents, "sample_rate": int(vae.sample_rate), "type": "audio"}


def ltxv_audio_vae_decode(vae: _LtxvAudioVaeDecoder, latent: Any) -> dict[str, Any]:
    """Decode latent audio with an LTXV audio VAE.

    Decodes on CPU to avoid VRAM contention with the video UNet/VAE.
    """
    latent_tensor = latent["samples"] if isinstance(latent, dict) else latent
    if getattr(latent_tensor, "is_nested", False):
        latent_tensor = latent_tensor.unbind()[-1]

    first_stage_model = getattr(vae, "first_stage_model", None)
    if first_stage_model is not None:
        if hasattr(vae, "patcher"):
            from ._runtime import ensure_comfyui_on_path

            ensure_comfyui_on_path()
            import comfy.model_management

            memory_required = 0
            if hasattr(vae, "memory_used_decode") and hasattr(latent_tensor, "shape"):
                memory_required = vae.memory_used_decode(latent_tensor.shape, vae.vae_dtype)
            comfy.model_management.load_models_gpu(
                [vae.patcher],
                memory_required=memory_required,
                force_full_load=getattr(vae, "disable_offload", False),
            )
        if hasattr(latent_tensor, "to") and hasattr(vae, "device") and hasattr(vae, "vae_dtype"):
            latent_tensor = latent_tensor.to(device=vae.device, dtype=vae.vae_dtype)
        with _inference_mode_context():
            audio = first_stage_model.decode(latent_tensor)
        if hasattr(audio, "to"):
            output_device = getattr(vae, "output_device", getattr(latent_tensor, "device", None))
            if hasattr(vae, "vae_output_dtype"):
                audio = audio.to(device=output_device, dtype=vae.vae_output_dtype(), copy=True)
            elif output_device is not None:
                audio = audio.to(output_device)
        return {
            "waveform": audio,
            "sample_rate": int(first_stage_model.output_sample_rate),
        }

    if hasattr(vae, "to"):
        vae.to("cpu")
    if hasattr(latent_tensor, "cpu"):
        latent_tensor = latent_tensor.cpu()

    with _inference_mode_context():
        audio = vae.decode(latent_tensor)
    if hasattr(audio, "detach"):
        audio = audio.detach()
    return {"waveform": audio, "sample_rate": int(vae.output_sample_rate)}


def ltxv_empty_latent_audio(
    audio_vae: _LtxvAudioVae,
    frames_number: int,
    frame_rate: int = 25,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Create empty LTXV audio latents compatible with ComfyUI's audio pipeline."""
    ltxv_empty_latent_audio_type = _get_ltxv_empty_latent_audio_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            ltxv_empty_latent_audio_type.execute(
                frames_number=frames_number,
                frame_rate=frame_rate,
                batch_size=batch_size,
                audio_vae=audio_vae,
            )
        )
    )


def encode_ace_step_15_audio(
    clip: _AceStep15Clip,
    tags: str,
    lyrics: str = "",
    seed: int = 0,
    bpm: int = 120,
    duration: float = 120.0,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    generate_audio_codes: bool = True,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
) -> Any:
    """Encode ACE Step 1.5 text/audio metadata conditioning."""
    tokens = clip.tokenize(
        tags,
        lyrics=lyrics,
        bpm=bpm,
        duration=duration,
        timesignature=int(timesignature),
        language=language,
        keyscale=keyscale,
        seed=seed,
        generate_audio_codes=generate_audio_codes,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    return clip.encode_from_tokens_scheduled(tokens)


def ltxv_concat_av_latent(
    video_latent: dict[str, Any],
    audio_latent: dict[str, Any],
) -> dict[str, Any]:
    """Concatenate video and audio latents into a single NestedTensor latent for joint denoising."""
    torch, comfy_nested_tensor = _get_concat_av_latent_dependencies()

    output: dict[str, Any] = {}
    output.update(video_latent)
    output.update(audio_latent)

    video_noise_mask = video_latent.get("noise_mask", None)
    audio_noise_mask = audio_latent.get("noise_mask", None)

    if video_noise_mask is not None or audio_noise_mask is not None:
        if video_noise_mask is None:
            video_noise_mask = torch.ones_like(video_latent["samples"])
        if audio_noise_mask is None:
            audio_noise_mask = torch.ones_like(audio_latent["samples"])
        output["noise_mask"] = comfy_nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask))

    output["samples"] = comfy_nested_tensor.NestedTensor((video_latent["samples"], audio_latent["samples"]))
    return output


def ltxv_separate_av_latent(
    av_latent: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate a joint AV NestedTensor latent into individual video and audio latents."""
    video_samples, audio_samples = av_latent["samples"].unbind()

    video_latent: dict[str, Any] = {"samples": video_samples}
    audio_latent: dict[str, Any] = {"samples": audio_samples}

    if "noise_mask" in av_latent:
        video_mask, audio_mask = av_latent["noise_mask"].unbind()
        video_latent["noise_mask"] = video_mask
        audio_latent["noise_mask"] = audio_mask

    return video_latent, audio_latent


def vae_decode_audio(vae: Any, latent: dict[str, Any]) -> Any:
    """Decode ACE Step (or compatible) audio latents into a waveform tensor.

    Calls ``vae.decode(latent["samples"])`` and moves the channel dimension
    from the last position to position 1, matching ComfyUI's AUDIO waveform
    convention (shape ``[B, C, N]``).

    Args:
        vae: Audio VAE with a ``decode`` method.
        latent: Latent dict with key ``"samples"``.

    Returns:
        Waveform tensor with shape ``[B, C, N]``.
    """
    with _inference_mode_context():
        return vae.decode(latent["samples"]).movedim(-1, 1)


def empty_ace_step_15_latent_audio(seconds: float, batch_size: int = 1) -> dict[str, Any]:
    """Create empty ACE Step 1.5 latents used as sampler noise input."""
    torch, model_management = _get_ace_step_15_latent_audio_dependencies()
    length = round(seconds * 48000 / 1920)
    latent = torch.zeros([batch_size, 64, length], device=model_management.intermediate_device())
    return {"samples": latent, "type": "audio"}


def _get_trim_audio_duration_node() -> Any:
    """Resolve ComfyUI TrimAudioDuration node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_audio import TrimAudioDuration

    return TrimAudioDuration


def audio_crop(
    audio: dict[str, Any],
    start_time: float,
    end_time: float,
) -> dict[str, Any]:
    """Crop audio to the time range [start_time, end_time] in seconds.

    Args:
        audio: ComfyUI AUDIO dict with keys ``"waveform"`` and ``"sample_rate"``.
        start_time: Start of the crop window in seconds (>=0).
        end_time: End of the crop window in seconds (>start_time).

    Returns:
        ComfyUI AUDIO dict containing the cropped waveform.

    Raises:
        ValueError: If start_time >= end_time.
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    audio_length = waveform.shape[-1]

    start_frame = max(0, int(round(start_time * sample_rate)))
    end_frame = min(audio_length, int(round(end_time * sample_rate)))

    if start_frame >= end_frame:
        raise ValueError(
            f"audio_crop: start_time ({start_time}) must be less than end_time ({end_time})."
        )

    return {"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate}


def audio_separation(
    audio: dict[str, Any],
    mode: str = "harmonic",
    fft_n: int = 2048,
    win_length: int | None = None,
) -> dict[str, Any]:
    """Separate audio into its harmonic or percussive component using STFT-based HPSS.

    Uses a Wiener filter built from median-filtered power spectrograms.

    Args:
        audio: ComfyUI AUDIO dict with keys ``"waveform"`` and ``"sample_rate"``.
        mode: ``"harmonic"`` or ``"percussive"``.
        fft_n: FFT window size (n_fft).
        win_length: Analysis window length; defaults to ``fft_n``.

    Returns:
        ComfyUI AUDIO dict containing the separated component.

    Raises:
        ValueError: If mode is not ``"harmonic"`` or ``"percussive"``.
    """
    import torch

    if mode not in ("harmonic", "percussive"):
        raise ValueError(
            f"audio_separation: mode must be 'harmonic' or 'percussive', got {mode!r}."
        )

    waveform = audio["waveform"]  # [B, C, N]
    sample_rate = audio["sample_rate"]
    _win_length = win_length if win_length is not None else fft_n
    hop_length = _win_length // 4

    B, C, N = waveform.shape
    results = []
    for b in range(B):
        channels = []
        for c in range(C):
            signal = waveform[b, c]  # [N]
            window = torch.hann_window(_win_length, device=signal.device)
            spec = torch.stft(
                signal,
                n_fft=fft_n,
                hop_length=hop_length,
                win_length=_win_length,
                window=window,
                return_complex=True,
            )  # [F, T]
            power = spec.abs() ** 2  # [F, T]

            k_h = 31  # harmonic: smooth across time frames
            k_v = 31  # percussive: smooth across frequency bins

            # Harmonic component via time-axis median filter
            pad_h = k_h // 2
            padded_h = torch.nn.functional.pad(
                power.unsqueeze(0).unsqueeze(0), (pad_h, pad_h, 0, 0), mode="reflect"
            )
            harmonic = padded_h.unfold(-1, k_h, 1).median(-1).values.squeeze(0).squeeze(0)

            # Percussive component via frequency-axis median filter
            pad_v = k_v // 2
            padded_v = torch.nn.functional.pad(
                power.unsqueeze(0).unsqueeze(0), (0, 0, pad_v, pad_v), mode="reflect"
            )
            percussive = padded_v.unfold(-2, k_v, 1).median(-1).values.squeeze(0).squeeze(0)

            total = harmonic + percussive + 1e-10
            mask = harmonic / total if mode == "harmonic" else percussive / total

            spec_separated = spec * mask
            signal_out = torch.istft(
                spec_separated,
                n_fft=fft_n,
                hop_length=hop_length,
                win_length=_win_length,
                window=window,
                length=N,
            )
            channels.append(signal_out)
        results.append(torch.stack(channels))
    out_waveform = torch.stack(results)  # [B, C, N]
    return {"waveform": out_waveform, "sample_rate": sample_rate}


def trim_audio_duration(
    audio: dict[str, Any],
    start: float,
    duration: float,
) -> dict[str, Any]:
    """Trim an audio dict to a specific duration starting at a given time.

    Wraps ComfyUI's ``TrimAudioDuration`` node.

    Args:
        audio: ComfyUI AUDIO dict with keys ``"waveform"`` and ``"sample_rate"``.
        start: Start time in seconds. Negative values count from the end of the audio.
        duration: Duration in seconds to keep.

    Returns:
        ComfyUI AUDIO dict containing the trimmed waveform.
    """
    trim_node = _get_trim_audio_duration_node()
    return cast(
        dict[str, Any],
        _unwrap_node_output(trim_node.execute(audio=audio, start_index=start, duration=duration)),
    )


def _load_audio_av(path: str, start_time: float, duration: "float | None") -> "tuple":
    """Load audio via PyAV — used when torchaudio is unavailable or broken."""
    import numpy as np
    import torch
    import av

    container = av.open(path)
    stream = container.streams.audio[0]
    sample_rate: int = stream.sample_rate
    channels: int = stream.channels

    chunks = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray()  # packed: [1, samples*channels] or planar: [channels, samples]
        # Normalise packed interleaved formats to [channels, samples].
        if arr.shape[0] == 1 and channels > 1:
            arr = arr.reshape(channels, -1, order="F")  # interleaved → planar
        chunks.append(arr)
    container.close()

    waveform = np.concatenate(chunks, axis=1)  # [C, N]

    # Convert integer PCM to float32 [-1, 1].
    if waveform.dtype == np.int16:
        waveform = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform = waveform.astype(np.float32) / 2147483648.0
    else:
        waveform = waveform.astype(np.float32)

    if start_time > 0.0 or duration is not None:
        start_s = int(start_time * sample_rate)
        waveform = waveform[:, start_s:]
        if duration is not None:
            waveform = waveform[:, : int(duration * sample_rate)]

    return torch.from_numpy(waveform.copy()), sample_rate


def load_audio(
    path: str | "Path",
    start_time: float = 0.0,
    duration: float | None = None,
) -> dict[str, Any]:
    """Load an audio file from disk into the ComfyUI AUDIO dict format.

    Args:
        path: Path to the audio file.
        start_time: Start offset in seconds (default 0.0).
        duration: Duration in seconds to load, or ``None`` to load until end of file.

    Returns:
        A dict with keys ``"waveform"`` (torch.Tensor, shape ``[1, C, N]``) and
        ``"sample_rate"`` (int), compatible with ``ltxv_audio_vae_encode``.
    """
    import math
    from pathlib import Path as _Path

    audio_path = str(_Path(path))

    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        if start_time > 0.0 or duration is not None:
            start_frame = max(0, math.floor(start_time * sample_rate))
            if duration is not None:
                end_frame = start_frame + max(0, math.ceil(duration * sample_rate))
                waveform = waveform[:, start_frame:end_frame]
            else:
                waveform = waveform[:, start_frame:]
    except Exception:
        # torchaudio >=2.9 may require TorchCodec; fall back to PyAV.
        waveform, sample_rate = _load_audio_av(audio_path, start_time, duration)

    # Shape: [C, N] → [1, C, N]
    waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def audio_encoder_encode(audio_encoder: Any, audio: dict) -> Any:
    """Encode an audio dict into an AUDIO_ENCODER_OUTPUT for S2V conditioning.

    Args:
        audio_encoder: A loaded audio encoder (e.g. from ``ModelManager.load_audio_encoder``).
        audio: ComfyUI AUDIO dict with keys ``"waveform"`` and ``"sample_rate"``.

    Returns:
        An ``AUDIO_ENCODER_OUTPUT`` object suitable for S2V conditioning.
    """
    return audio_encoder.encode_audio(audio["waveform"], audio["sample_rate"])


def ltxv_audio_video_mask(
    video_latent: dict[str, Any],
    audio_latent: dict[str, Any],
    video_fps: float,
    video_end_time: float,
    audio_start_time: float,
    audio_end_time: float,
    num_video_frames_to_guide: int = 10,
    audio_overlap_latents: int = 10,
    audio_overlap: int = 10,
    pad_mode: str = "pad",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set noise masks on video and audio latents for AV extension sampling.

    Mirrors ``LTXVAudioVideoMask.execute()`` from KJNodes (comfyui-kjnodes).
    Applied before ``ltxv_concat_av_latent`` in video extension passes to
    preserve overlap frames from the previous generation.

    The first ``num_video_frames_to_guide`` pixel-frames of the video latent
    and the first ``audio_overlap_latents`` audio latents are masked out
    (``noise_mask = 0.0``), so the sampler treats them as fixed context.

    Args:
        video_latent: Video LATENT dict (``{"samples": tensor, ...}``).
        audio_latent: Audio LATENT dict (``{"samples": tensor, ...}``).
        video_fps: Frames per second of the video.
        video_end_time: End time (seconds) of the current video segment.
        audio_start_time: Start time (seconds) of the audio window.
        audio_end_time: End time (seconds) of the audio window.
        num_video_frames_to_guide: Pixel-frame count of the video overlap region.
        audio_overlap_latents: Number of audio latents in the overlap region.
        audio_overlap: Pixel-frame count of the audio overlap region (unused by
            default; provided for API parity with the node).
        pad_mode: Padding mode (``"pad"`` or ``"no_pad"``).  Currently unused.

    Returns:
        ``(video_latent_masked, audio_latent_masked)`` — copies of the input
        latents with ``noise_mask`` set.
    """
    import torch

    video_samples = video_latent["samples"]

    # Video noise mask: shape [B, 1, T, H, W]
    batch, _ch, latent_t, latent_h, latent_w = video_samples.shape
    video_mask = torch.ones(
        (batch, 1, latent_t, latent_h, latent_w),
        dtype=torch.float32,
        device=video_samples.device,
    )
    # LTX2 time scale factor: 8 pixel frames per latent frame
    _time_scale = 8
    guide_latent_t = max(1, num_video_frames_to_guide // _time_scale)
    video_mask[:, :, :guide_latent_t, :, :] = 0.0

    video_out: dict[str, Any] = dict(video_latent)
    video_out["noise_mask"] = video_mask

    # Audio noise mask: shape [B, 1, T_audio] (broadcast over frequency/channel)
    audio_samples = audio_latent["samples"]
    a_batch, a_ch, a_t = audio_samples.shape[:3]
    audio_mask = torch.ones((a_batch, 1, a_t), dtype=torch.float32, device=audio_samples.device)
    overlap = min(audio_overlap_latents, a_t)
    if overlap > 0:
        audio_mask[:, :, :overlap] = 0.0

    audio_out: dict[str, Any] = dict(audio_latent)
    audio_out["noise_mask"] = audio_mask

    return video_out, audio_out


__all__ = [
    "ltxv_audio_vae_encode",
    "ltxv_audio_vae_decode",
    "ltxv_empty_latent_audio",
    "encode_ace_step_15_audio",
    "empty_ace_step_15_latent_audio",
    "ltxv_concat_av_latent",
    "ltxv_separate_av_latent",
    "load_audio",
    "audio_crop",
    "audio_separation",
    "trim_audio_duration",
    "ltxv_audio_video_mask",
    "audio_encoder_encode",
    "vae_decode_audio",
]
