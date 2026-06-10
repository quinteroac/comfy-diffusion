"""Prompt conditioning helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol


class _ClipTextEncoder(Protocol):
    def tokenize(self, text: str) -> Any: ...

    def encode_from_tokens_scheduled(self, tokens: Any) -> Any: ...


class _FluxClipTextEncoder(Protocol):
    def tokenize(self, text: str) -> Any: ...

    def encode_from_tokens_scheduled(
        self,
        tokens: Any,
        add_dict: dict[str, Any] | None = None,
    ) -> Any: ...


class _ClipVisionEncoder(Protocol):
    def encode_image(self, image: Any, crop: bool = True) -> Any: ...


class _VaeEncoder(Protocol):
    latent_channels: int

    def spacial_compression_encode(self) -> int: ...

    def encode(self, image: Any) -> Any: ...


class _LtxvVaeEncoder(Protocol):
    def encode(self, image: Any) -> Any: ...


def _get_ltxv_conditioning_dependencies() -> tuple[Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers


def _get_wan_conditioning_dependencies() -> tuple[Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers


def _get_wan_vace_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.latent_formats
    import comfy.model_management
    import comfy.utils
    import node_helpers
    import torch

    return torch, comfy.model_management, comfy.utils, node_helpers, comfy.latent_formats


def _get_clip_vision_output_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.clip_vision

    return comfy.clip_vision.Output


def _normalize_prompt_text(text: str) -> str:
    return " " if text == "" else text


def encode_prompt(
    clip: _ClipTextEncoder,
    text: str,
    negative_text: str | None = None,
) -> Any:
    """Encode prompt text with a ComfyUI-compatible CLIP object.

    When ``negative_text`` is ``None`` (default), returns a single conditioning
    object for the given ``text``.

    When ``negative_text`` is provided, encodes both prompts and returns a
    ``(positive, negative)`` tuple — the form expected by sampling functions.
    """
    def _encode(t: str) -> Any:
        tokens = clip.tokenize(_normalize_prompt_text(t))
        return clip.encode_from_tokens_scheduled(tokens)

    if negative_text is None:
        return _encode(text)
    return _encode(text), _encode(negative_text)


def encode_prompt_flux(
    clip: _FluxClipTextEncoder,
    text: str,
    clip_l_text: str,
    guidance: float = 3.5,
) -> Any:
    """Encode prompts with Flux dual-encoder token layout."""
    clip_l_tokens = clip.tokenize(_normalize_prompt_text(clip_l_text))
    clip_l_tokens["t5xxl"] = clip.tokenize(_normalize_prompt_text(text))["t5xxl"]
    return clip.encode_from_tokens_scheduled(clip_l_tokens, add_dict={"guidance": guidance})


def encode_clip_vision(
    clip_vision: _ClipVisionEncoder,
    image: Any,
    crop: Literal["center", "none"] = "center",
) -> Any:
    """Encode image features with a CLIP-vision model."""
    return clip_vision.encode_image(image, crop=(crop == "center"))


def wan_image_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    start_image: Any | None = None,
    clip_vision_output: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN image-to-video conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers = _get_wan_conditioning_dependencies()

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        image = torch.ones(
            (length, height, width, start_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        ) * 0.5
        image[: start_image.shape[0]] = start_image

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.ones(
            (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        )
        mask[:, :, : ((start_image.shape[0] - 1) // 4) + 1] = 0.0

        values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

    if clip_vision_output is not None:
        values = {"clip_vision_output": clip_vision_output}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

    return positive, negative, {"samples": latent}


def wan_first_last_frame_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    start_image: Any | None = None,
    end_image: Any | None = None,
    clip_vision_start_image: Any | None = None,
    clip_vision_end_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN first/last-frame conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers = _get_wan_conditioning_dependencies()

    spacial_scale = vae.spacial_compression_encode()
    latent = torch.zeros(
        [
            batch_size,
            vae.latent_channels,
            ((length - 1) // 4) + 1,
            height // spacial_scale,
            width // spacial_scale,
        ],
        device=model_management.intermediate_device(),
    )

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    if end_image is not None:
        end_image = comfy_utils.common_upscale(
            end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

    image = torch.ones((length, height, width, 3)) * 0.5
    mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

    if start_image is not None:
        image[: start_image.shape[0]] = start_image
        mask[:, :, : start_image.shape[0] + 3] = 0.0

    if end_image is not None:
        image[-end_image.shape[0] :] = end_image
        mask[:, :, -end_image.shape[0] :] = 0.0

    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
    values = {"concat_latent_image": concat_latent_image, "concat_mask": mask}
    positive = node_helpers.conditioning_set_values(positive, values)
    negative = node_helpers.conditioning_set_values(negative, values)

    clip_vision_output = None
    if clip_vision_start_image is not None:
        clip_vision_output = clip_vision_start_image

    if clip_vision_end_image is not None:
        if clip_vision_output is not None:
            states = torch.cat(
                [
                    clip_vision_output.penultimate_hidden_states,
                    clip_vision_end_image.penultimate_hidden_states,
                ],
                dim=-2,
            )
            clip_vision_output = _get_clip_vision_output_type()()
            clip_vision_output.penultimate_hidden_states = states
        else:
            clip_vision_output = clip_vision_end_image

    if clip_vision_output is not None:
        values = {"clip_vision_output": clip_vision_output}
        positive = node_helpers.conditioning_set_values(positive, values)
        negative = node_helpers.conditioning_set_values(negative, values)

    return positive, negative, {"samples": latent}


def ltxv_img_to_video(
    positive: Any,
    negative: Any,
    image: Any,
    vae: _LtxvVaeEncoder,
    width: int = 768,
    height: int = 512,
    length: int = 97,
    batch_size: int = 1,
    strength: float = 1.0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build LTXV image-to-video conditioning and latent with frame noise mask."""
    torch, model_management, comfy_utils, _ = _get_ltxv_conditioning_dependencies()

    pixels = comfy_utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    latent_conditioning = vae.encode(pixels[:, :, :, :3])

    latent = torch.zeros(
        [batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32],
        device=model_management.intermediate_device(),
    )
    latent[:, :, : latent_conditioning.shape[2]] = latent_conditioning

    conditioning_latent_frames_mask = torch.ones(
        (batch_size, 1, latent.shape[2], 1, 1),
        dtype=torch.float32,
        device=latent.device,
    )
    conditioning_latent_frames_mask[:, :, : latent_conditioning.shape[2]] = 1.0 - strength

    latent_with_noise_mask = {
        "samples": latent,
        "noise_mask": conditioning_latent_frames_mask,
    }
    return positive, negative, latent_with_noise_mask


def ltxv_conditioning(
    positive: Any,
    negative: Any,
    frame_rate: float = 25.0,
) -> tuple[Any, Any]:
    """Inject frame rate metadata into LTXV positive and negative conditioning."""
    _, _, _, node_helpers = _get_ltxv_conditioning_dependencies()
    values = {"frame_rate": frame_rate}
    return (
        node_helpers.conditioning_set_values(positive, values),
        node_helpers.conditioning_set_values(negative, values),
    )


def ltxv_crop_guides(
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Crop keyframe guide frames from LTXV conditioning and latent before the main sampling pass.

    Removes ``keyframe_idxs`` and ``guide_attention_entries`` from both conditioning
    objects and slices the corresponding frames off the end of the latent tensor.
    When there are no keyframe guides (``num_keyframes == 0``) the inputs are
    returned unchanged.
    """
    # Extract keyframe_idxs without importing torch so the function is import-safe.
    keyframe_idxs = None
    for entry in positive:
        if "keyframe_idxs" in entry[1]:
            keyframe_idxs = entry[1]["keyframe_idxs"]
            break

    if keyframe_idxs is None:
        return positive, negative, latent

    torch, _, _, node_helpers = _get_ltxv_conditioning_dependencies()

    num_keyframes: int = torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]

    if num_keyframes == 0:
        return positive, negative, latent

    latent_image = latent["samples"].clone()
    noise_mask = latent.get("noise_mask", None)
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()

    latent_image = latent_image[:, :, :-num_keyframes]
    noise_mask = noise_mask[:, :, :-num_keyframes]

    guide_values: dict[str, Any] = {"keyframe_idxs": None, "guide_attention_entries": None}
    positive = node_helpers.conditioning_set_values(positive, guide_values)
    negative = node_helpers.conditioning_set_values(negative, guide_values)

    return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


def _get_void_quadmask_preprocess_type() -> Any:
    """Resolve ComfyUI VOIDQuadmaskPreprocess node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_void import VOIDQuadmaskPreprocess

    return VOIDQuadmaskPreprocess


def _get_void_inpaint_conditioning_type() -> Any:
    """Resolve ComfyUI VOIDInpaintConditioning node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_void import VOIDInpaintConditioning

    return VOIDInpaintConditioning


def void_quadmask_preprocess(mask: Any, dilate_width: int = 0) -> Any:
    """Preprocess a VOID quadmask video mask.

    Mirrors ComfyUI's ``VOIDQuadmaskPreprocess``.  The mask is quantized to the
    four semantic VOID levels, inverted, and normalized for
    ``void_inpaint_conditioning``.
    """
    node_type = _get_void_quadmask_preprocess_type()
    output = node_type.execute(mask=mask, dilate_width=dilate_width)
    result = getattr(output, "result", output)
    return result[0]


def void_inpaint_conditioning(
    positive: Any,
    negative: Any,
    vae: Any,
    video: Any,
    quadmask: Any,
    width: int = 672,
    height: int = 384,
    length: int = 45,
    batch_size: int = 1,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build VOID inpainting conditioning and the matching noise latent.

    Mirrors ComfyUI's ``VOIDInpaintConditioning`` for CogVideoX-Fun-V1.5 VOID
    inpainting models.
    """
    node_type = _get_void_inpaint_conditioning_type()
    output = node_type.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        video=video,
        quadmask=quadmask,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
    )
    result = getattr(output, "result", output)
    return result[0], result[1], result[2]


def _get_ltxv_add_guide_type() -> Any:
    """Resolve ComfyUI LTXVAddGuide node at call time."""
    from comfy_extras.nodes_lt import LTXVAddGuide

    return LTXVAddGuide


def ltxv_add_guide(
    positive: Any,
    negative: Any,
    vae: Any,
    latent: dict[str, Any],
    image: Any,
    frame_idx: int = 0,
    strength: float = 1.0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Inject an image guide into LTXV video conditioning and latent.

    Mirrors ``LTXVAddGuide.execute()`` from ComfyUI (``comfy_extras.nodes_lt``).

    Use ``frame_idx=0`` to condition the **first** frame and ``frame_idx=-1``
    to condition the **last** frame.  For multi-frame guide videos the
    ``frame_idx`` must be divisible by 8 (except when it is 0).

    Parameters
    ----------
    positive : Any
        Positive conditioning from :func:`ltxv_conditioning` (or
        :func:`encode_prompt`).
    negative : Any
        Negative conditioning.
    vae : Any
        VAE loaded from the LTX-Video checkpoint (used to encode the guide
        image into latent space).
    latent : dict[str, Any]
        Video latent dictionary (``{"samples": tensor, ...}``).
    image : Any
        Image or video tensor (BHWC float32) to use as the guide frame.
    frame_idx : int, optional
        Latent frame index to place the guide at.  Negative values are
        counted from the end of the sequence.  Default ``0`` (first frame).
    strength : float, optional
        Conditioning strength in ``[0.0, 1.0]``.  Default ``1.0``.

    Returns
    -------
    tuple[Any, Any, dict[str, Any]]
        Updated ``(positive, negative, latent)`` with the guide frame
        appended to the keyframe sequence.
    """
    add_guide_type = _get_ltxv_add_guide_type()
    result = add_guide_type.execute(
        positive=positive,
        negative=negative,
        vae=vae,
        latent=latent,
        image=image,
        frame_idx=frame_idx,
        strength=strength,
    )
    r = getattr(result, "result", result)
    return r[0], r[1], r[2]


def _append_guide_attention_entry(
    conditioning: Any,
    pre_filter_count: int,
    latent_shape: list[int],
) -> Any:
    """Append IC-LoRA guide attention metadata to conditioning."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import node_helpers

    existing_entries: list[Any] = []
    for entry in conditioning:
        entries = entry[1].get("guide_attention_entries")
        if entries is not None:
            existing_entries = list(entries)
            break

    updated_entries = [
        *existing_entries,
        {
            "pre_filter_count": pre_filter_count,
            "strength": 1.0,
            "pixel_mask": None,
            "latent_shape": latent_shape,
        },
    ]
    return node_helpers.conditioning_set_values(
        conditioning, {"guide_attention_entries": updated_entries}
    )


def _ltxv_dilate_latent(
    latent: dict[str, Any],
    horizontal_scale: int,
    vertical_scale: int,
) -> dict[str, Any]:
    if horizontal_scale == 1 and vertical_scale == 1:
        return latent

    import torch

    samples = latent["samples"]
    mask = latent.get("noise_mask", None)
    dilated_shape = samples.shape[:3] + (
        samples.shape[3] * vertical_scale,
        samples.shape[4] * horizontal_scale,
    )

    dilated_samples = torch.zeros(
        dilated_shape,
        device=samples.device,
        dtype=samples.dtype,
        requires_grad=False,
    )
    dilated_samples[..., ::vertical_scale, ::horizontal_scale] = samples

    dilated_mask_shape = (
        dilated_samples.shape[0],
        1,
        dilated_samples.shape[2],
        dilated_samples.shape[3],
        dilated_samples.shape[4],
    )
    dilated_mask = torch.full(
        dilated_mask_shape,
        -1.0,
        device=samples.device,
        dtype=samples.dtype,
        requires_grad=False,
    )
    dilated_mask[..., ::vertical_scale, ::horizontal_scale] = (
        mask if mask is not None else 1.0
    )
    return {"samples": dilated_samples, "noise_mask": dilated_mask}


def ltx_add_video_ic_lora_guide(
    positive: Any,
    negative: Any,
    vae: Any,
    latent: dict[str, Any],
    image: Any,
    frame_idx: int = 0,
    strength: float = 1.0,
    latent_downscale_factor: float = 1.0,
    crop: str = "disabled",
    use_tiled_encode: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 64,
) -> tuple[Any, Any, dict[str, Any]]:
    """Inject a video IC-LoRA guide into LTXV conditioning and latent.

    Mirrors Lightricks' ``LTXAddVideoICLoRAGuide`` node. This variant supports
    downscaled reference latents, which the LTX-2.3 motion-track IC-LoRA uses
    with ``reference_downscale_factor=2`` / reference resolution ``0.5x``.
    """
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()

    import comfy.utils
    import comfy_extras.nodes_lt as nodes_lt
    import torch

    scale_factors = vae.downscale_index_formula
    latent_image = latent["samples"]
    noise_mask = nodes_lt.get_noise_mask(latent)

    _, _, latent_length, latent_height, latent_width = latent_image.shape

    time_scale_factor = scale_factors[0]
    num_frames_to_keep = (
        ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
    )
    causal_fix = frame_idx == 0 or num_frames_to_keep == 1
    if not causal_fix:
        image = torch.cat([image[:1], image], dim=0)

    images = image[
        : ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
    ]
    target_width = int(latent_width * scale_factors[1] / latent_downscale_factor)
    target_height = int(latent_height * scale_factors[2] / latent_downscale_factor)
    pixels = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        target_width,
        target_height,
        "bilinear",
        crop=crop,
    ).movedim(1, -1)
    encode_pixels = pixels[:, :, :, :3]
    if use_tiled_encode:
        guide_latent = vae.encode_tiled(
            encode_pixels,
            tile_x=tile_size,
            tile_y=tile_size,
            overlap=tile_overlap,
        )
    else:
        guide_latent = vae.encode(encode_pixels)

    if not causal_fix:
        guide_latent = guide_latent[:, :, 1:, :, :]
        image = image[1:]

    guide_orig_shape = list(guide_latent.shape[2:])
    guide_mask = None

    if latent_downscale_factor > 1:
        downscale_int = int(latent_downscale_factor)
        if latent_width % downscale_int != 0 or latent_height % downscale_int != 0:
            raise ValueError(
                f"Latent spatial size {latent_width}x{latent_height} must be divisible "
                f"by latent_downscale_factor {latent_downscale_factor}"
            )

        dilated = _ltxv_dilate_latent(
            {"samples": guide_latent},
            horizontal_scale=downscale_int,
            vertical_scale=downscale_int,
        )
        guide_mask = dilated["noise_mask"]
        guide_latent = dilated["samples"]

    iclora_tokens_added = (
        guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
    )

    frame_idx, latent_idx = nodes_lt.LTXVAddGuide.get_latent_index(
        positive,
        latent_length,
        len(image),
        frame_idx,
        scale_factors,
    )
    if latent_idx + guide_latent.shape[2] > latent_length:
        raise ValueError("Conditioning frames exceed the length of the latent sequence.")

    positive, negative, latent_image, noise_mask = nodes_lt.LTXVAddGuide.append_keyframe(
        positive,
        negative,
        frame_idx,
        latent_image,
        noise_mask,
        guide_latent,
        strength,
        scale_factors,
        guide_mask=guide_mask,
        latent_downscale_factor=latent_downscale_factor,
        causal_fix=causal_fix,
    )

    positive = _append_guide_attention_entry(
        positive,
        iclora_tokens_added,
        guide_orig_shape,
    )
    negative = _append_guide_attention_entry(
        negative,
        iclora_tokens_added,
        guide_orig_shape,
    )

    return positive, negative, {"samples": latent_image, "noise_mask": noise_mask}


def conditioning_zero_out(conditioning: Any) -> list[Any]:
    """Zero out conditioning tensors.

    Mirrors ``ConditioningZeroOut.zero_out()`` from ComfyUI.  All conditioning
    tensors and any pooled / lyrics auxiliary tensors are replaced with zeros.
    Used to create a null conditioning object for negative guidance.
    """
    import torch

    result = []
    for entry in conditioning:
        d = entry[1].copy()
        pooled = d.get("pooled_output", None)
        if pooled is not None:
            d["pooled_output"] = torch.zeros_like(pooled)
        lyrics = d.get("conditioning_lyrics", None)
        if lyrics is not None:
            d["conditioning_lyrics"] = torch.zeros_like(lyrics)
        result.append([torch.zeros_like(entry[0]), d])
    return result


def conditioning_combine(
    cond_a: Any,
    cond_b: Any | None = None,
    *additional: Any,
) -> list[Any]:
    """Combine two or more conditioning objects into one list.

    Supports both:
    - ``conditioning_combine(cond_a, cond_b, cond_c, ...)``
    - ``conditioning_combine([cond_a, cond_b, cond_c, ...])``
    """
    sources: list[Any]
    if cond_b is None and not additional:
        if not isinstance(cond_a, Sequence) or isinstance(cond_a, (str, bytes)):
            raise ValueError(
                "conditioning_combine() requires at least two conditioning objects"
            )

        sources = list(cond_a)
        if len(sources) < 2:
            raise ValueError(
                "conditioning_combine() requires at least two conditioning objects"
            )
    else:
        if cond_b is None:
            raise ValueError(
                "conditioning_combine() requires at least two conditioning objects"
            )
        sources = [cond_a, cond_b, *additional]

    merged: list[Any] = []
    for source in sources:
        merged.extend(source)

    return merged


def conditioning_set_mask(
    conditioning: Any,
    mask: Any,
    strength: float = 1.0,
    set_cond_area: Literal["default", "mask bounds"] = "default",
) -> list[Any]:
    """Attach mask metadata to each conditioning entry.

    Mirrors ComfyUI's ``ConditioningSetMask`` node behavior.
    """
    set_area_to_bounds = set_cond_area != "default"
    normalized_mask = mask if len(mask.shape) >= 3 else mask.unsqueeze(0)

    output: list[Any] = []
    for item in conditioning:
        updated = [item[0], item[1].copy()]
        updated[1]["mask"] = normalized_mask
        updated[1]["set_area_to_bounds"] = set_area_to_bounds
        updated[1]["mask_strength"] = strength
        output.append(updated)

    return output


def _validate_timestep_percent(name: str, value: float) -> None:
    if not isinstance(value, float):
        raise TypeError(f"{name} must be a float")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def conditioning_set_timestep_range(
    conditioning: Any,
    start: float,
    end: float,
) -> list[Any]:
    """Attach timestep bounds metadata to each conditioning entry."""
    _validate_timestep_percent("start", start)
    _validate_timestep_percent("end", end)

    output: list[Any] = []
    for item in conditioning:
        updated = [item[0], item[1].copy()]
        updated[1]["start_percent"] = start
        updated[1]["end_percent"] = end
        output.append(updated)

    return output


def flux_guidance(conditioning: Any, guidance: float = 3.5) -> list[Any]:
    """Apply Flux guidance value metadata to each conditioning entry."""
    output: list[Any] = []
    for item in conditioning:
        updated = [item[0], item[1].copy()]
        updated[1]["guidance"] = guidance
        output.append(updated)

    return output


def wan_vace_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    strength: float = 1.0,
    control_video: Any | None = None,
    control_masks: Any | None = None,
    reference_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any], int]:
    """Build WAN VACE conditioning and empty latent samples.

    Args:
        positive: Positive conditioning.
        negative: Negative conditioning.
        vae: VAE model used to encode control frames and reference image.
        width: Video width in pixels (multiple of 8).
        height: Video height in pixels (multiple of 8).
        length: Number of video frames.
        batch_size: Latent batch size.
        strength: VACE conditioning strength.
        control_video: Optional control video tensor ``[T, H, W, C]`` in [0, 1].
        control_masks: Optional mask tensor ``[T, H, W]`` or ``[T, 1, H, W]`` in [0, 1].
        reference_image: Optional reference image tensor ``[1, H, W, C]`` in [0, 1].

    Returns:
        ``(positive, negative, latent, trim_latent)`` where ``trim_latent`` is
        the number of latent frames prepended for the reference image (0 if none).
    """
    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    latent_length = ((length - 1) // 4) + 1

    if control_video is not None:
        control_video = comfy_utils.common_upscale(
            control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        if control_video.shape[0] < length:
            control_video = torch.nn.functional.pad(
                control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5
            )
    else:
        control_video = torch.ones((length, height, width, 3)) * 0.5

    if reference_image is not None:
        reference_image = comfy_utils.common_upscale(
            reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        ref_latent = vae.encode(reference_image[:, :, :, :3])
        reference_image = torch.cat(
            [ref_latent, comfy_latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))], dim=1
        )

    if control_masks is None:
        mask = torch.ones((length, height, width, 1))
    else:
        mask = control_masks
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = comfy_utils.common_upscale(
            mask[:length], width, height, "bilinear", "center"
        ).movedim(1, -1)
        if mask.shape[0] < length:
            mask = torch.nn.functional.pad(
                mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0
            )

    control_video = control_video - 0.5
    inactive = (control_video * (1 - mask)) + 0.5
    reactive = (control_video * mask) + 0.5

    inactive = vae.encode(inactive[:, :, :, :3])
    reactive = vae.encode(reactive[:, :, :, :3])
    control_video_latent = torch.cat((inactive, reactive), dim=1)
    if reference_image is not None:
        control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask = mask.permute(2, 4, 0, 1, 3)
    mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode="nearest-exact"
    ).squeeze(0)

    trim_latent = 0
    if reference_image is not None:
        mask_pad = torch.zeros_like(mask[:, : reference_image.shape[2], :, :])
        mask = torch.cat((mask_pad, mask), dim=1)
        latent_length += reference_image.shape[2]
        trim_latent = reference_image.shape[2]

    mask = mask.unsqueeze(0)

    positive = node_helpers.conditioning_set_values(
        positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True
    )
    negative = node_helpers.conditioning_set_values(
        negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True
    )

    latent = torch.zeros(
        [batch_size, 16, latent_length, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    return positive, negative, {"samples": latent}, trim_latent


# ── WAN Fun Control ──────────────────────────────────────────────────────────


def wan_fun_control_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    *,
    clip_vision_output: Any | None = None,
    start_image: Any | None = None,
    control_video: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN Fun Control conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    concat_latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    concat_latent = comfy_latent_formats.Wan21().process_out(concat_latent)
    concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        concat_latent_image = vae.encode(start_image[:, :, :, :3])
        concat_latent[:, 16:, : concat_latent_image.shape[2]] = concat_latent_image[:, :, : concat_latent.shape[2]]

    if control_video is not None:
        control_video = comfy_utils.common_upscale(
            control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        concat_latent_image = vae.encode(control_video[:, :, :, :3])
        concat_latent[:, :16, : concat_latent_image.shape[2]] = concat_latent_image[:, :, : concat_latent.shape[2]]

    values: dict[str, Any] = {"concat_latent_image": concat_latent}
    positive = node_helpers.conditioning_set_values(positive, values)
    negative = node_helpers.conditioning_set_values(negative, values)

    if clip_vision_output is not None:
        cv_values: dict[str, Any] = {"clip_vision_output": clip_vision_output}
        positive = node_helpers.conditioning_set_values(positive, cv_values)
        negative = node_helpers.conditioning_set_values(negative, cv_values)

    return positive, negative, {"samples": latent}


def wan22_fun_control_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    *,
    ref_image: Any | None = None,
    control_video: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN 2.2 Fun Control conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    spacial_scale = vae.spacial_compression_encode()
    latent_channels = vae.latent_channels
    latent = torch.zeros(
        [batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
        device=model_management.intermediate_device(),
    )
    concat_latent = torch.zeros(
        [batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
        device=model_management.intermediate_device(),
    )
    if latent_channels == 48:
        concat_latent = comfy_latent_formats.Wan22().process_out(concat_latent)
    else:
        concat_latent = comfy_latent_formats.Wan21().process_out(concat_latent)
    concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)
    mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

    ref_latent = None
    if ref_image is not None:
        ref_image = comfy_utils.common_upscale(
            ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        ref_latent = vae.encode(ref_image[:, :, :, :3])

    if control_video is not None:
        control_video = comfy_utils.common_upscale(
            control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        concat_latent_image = vae.encode(control_video[:, :, :, :3])
        concat_latent[:, :latent_channels, : concat_latent_image.shape[2]] = concat_latent_image[:, :, : concat_latent.shape[2]]

    mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
    values22: dict[str, Any] = {"concat_latent_image": concat_latent, "concat_mask": mask, "concat_mask_index": latent_channels}
    positive = node_helpers.conditioning_set_values(positive, values22)
    negative = node_helpers.conditioning_set_values(negative, values22)

    if ref_latent is not None:
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    return positive, negative, {"samples": latent}


# ── WAN Fun Inpaint ──────────────────────────────────────────────────────────


def wan_fun_inpaint_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    *,
    clip_vision_output: Any | None = None,
    start_image: Any | None = None,
    end_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN Fun Inpaint conditioning (thin wrapper over first/last frame)."""
    return wan_first_last_frame_to_video(
        positive=positive,
        negative=negative,
        vae=vae,
        width=width,
        height=height,
        length=length,
        batch_size=batch_size,
        start_image=start_image,
        end_image=end_image,
        clip_vision_start_image=clip_vision_output,
    )


# ── WAN Camera ───────────────────────────────────────────────────────────────

_WAN_CAMERA_POSES: tuple[str, ...] = (
    "Static", "Pan Up", "Pan Down", "Pan Left", "Pan Right",
    "Zoom In", "Zoom Out", "Anti Clockwise (ACW)", "ClockWise (CW)",
)


def _get_wan_camera_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import numpy as np
    import torch
    from comfy_extras.nodes_camera_trajectory import (
        CAMERA_DICT,
        get_camera_motion,
        process_pose_params,
    )

    return torch, np, comfy.model_management, CAMERA_DICT, get_camera_motion, process_pose_params


def wan_camera_embedding(
    camera_pose: str,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    *,
    speed: float = 1.0,
    fx: float = 0.5,
    fy: float = 0.5,
    cx: float = 0.5,
    cy: float = 0.5,
) -> tuple[Any, int, int, int]:
    """Build a WAN camera trajectory embedding.

    ``camera_pose`` must be one of ``_WAN_CAMERA_POSES``.
    Returns ``(camera_embedding, width, height, length)``.
    """
    torch, np, model_management, CAMERA_DICT, get_camera_motion, process_pose_params = _get_wan_camera_dependencies()

    angle = np.array(CAMERA_DICT[camera_pose]["angle"])
    T = np.array(CAMERA_DICT[camera_pose]["T"])
    RT = get_camera_motion(angle, T, speed, length)

    trajs = []
    for cp in RT.tolist():
        traj = [fx, fy, cx, cy, 0, 0]
        traj.extend(cp[0])
        traj.extend(cp[1])
        traj.extend(cp[2])
        traj.extend([0, 0, 0, 1])
        trajs.append(traj)

    cam_params = np.array([[float(x) for x in pose] for pose in trajs])
    cam_params = np.concatenate([np.zeros_like(cam_params[:, :1]), cam_params], 1)
    control_camera_video = process_pose_params(cam_params, width=width, height=height)
    control_camera_video = (
        control_camera_video.permute([3, 0, 1, 2])
        .unsqueeze(0)
        .to(device=model_management.intermediate_device())
    )
    control_camera_video = torch.concat(
        [
            torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
            control_camera_video[:, :, 1:],
        ],
        dim=2,
    ).transpose(1, 2)
    b, f, c, h, w = control_camera_video.shape
    control_camera_video = (
        control_camera_video.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    )
    control_camera_video = control_camera_video.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
    return control_camera_video, width, height, length


def wan_camera_image_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    *,
    clip_vision_output: Any | None = None,
    start_image: Any | None = None,
    camera_conditions: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN camera-guided image-to-video conditioning and empty latent samples."""
    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    concat_latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    concat_latent = comfy_latent_formats.Wan21().process_out(concat_latent)

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        concat_latent_image = vae.encode(start_image[:, :, :, :3])
        concat_latent[:, :, : concat_latent_image.shape[2]] = concat_latent_image[:, :, : concat_latent.shape[2]]
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
        mask[:, :, : start_image.shape[0] + 3] = 0.0
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent, "concat_mask": mask})

    if camera_conditions is not None:
        positive = node_helpers.conditioning_set_values(positive, {"camera_conditions": camera_conditions})
        negative = node_helpers.conditioning_set_values(negative, {"camera_conditions": camera_conditions})

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    return positive, negative, {"samples": latent}


# ── WAN Phantom Subject ──────────────────────────────────────────────────────


def wan_phantom_subject_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    *,
    images: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build WAN Phantom subject conditioning.

    Returns ``(positive, negative_text, negative_img_text, latent)``.
    ``negative_text`` carries subject image conditioning;
    ``negative_img_text`` carries zeroed-out conditioning for CFG.
    """
    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )
    negative_text = negative
    if images is not None:
        images = comfy_utils.common_upscale(
            images[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        latent_images = [vae.encode(images[i].unsqueeze(0)[:, :, :, :3]) for i in range(images.shape[0])]
        concat_latent_image = torch.cat(latent_images, dim=2)

        positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
        negative_text = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
        negative = node_helpers.conditioning_set_values(
            negative,
            {"time_dim_concat": comfy_latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))},
        )

    return positive, negative_text, negative, {"samples": latent}


# ── WAN Track to Video ───────────────────────────────────────────────────────


def _wan_track_parse_json_tracks(tracks: str) -> list[Any]:
    import json

    tracks_data: list[Any] = []
    try:
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)
        if tracks_data and isinstance(tracks_data[0], dict) and "x" in tracks_data[0]:
            tracks_data = [tracks_data]
    except json.JSONDecodeError:
        tracks_data = []
    return tracks_data


def _wan_track_pad_pts(tr: list[Any]) -> Any:
    import numpy as np

    FIXED_LENGTH = 121
    pts = np.array([[p["x"], p["y"], 1] for p in tr], dtype=np.float32)
    n = pts.shape[0]
    if n < FIXED_LENGTH:
        pts = np.vstack((pts, np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)))
    else:
        pts = pts[:FIXED_LENGTH]
    return pts.reshape(FIXED_LENGTH, 1, 3)


def _wan_track_process_tracks(tracks_np: Any, frame_size: tuple[int, int], num_frames: int) -> Any:
    import math

    import torch

    tracks = torch.from_numpy(tracks_np).float()
    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))
    tracks, visibles = tracks[..., :2], tracks[..., 2:3]
    short_edge = min(*frame_size)
    frame_center = torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = (tracks - frame_center) / short_edge * 2
    visibles = visibles * 2 - 1
    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)
    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)
    out_0 = out_[:1]
    out_l = out_[1:]
    a = 120 // math.gcd(120, num_frames)
    b_val = num_frames // math.gcd(120, num_frames)
    out_l = torch.repeat_interleave(out_l, b_val, dim=0)[1::a]
    return torch.cat([out_0, out_l], dim=0)


def _wan_track_ind_sel(target: Any, ind: Any, dim: int = 1) -> Any:
    import torch

    assert len(ind.shape) > dim
    target = target.expand(
        *([ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)] + [-1] * (len(target.shape) - dim))
    )
    ind_pad = ind
    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1):])
    return torch.gather(target, dim=dim, index=ind_pad)


def _wan_track_merge_final(vert_attr: Any, weight: Any, vert_assign: Any) -> Any:
    import torch

    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        tensor = vert_attr.reshape([1] * target_dim + list(vert_attr.shape))
    else:
        tensor = vert_attr.reshape([vert_attr.shape[0]] + [1] * (target_dim - 1) + list(vert_attr.shape[1:]))
    sel_attr = _wan_track_ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)
    return torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)


def _wan_track_patch_motion_single(
    tracks: Any, vid: Any, temperature: float, vae_divide: tuple[int, int], topk: int
) -> tuple[Any, Any]:
    import torch

    _, T, H, W = vid.shape
    N = tracks.shape[2]
    _, tracks_xy, visible = torch.split(tracks, [1, 2, 1], dim=-1)
    tracks_n = (tracks_xy / torch.tensor([W / min(H, W), H / min(H, W)], device=tracks_xy.device)).clamp(-1, 1)
    visible = visible.clamp(0, 1)
    xx = torch.linspace(-W / min(H, W), W / min(H, W), W)
    yy = torch.linspace(-H / min(H, W), H / min(H, W), H)
    grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(tracks_xy.device)
    tracks_pad = tracks_xy[:, 1:]
    visible_pad = visible[:, 1:]
    visible_align = visible_pad.view(T - 1, 4, *visible_pad.shape[2:]).sum(1)
    tracks_align = (tracks_pad * visible_pad).view(T - 1, 4, *tracks_pad.shape[2:]).sum(1) / (visible_align + 1e-5)
    dist_ = (tracks_align[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
    weight = torch.exp(-dist_ * temperature) * visible_align.clamp(0, 1).view(T - 1, 1, 1, N)
    vert_weight, vert_index = torch.topk(weight, k=min(topk, weight.shape[-1]), dim=-1)
    point_feature = torch.nn.functional.grid_sample(
        vid.permute(1, 0, 2, 3)[:1], tracks_n[:, :1].type(vid.dtype),
        mode="bilinear", padding_mode="zeros", align_corners=False,
    ).squeeze(0).squeeze(1).permute(1, 0)
    out_feature = _wan_track_merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2)
    mix_feature = out_feature + vid[:, 1:] * (1 - vert_weight.sum(-1).clamp(0, 1))
    out_feature_full = torch.cat([vid[:, :1], mix_feature], dim=1)
    out_mask_full = torch.cat([torch.ones_like(vert_weight.sum(-1)[:1]), vert_weight.sum(-1)], dim=0)
    return out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full


def _wan_track_patch_motion(
    tracks: Any, vid: Any, temperature: float = 220.0, vae_divide: tuple[int, int] = (4, 16), topk: int = 2
) -> tuple[Any, Any]:
    import torch

    out_masks, out_features = [], []
    for b in range(len(tracks)):
        mask, feature = _wan_track_patch_motion_single(tracks[b], vid[b], temperature, vae_divide, topk)
        out_masks.append(mask)
        out_features.append(feature)
    return torch.stack(out_masks, dim=0), torch.stack(out_features, dim=0)


def wan_track_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    tracks: str,
    width: int = 832,
    height: int = 480,
    length: int = 81,
    batch_size: int = 1,
    *,
    temperature: float = 220.0,
    topk: int = 2,
    start_image: Any | None = None,
    clip_vision_output: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN motion-track conditioning from JSON point-track data.

    Falls back to ``wan_image_to_video`` when ``tracks`` is empty or invalid JSON.
    """
    import numpy as np

    torch, model_management, comfy_utils, node_helpers, comfy_latent_formats = _get_wan_vace_dependencies()

    tracks_data = _wan_track_parse_json_tracks(tracks)

    if not tracks_data:
        return wan_image_to_video(
            positive=positive, negative=negative, vae=vae,
            width=width, height=height, length=length, batch_size=batch_size,
            start_image=start_image, clip_vision_output=clip_vision_output,
        )

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=model_management.intermediate_device(),
    )

    if isinstance(tracks_data[0][0], dict):
        tracks_data = [tracks_data]

    processed_tracks = []
    for batch in tracks_data:
        arrs = [_wan_track_pad_pts(track) for track in batch]
        tracks_np = np.stack(arrs, axis=0)
        processed_tracks.append(_wan_track_process_tracks(tracks_np, (width, height), length - 1).unsqueeze(0))

    if start_image is not None:
        start_image = comfy_utils.common_upscale(
            start_image[:batch_size].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        videos = torch.ones(
            (start_image.shape[0], length, height, width, start_image.shape[-1]),
            device=start_image.device, dtype=start_image.dtype,
        ) * 0.5
        for i in range(start_image.shape[0]):
            videos[i, 0] = start_image[i]
        videos = comfy_utils.resize_to_batch_size(videos, batch_size)
        latent_videos = [vae.encode(videos[i, :, :, :, :3]) for i in range(batch_size)]
        y = comfy_latent_formats.Wan21().process_in(torch.cat(latent_videos, dim=0))
        processed_tracks = comfy_utils.resize_list_to_batch_size(processed_tracks, batch_size)
        mask, concat_latent_image = _wan_track_patch_motion(
            processed_tracks, y, temperature=temperature, topk=topk, vae_divide=(4, 16)
        )
        concat_latent_image = comfy_latent_formats.Wan21().process_out(concat_latent_image)
        mask = -mask + 1.0
        positive = node_helpers.conditioning_set_values(positive, {"concat_mask": mask, "concat_latent_image": concat_latent_image})
        negative = node_helpers.conditioning_set_values(negative, {"concat_mask": mask, "concat_latent_image": concat_latent_image})

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    return positive, negative, {"samples": latent}


# ── WAN Sound nodes ──────────────────────────────────────────────────────────


def _wan_sound_linear_interpolation(features: Any, input_fps: float, output_fps: float, output_len: int | None = None) -> Any:
    import torch

    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(features, size=output_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


def _wan_sound_get_sample_indices(
    original_fps: float, total_frames: int, target_fps: float, num_sample: int, fixed_start: int | None = None
) -> Any:
    import math

    import numpy as np

    required_duration = num_sample / target_fps
    required_origin_frames = int(math.ceil(required_duration * original_fps))
    if required_duration > total_frames / original_fps:
        raise ValueError("required_duration must be less than video length")
    if fixed_start is not None and fixed_start >= 0:
        start_frame = fixed_start
    else:
        max_start = total_frames - required_origin_frames
        if max_start < 0:
            raise ValueError("video length is too short")
        start_frame = int(np.random.randint(0, max_start + 1))
    start_time = start_frame / original_fps
    end_time = start_time + required_duration
    time_points = np.linspace(start_time, end_time, num_sample, endpoint=False)
    return np.clip(np.round(np.array(time_points) * original_fps).astype(int), 0, total_frames - 1)


def _wan_sound_get_audio_embed_bucket_fps(
    audio_embed: Any, fps: float = 16, batch_frames: int = 81, m: int = 0, video_rate: float = 30
) -> tuple[Any, int]:
    import math

    import torch

    num_layers, audio_frame_num, audio_dim = audio_embed.shape
    return_all_layers = num_layers > 1
    scale = video_rate / fps
    min_batch_num = int(audio_frame_num / (batch_frames * scale)) + 1
    bucket_num = min_batch_num * batch_frames
    padd_audio_num = math.ceil(min_batch_num * batch_frames / fps * video_rate) - audio_frame_num
    batch_idx = _wan_sound_get_sample_indices(
        original_fps=video_rate, total_frames=audio_frame_num + padd_audio_num,
        target_fps=fps, num_sample=bucket_num, fixed_start=0,
    )
    audio_sample_stride = int(video_rate / fps)
    batch_audio_eb = []
    for bi in batch_idx:
        if bi < audio_frame_num:
            chosen_idx = list(range(bi - m * audio_sample_stride, bi + (m + 1) * audio_sample_stride, audio_sample_stride))
            chosen_idx = [max(0, min(c, audio_frame_num - 1)) for c in chosen_idx]
            frame_audio_embed = (
                audio_embed[:, chosen_idx].flatten(start_dim=-2, end_dim=-1)
                if return_all_layers
                else audio_embed[0][chosen_idx].flatten()
            )
        else:
            frame_audio_embed = (
                torch.zeros([num_layers, audio_dim * (2 * m + 1)], device=audio_embed.device)
                if return_all_layers
                else torch.zeros([audio_dim * (2 * m + 1)], device=audio_embed.device)
            )
        batch_audio_eb.append(frame_audio_embed)
    return torch.cat([c.unsqueeze(0) for c in batch_audio_eb], dim=0), min_batch_num


def _wan_sound_to_video(
    positive: Any,
    negative: Any,
    vae: Any,
    width: int,
    height: int,
    length: int,
    batch_size: int,
    *,
    frame_offset: int = 0,
    ref_image: Any | None = None,
    audio_encoder_output: Any | None = None,
    control_video: Any | None = None,
    ref_motion: Any | None = None,
    ref_motion_latent: Any | None = None,
) -> tuple[Any, Any, dict[str, Any], int]:
    torch, model_management, comfy_utils, node_helpers, latent_formats = _get_wan_vace_dependencies()

    latent_t = ((length - 1) // 4) + 1

    if audio_encoder_output is not None:
        feat = torch.cat(audio_encoder_output["encoded_audio_all_layers"])
        feat = _wan_sound_linear_interpolation(feat, input_fps=50, output_fps=30)
        batch_frames = latent_t * 4
        audio_embed_bucket, _ = _wan_sound_get_audio_embed_bucket_fps(feat, fps=16, batch_frames=batch_frames, m=0, video_rate=30)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        audio_embed_bucket = audio_embed_bucket[:, :, :, frame_offset: frame_offset + batch_frames]
        if audio_embed_bucket.shape[3] > 0:
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_embed_bucket})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_embed_bucket * 0.0})
            frame_offset += batch_frames

    if ref_image is not None:
        ref_image = comfy_utils.common_upscale(
            ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        ref_latent = vae.encode(ref_image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

    if ref_motion is not None:
        if ref_motion.shape[0] > 73:
            ref_motion = ref_motion[-73:]
        ref_motion = comfy_utils.common_upscale(ref_motion.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if ref_motion.shape[0] < 73:
            r = torch.ones([73, height, width, 3]) * 0.5
            r[-ref_motion.shape[0]:] = ref_motion
            ref_motion = r
        ref_motion_latent = vae.encode(ref_motion[:, :, :, :3])

    if ref_motion_latent is not None:
        ref_motion_latent = ref_motion_latent[:, :, -19:]
        positive = node_helpers.conditioning_set_values(positive, {"reference_motion": ref_motion_latent})
        negative = node_helpers.conditioning_set_values(negative, {"reference_motion": ref_motion_latent})

    latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=model_management.intermediate_device())
    control_video_out = latent_formats.Wan21().process_out(torch.zeros_like(latent))
    if control_video is not None:
        control_video = comfy_utils.common_upscale(
            control_video[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        cv_latent = vae.encode(control_video[:, :, :, :3])
        control_video_out[:, :, : cv_latent.shape[2]] = cv_latent

    positive = node_helpers.conditioning_set_values(positive, {"control_video": control_video_out})
    negative = node_helpers.conditioning_set_values(negative, {"control_video": control_video_out})
    return positive, negative, {"samples": latent}, frame_offset


def wan_sound_image_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 77,
    batch_size: int = 1,
    *,
    audio_encoder_output: Any | None = None,
    ref_image: Any | None = None,
    control_video: Any | None = None,
    ref_motion: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN sound-driven image-to-video conditioning."""
    positive, negative, out_latent, _ = _wan_sound_to_video(
        positive=positive, negative=negative, vae=vae,
        width=width, height=height, length=length, batch_size=batch_size,
        ref_image=ref_image, audio_encoder_output=audio_encoder_output,
        control_video=control_video, ref_motion=ref_motion,
    )
    return positive, negative, out_latent


def wan_sound_image_to_video_extend(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    length: int,
    video_latent: dict[str, Any],
    *,
    audio_encoder_output: Any | None = None,
    ref_image: Any | None = None,
    control_video: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Extend a WAN sound-driven video by processing the next audio segment."""
    video_samples = video_latent["samples"]
    width = video_samples.shape[-1] * 8
    height = video_samples.shape[-2] * 8
    batch_size = video_samples.shape[0]
    frame_offset = video_samples.shape[-3] * 4
    positive, negative, out_latent, _ = _wan_sound_to_video(
        positive=positive, negative=negative, vae=vae,
        width=width, height=height, length=length, batch_size=batch_size,
        frame_offset=frame_offset, ref_image=ref_image,
        audio_encoder_output=audio_encoder_output, control_video=control_video,
        ref_motion_latent=video_samples,
    )
    return positive, negative, out_latent


def _wan_humo_get_audio_emb_window(audio_emb: Any, frame_num: int, frame0_idx: int, audio_shift: int = 2) -> tuple[Any, int]:
    import torch

    zero_audio = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    ed = 0
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack(
                [audio_emb[i] if 0 <= i < audio_emb.shape[0] else zero_audio for i in range(st, ed)], dim=0
            )
            wind_feat = torch.cat((zero_audio_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack(
                [audio_emb[i] if 0 <= i < audio_emb.shape[0] else zero_audio for i in range(st, ed)], dim=0
            )
        audio_emb_wind.append(wind_feat)
    return torch.stack(audio_emb_wind, dim=0), ed - audio_shift


def wan_humo_image_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 97,
    batch_size: int = 1,
    *,
    audio_encoder_output: Any | None = None,
    ref_image: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN Human Motion (HuMo) audio-driven conditioning."""
    torch, model_management, comfy_utils, node_helpers, _ = _get_wan_vace_dependencies()

    latent_t = ((length - 1) // 4) + 1
    latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=model_management.intermediate_device())

    if ref_image is not None:
        ref_image = comfy_utils.common_upscale(ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        ref_latent = vae.encode(ref_image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
    else:
        zero_latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=model_management.intermediate_device())
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [zero_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [zero_latent]}, append=True)

    if audio_encoder_output is not None:
        audio_emb = torch.stack(audio_encoder_output["encoded_audio_all_layers"], dim=2)
        audio_len = audio_encoder_output["audio_samples"] // 640
        audio_emb = audio_emb[:, : audio_len * 2]
        feat0 = _wan_sound_linear_interpolation(audio_emb[:, :, 0:8].mean(dim=2), 50, 25)
        feat1 = _wan_sound_linear_interpolation(audio_emb[:, :, 8:16].mean(dim=2), 50, 25)
        feat2 = _wan_sound_linear_interpolation(audio_emb[:, :, 16:24].mean(dim=2), 50, 25)
        feat3 = _wan_sound_linear_interpolation(audio_emb[:, :, 24:32].mean(dim=2), 50, 25)
        feat4 = _wan_sound_linear_interpolation(audio_emb[:, :, 32], 50, 25)
        audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]
        audio_emb, _ = _wan_humo_get_audio_emb_window(audio_emb, length, frame0_idx=0)
        audio_emb = audio_emb.unsqueeze(0)
        positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_emb})
        negative = node_helpers.conditioning_set_values(negative, {"audio_embed": torch.zeros_like(audio_emb)})
    else:
        zero_audio = torch.zeros([batch_size, latent_t + 1, 8, 5, 1280], device=model_management.intermediate_device())
        positive = node_helpers.conditioning_set_values(positive, {"audio_embed": zero_audio})
        negative = node_helpers.conditioning_set_values(negative, {"audio_embed": zero_audio})

    return positive, negative, {"samples": latent}


# ── WAN Animate to Video ─────────────────────────────────────────────────────


def wan_animate_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 832,
    height: int = 480,
    length: int = 77,
    batch_size: int = 1,
    continue_motion_max_frames: int = 5,
    video_frame_offset: int = 0,
    *,
    clip_vision_output: Any | None = None,
    reference_image: Any | None = None,
    face_video: Any | None = None,
    pose_video: Any | None = None,
    continue_motion: Any | None = None,
    background_video: Any | None = None,
    character_mask: Any | None = None,
) -> tuple[Any, Any, dict[str, Any], int, int, int]:
    """Build WAN Animate conditioning (experimental).

    Returns ``(positive, negative, latent, trim_latent, trim_image, video_frame_offset_new)``.
    """
    torch, model_management, comfy_utils, node_helpers, _ = _get_wan_vace_dependencies()

    trim_to_pose_video = False
    latent_length = ((length - 1) // 4) + 1
    latent_width = width // 8
    latent_height = height // 8
    trim_latent = 0

    if reference_image is None:
        reference_image = torch.zeros((1, height, width, 3))

    image = comfy_utils.common_upscale(reference_image[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
    concat_latent_image = vae.encode(image[:, :, :, :3])
    mask = torch.zeros(
        (1, 4, concat_latent_image.shape[-3], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
        device=concat_latent_image.device, dtype=concat_latent_image.dtype,
    )
    trim_latent += concat_latent_image.shape[2]
    ref_motion_latent_length = 0

    if continue_motion is None:
        image = torch.ones((length, height, width, 3)) * 0.5
    else:
        continue_motion = continue_motion[-continue_motion_max_frames:]
        video_frame_offset = max(0, video_frame_offset - continue_motion.shape[0])
        continue_motion = comfy_utils.common_upscale(continue_motion[-length:].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
        image = torch.ones((length, height, width, continue_motion.shape[-1]), device=continue_motion.device, dtype=continue_motion.dtype) * 0.5
        image[: continue_motion.shape[0]] = continue_motion
        ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    if pose_video is not None:
        if pose_video.shape[0] <= video_frame_offset:
            pose_video = None
        else:
            pose_video = pose_video[video_frame_offset:]

    if pose_video is not None:
        pose_video = comfy_utils.common_upscale(pose_video[:length].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)
        if not trim_to_pose_video and pose_video.shape[0] < length:
            pose_video = torch.cat((pose_video,) + (pose_video[-1:],) * (length - pose_video.shape[0]), dim=0)
        pose_video_latent = vae.encode(pose_video[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(positive, {"pose_video_latent": pose_video_latent})
        negative = node_helpers.conditioning_set_values(negative, {"pose_video_latent": pose_video_latent})
        if trim_to_pose_video:
            latent_length = pose_video_latent.shape[2]
            length = latent_length * 4 - 3
            image = image[:length]

    if face_video is not None:
        if face_video.shape[0] <= video_frame_offset:
            face_video = None
        else:
            face_video = face_video[video_frame_offset:]

    if face_video is not None:
        face_video = (comfy_utils.common_upscale(face_video[:length].movedim(-1, 1), 512, 512, "area", "center") * 2.0 - 1.0).movedim(0, 1).unsqueeze(0)
        positive = node_helpers.conditioning_set_values(positive, {"face_video_pixels": face_video})
        negative = node_helpers.conditioning_set_values(negative, {"face_video_pixels": face_video * 0.0 - 1.0})

    ref_images_num = max(0, ref_motion_latent_length * 4 - 3)
    if background_video is not None and background_video.shape[0] > video_frame_offset:
        background_video = comfy_utils.common_upscale(
            background_video[video_frame_offset:][:length].movedim(-1, 1), width, height, "area", "center"
        ).movedim(1, -1)
        if background_video.shape[0] > ref_images_num:
            image[ref_images_num: background_video.shape[0]] = background_video[ref_images_num:]

    mask_refmotion = torch.ones(
        (1, 1, latent_length * 4, concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
        device=mask.device, dtype=mask.dtype,
    )
    if continue_motion is not None:
        mask_refmotion[:, :, : ref_motion_latent_length * 4] = 0.0

    if character_mask is not None:
        if character_mask.shape[0] > video_frame_offset or character_mask.shape[0] == 1:
            if character_mask.shape[0] == 1:
                character_mask = character_mask.repeat((length,) + (1,) * (character_mask.ndim - 1))
            else:
                character_mask = character_mask[video_frame_offset:]
            if character_mask.ndim == 3:
                character_mask = character_mask.unsqueeze(1).movedim(0, 1)
            if character_mask.ndim == 4:
                character_mask = character_mask.unsqueeze(1)
            character_mask = comfy_utils.common_upscale(character_mask[:, :, :length], concat_latent_image.shape[-1], concat_latent_image.shape[-2], "nearest-exact", "center")
            if character_mask.shape[2] > ref_images_num:
                mask_refmotion[:, :, ref_images_num: character_mask.shape[2]] = character_mask[:, :, ref_images_num:]

    concat_latent_image = torch.cat((concat_latent_image, vae.encode(image[:, :, :, :3])), dim=2)
    mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4, mask_refmotion.shape[3], mask_refmotion.shape[4]).transpose(1, 2)
    mask = torch.cat((mask, mask_refmotion), dim=2)
    positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
    negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

    latent = torch.zeros([batch_size, 16, latent_length + trim_latent, latent_height, latent_width], device=model_management.intermediate_device())
    return positive, negative, {"samples": latent}, trim_latent, max(0, ref_motion_latent_length * 4 - 3), video_frame_offset + length


# ── WAN Infinite Talk to Video ───────────────────────────────────────────────


def wan_infinite_talk_to_video(
    model: Any,
    model_patch: Any,
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int,
    height: int,
    length: int,
    audio_encoder_output_1: Any,
    *,
    mode: str = "single_speaker",
    start_image: Any | None = None,
    clip_vision_output: Any | None = None,
    audio_encoder_output_2: Any | None = None,
    mask_1: Any | None = None,
    mask_2: Any | None = None,
    motion_frame_count: int = 9,
    audio_scale: float = 1.0,
    previous_frames: Any | None = None,
) -> tuple[Any, Any, Any, dict[str, Any], int]:
    """Build WAN Infinite Talk conditioning with model patching.

    Returns ``(model_patched, positive, negative, latent, trim_image)``.
    ``mode`` is ``"single_speaker"`` or ``"two_speakers"``.
    """
    import logging

    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.utils
    import node_helpers
    from comfy.ldm.wan.model_multitalk import (
        InfiniteTalkOuterSampleWrapper,
        MultiTalkCrossAttnPatch,
        MultiTalkGetAttnMapPatch,
        project_audio_features,
    )

    if previous_frames is not None and previous_frames.shape[0] < motion_frame_count:
        raise ValueError("Not enough previous frames provided.")
    if mode == "two_speakers" and (audio_encoder_output_2 is None or mask_1 is None or mask_2 is None):
        raise ValueError("audio_encoder_output_2, mask_1, and mask_2 are required in two_speakers mode.")
    if audio_encoder_output_2 is not None and (mask_1 is None or mask_2 is None):
        raise ValueError("Both mask_1 and mask_2 must be provided when using two audio encoder outputs.")

    ref_masks = None
    if mask_1 is not None and mask_2 is not None:
        if audio_encoder_output_2 is None:
            raise ValueError("Second audio encoder output must be provided if two masks are used.")
        ref_masks = torch.cat([mask_1, mask_2])

    latent = torch.zeros([1, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
    concat_latent_image = None
    if start_image is not None:
        start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
        image[: start_image.shape[0]] = start_image
        concat_latent_image = vae.encode(image[:, :, :, :3])
        concat_mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
        concat_mask[:, :, : ((start_image.shape[0] - 1) // 4) + 1] = 0.0
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    model_patched = model.clone()
    encoded_audio_list = []
    seq_lengths = []
    for aeo in [audio_encoder_output_1, audio_encoder_output_2]:
        if aeo is None:
            continue
        encoded_audio = torch.stack(aeo["encoded_audio_all_layers"], dim=0).squeeze(1)[1:]
        encoded_audio = _wan_sound_linear_interpolation(encoded_audio, input_fps=50, output_fps=25).movedim(0, 1)
        encoded_audio_list.append(encoded_audio)
        seq_lengths.append(encoded_audio.shape[0])

    if len(encoded_audio_list) > 1:
        total_len = sum(seq_lengths)
        full_list = []
        offset = 0
        for emb, seq_len in zip(encoded_audio_list, seq_lengths):
            full = torch.zeros(total_len, *emb.shape[1:], dtype=emb.dtype)
            full[offset: offset + seq_len] = emb
            full_list.append(full)
            offset += seq_len
        encoded_audio_list = full_list

    token_ref_target_masks = None
    if ref_masks is not None:
        token_ref_target_masks = torch.nn.functional.interpolate(
            ref_masks.unsqueeze(0), size=(latent.shape[-2] // 2, latent.shape[-1] // 2), mode="nearest"
        )[0]
        token_ref_target_masks = (token_ref_target_masks > 0).view(token_ref_target_masks.shape[0], -1)

    if previous_frames is not None:
        motion_frames = comfy.utils.common_upscale(previous_frames[-motion_frame_count:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        frame_offset = previous_frames.shape[0] - motion_frame_count
        audio_start, audio_end = frame_offset, frame_offset + length
        logging.info("InfiniteTalk: Processing audio frames %d - %d", audio_start, audio_end)
        motion_frames_latent = vae.encode(motion_frames[:, :, :, :3])
        trim_image = motion_frame_count
    else:
        audio_start, audio_end, trim_image = 0, length, 0
        motion_frames_latent = concat_latent_image[:, :, :1] if concat_latent_image is not None else torch.zeros([1, 16, 1, height // 8, width // 8])

    audio_embed = project_audio_features(model_patch.model.audio_proj, encoded_audio_list, audio_start, audio_end).to(model_patched.model_dtype())
    model_patched.model_options["transformer_options"]["audio_embeds"] = audio_embed
    model_patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        "infinite_talk_outer_sample",
        InfiniteTalkOuterSampleWrapper(motion_frames_latent, model_patch, is_extend=previous_frames is not None),
    )
    model_patched.set_model_patch(MultiTalkCrossAttnPatch(model_patch, audio_scale), "attn2_patch")
    if token_ref_target_masks is not None:
        model_patched.set_model_patch(MultiTalkGetAttnMapPatch(token_ref_target_masks), "attn1_patch")

    return model_patched, positive, negative, {"samples": latent}, trim_image


# ── WAN SCAIL to Video ───────────────────────────────────────────────────────


def wan_scail_to_video(
    positive: Any,
    negative: Any,
    vae: _VaeEncoder,
    width: int = 512,
    height: int = 896,
    length: int = 81,
    batch_size: int = 1,
    pose_strength: float = 1.0,
    pose_start: float = 0.0,
    pose_end: float = 1.0,
    *,
    clip_vision_output: Any | None = None,
    reference_image: Any | None = None,
    pose_video: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build WAN SCAIL pose-driven conditioning (experimental)."""
    torch, model_management, comfy_utils, node_helpers, _ = _get_wan_vace_dependencies()

    latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=model_management.intermediate_device())

    ref_latent = None
    if reference_image is not None:
        reference_image = comfy_utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        ref_latent = vae.encode(reference_image[:, :, :, :3])

    if ref_latent is not None:
        positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)

    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

    if pose_video is not None:
        pose_video = comfy_utils.common_upscale(pose_video[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
        pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
        positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)
        negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)

    return positive, negative, {"samples": latent}


# ── WAN 2.2 Image-to-Video Latent ────────────────────────────────────────────


def wan22_image_to_video_latent(
    vae: _VaeEncoder,
    width: int = 1280,
    height: int = 704,
    length: int = 49,
    batch_size: int = 1,
    *,
    start_image: Any | None = None,
) -> dict[str, Any]:
    """Create WAN 2.2 image-to-video latent with optional start-image encoding."""
    torch, model_management, comfy_utils, _, comfy_latent_formats = _get_wan_vace_dependencies()

    latent = torch.zeros([1, 48, ((length - 1) // 4) + 1, height // 16, width // 16], device=model_management.intermediate_device())

    if start_image is None:
        return {"samples": latent}

    mask = torch.ones([latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]], device=model_management.intermediate_device())
    start_image = comfy_utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    latent_temp = vae.encode(start_image)
    latent[:, :, : latent_temp.shape[-3]] = latent_temp
    mask[:, :, : latent_temp.shape[-3]] *= 0.0
    latent_fmt = comfy_latent_formats.Wan22()
    latent = latent_fmt.process_out(latent) * mask + latent * (1.0 - mask)
    return {
        "samples": latent.repeat((batch_size,) + (1,) * (latent.ndim - 1)),
        "noise_mask": mask.repeat((batch_size,) + (1,) * (mask.ndim - 1)),
    }


def _get_qwen_image_edit_plus_dependencies() -> tuple[Any, Any, Any]:
    """Resolve math, comfy.utils, and node_helpers at call time."""
    import math as _math

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.utils
    import node_helpers

    return _math, comfy.utils, node_helpers


def _get_node_helpers() -> Any:
    """Resolve node_helpers at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import node_helpers

    return node_helpers


def _get_reference_latent_type() -> Any:
    """Resolve ComfyUI ReferenceLatent node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_edit_model import ReferenceLatent

    return ReferenceLatent


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def reference_latent(conditioning: list, latent: dict) -> list:
    """Inject a reference-image latent into conditioning.

    Wraps ``ReferenceLatent.execute()`` from ``comfy_extras.nodes_edit_model``.
    Used by Flux.2 Klein editing and Qwen Image Layered workflows to attach a
    guiding latent to the conditioning list.

    Args:
        conditioning: Conditioning list to augment.
        latent: Latent dict with a ``"samples"`` tensor (standard ComfyUI latent).

    Returns:
        New conditioning list with ``reference_latents`` appended.
    """
    node_type = _get_reference_latent_type()
    return _unwrap_node_output(node_type.execute(conditioning=conditioning, latent=latent))


def encode_qwen_image_edit_plus(
    clip: Any,
    vae: Any,
    image1: Any,
    image2: Any | None = None,
    image3: Any | None = None,
    prompt: str = "",
) -> Any:
    """Encode a text prompt together with up to three reference images for Qwen image editing.

    Mirrors ``TextEncodeQwenImageEditPlus.execute()`` from
    ``comfy_extras/nodes_qwen.py``.

    Args:
        clip: ComfyUI CLIP object with Qwen tokenizer support.
        vae: VAE used to encode reference image latents.
        image1: First reference image tensor (BHWC).
        image2: Optional second reference image tensor.
        image3: Optional third reference image tensor.
        prompt: Text editing instruction.

    Returns:
        Conditioning list with reference latents attached.
    """
    math, comfy_utils, node_helpers = _get_qwen_image_edit_plus_dependencies()

    llama_template = (
        "<|im_start|>system\nDescribe the key features of the input image "
        "(color, shape, size, texture, objects, background), then explain how the "
        "user's text instruction should alter or modify the image. Generate a new "
        "image that meets the user's requirements while maintaining consistency with "
        "the original input where appropriate.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )

    ref_latents: list[Any] = []
    images_vl: list[Any] = []
    image_prompt = ""

    for i, image in enumerate([image1, image2, image3]):
        if image is None:
            continue
        samples = image.movedim(-1, 1)
        total_vl = int(384 * 384)
        scale_by = math.sqrt(total_vl / (samples.shape[3] * samples.shape[2]))
        w_vl = round(samples.shape[3] * scale_by)
        h_vl = round(samples.shape[2] * scale_by)
        s_vl = comfy_utils.common_upscale(samples, w_vl, h_vl, "area", "disabled")
        images_vl.append(s_vl.movedim(1, -1))

        if vae is not None:
            total_enc = int(1024 * 1024)
            scale_by_enc = math.sqrt(total_enc / (samples.shape[3] * samples.shape[2]))
            w_enc = round(samples.shape[3] * scale_by_enc / 8.0) * 8
            h_enc = round(samples.shape[2] * scale_by_enc / 8.0) * 8
            s_enc = comfy_utils.common_upscale(samples, w_enc, h_enc, "area", "disabled")
            ref_latents.append(vae.encode(s_enc.movedim(1, -1)[:, :, :, :3]))

        image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

    tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_latents:
        conditioning = node_helpers.conditioning_set_values(
            conditioning, {"reference_latents": ref_latents}, append=True
        )
    return conditioning


def apply_flux_kontext_multi_reference(
    conditioning: Any,
    reference_latents_method: str = "index_timestep_zero",
) -> Any:
    """Set the reference-latent blending method for Flux Kontext multi-reference.

    Mirrors ``FluxKontextMultiReferenceLatentMethod.execute()`` from
    ``comfy_extras/nodes_flux.py``.

    Args:
        conditioning: Conditioning list produced by a text or image encoder.
        reference_latents_method: One of ``"offset"``, ``"index"``,
            ``"uxo/uno"``, or ``"index_timestep_zero"`` (default).
            Any value containing ``"uxo"`` or ``"uso"`` is normalised to
            ``"uxo"`` internally.

    Returns:
        New conditioning list with ``reference_latents_method`` set.
    """
    node_helpers = _get_node_helpers()
    if "uxo" in reference_latents_method or "uso" in reference_latents_method:
        reference_latents_method = "uxo"
    return node_helpers.conditioning_set_values(
        conditioning, {"reference_latents_method": reference_latents_method}
    )


__all__ = [
    "encode_prompt",
    "encode_prompt_flux",
    "encode_clip_vision",
    "wan_image_to_video",
    "wan_first_last_frame_to_video",
    "wan_vace_to_video",
    "wan_fun_control_to_video",
    "wan22_fun_control_to_video",
    "wan_fun_inpaint_to_video",
    "wan_camera_embedding",
    "wan_camera_image_to_video",
    "wan_phantom_subject_to_video",
    "wan_track_to_video",
    "wan_sound_image_to_video",
    "wan_sound_image_to_video_extend",
    "wan_humo_image_to_video",
    "wan_animate_to_video",
    "wan_infinite_talk_to_video",
    "wan_scail_to_video",
    "wan22_image_to_video_latent",
    "ltxv_img_to_video",
    "ltxv_conditioning",
    "ltxv_crop_guides",
    "void_quadmask_preprocess",
    "void_inpaint_conditioning",
    "ltx_add_video_ic_lora_guide",
    "conditioning_zero_out",
    "conditioning_combine",
    "conditioning_set_mask",
    "conditioning_set_timestep_range",
    "flux_guidance",
    "reference_latent",
    "encode_qwen_image_edit_plus",
    "apply_flux_kontext_multi_reference",
]
