"""Latent image helpers."""

from __future__ import annotations

from typing import Any, cast

_SUPPORTED_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
_SUPPORTED_CONCAT_DIMS = ("x", "-x", "y", "-y", "t", "-t")


def _get_empty_latent_image_type() -> Any:
    """Resolve ComfyUI EmptyLatentImage node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.EmptyLatentImage


def _get_latent_upscale_type() -> Any:
    """Resolve ComfyUI LatentUpscale node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentUpscale


def _get_latent_upscale_by_type() -> Any:
    """Resolve ComfyUI LatentUpscaleBy node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentUpscaleBy


def _get_latent_crop_type() -> Any:
    """Resolve ComfyUI LatentCrop node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentCrop


def _get_latent_from_batch_type() -> Any:
    """Resolve ComfyUI LatentFromBatch node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentFromBatch


def _get_repeat_latent_batch_type() -> Any:
    """Resolve ComfyUI RepeatLatentBatch node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.RepeatLatentBatch


def _get_latent_composite_type() -> Any:
    """Resolve ComfyUI LatentComposite node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import nodes

    return nodes.LatentComposite


def _get_latent_composite_masked_type() -> Any:
    """Resolve ComfyUI LatentCompositeMasked node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_mask import LatentCompositeMasked

    return LatentCompositeMasked


def _get_latent_concat_type() -> Any:
    """Resolve ComfyUI LatentConcat node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy_extras.nodes_post_processing  # noqa: F401
    from comfy_extras.nodes_latent import LatentConcat

    return LatentConcat


def _get_latent_cut_type() -> Any:
    """Resolve ComfyUI LatentCut node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy_extras.nodes_post_processing  # noqa: F401
    from comfy_extras.nodes_latent import LatentCut

    return LatentCut


def _get_replace_video_latent_frames_type() -> Any:
    """Resolve ComfyUI ReplaceVideoLatentFrames node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy_extras.nodes_post_processing  # noqa: F401
    from comfy_extras.nodes_latent import ReplaceVideoLatentFrames

    return ReplaceVideoLatentFrames


def _get_torch_tensor_type() -> Any:
    """Resolve torch.Tensor at call time."""
    import torch

    return torch.Tensor


def _get_torch_module() -> Any:
    """Resolve torch at call time."""
    import torch

    return torch


def _get_node_helpers_module() -> Any:
    """Resolve node_helpers at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import node_helpers

    return node_helpers


def _unwrap_node_output(output: Any) -> Any:
    """Return first output for ComfyUI V3 nodes and tuple-style APIs."""
    if hasattr(output, "result"):
        return output.result[0]
    if isinstance(output, tuple):
        return output[0]
    return output


def _validate_upscale_method(method: str) -> None:
    """Validate latent upscale methods against the ComfyUI set."""
    if method not in _SUPPORTED_UPSCALE_METHODS:
        raise ValueError(
            f"method must be one of {list(_SUPPORTED_UPSCALE_METHODS)!r}; got {method!r}"
        )


def _validate_concat_dim(dim: str) -> None:
    """Validate latent concat dimensions against the ComfyUI set."""
    if dim not in _SUPPORTED_CONCAT_DIMS:
        raise ValueError(f"dim must be one of {list(_SUPPORTED_CONCAT_DIMS)!r}; got {dim!r}")


def empty_latent_image(width: int, height: int, batch_size: int = 1) -> dict[str, Any]:
    """Create empty image latents compatible with ComfyUI samplers."""
    empty_latent_image_type = _get_empty_latent_image_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            empty_latent_image_type().generate(width=width, height=height, batch_size=batch_size)
        ),
    )


def latent_upscale(
    latent: dict[str, Any], method: str, width: int, height: int
) -> dict[str, Any]:
    """Upscale a LATENT dict to target pixel dimensions."""
    _validate_upscale_method(method)
    latent_upscale_type = _get_latent_upscale_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_upscale_type().upscale(
                samples=latent,
                upscale_method=method,
                width=width,
                height=height,
                crop="disabled",
            )
        ),
    )


def latent_upscale_by(latent: dict[str, Any], method: str, scale_by: float) -> dict[str, Any]:
    """Upscale a LATENT dict by floating-point factor."""
    _validate_upscale_method(method)
    latent_upscale_by_type = _get_latent_upscale_by_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_upscale_by_type().upscale(
                samples=latent,
                upscale_method=method,
                scale_by=scale_by,
            )
        ),
    )


def latent_crop(
    latent: dict[str, Any], x: int, y: int, width: int, height: int
) -> dict[str, Any]:
    """Crop a LATENT dict using pixel-space coordinates and dimensions."""
    latent_crop_type = _get_latent_crop_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_crop_type().crop(samples=latent, width=width, height=height, x=x, y=y)
        ),
    )


def latent_from_batch(
    latent: dict[str, Any], batch_index: int, length: int = 1
) -> dict[str, Any]:
    """Extract a contiguous subset from the latent batch dimension."""
    latent_from_batch_type = _get_latent_from_batch_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_from_batch_type().frombatch(
                samples=latent,
                batch_index=batch_index,
                length=length,
            )
        ),
    )


def repeat_latent_batch(latent: dict[str, Any], amount: int) -> dict[str, Any]:
    """Repeat latent samples along the batch dimension."""
    repeat_latent_batch_type = _get_repeat_latent_batch_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(repeat_latent_batch_type().repeat(samples=latent, amount=amount)),
    )


def latent_concat(*latents: dict[str, Any], dim: str = "t") -> dict[str, Any]:
    """Concatenate multiple LATENT dicts across x/y/t dimensions."""
    if len(latents) < 2:
        raise ValueError("at least two latents are required")

    _validate_concat_dim(dim)
    latent_concat_type = _get_latent_concat_type()

    result = latents[0]
    for next_latent in latents[1:]:
        result = cast(
            dict[str, Any],
            _unwrap_node_output(
                latent_concat_type.execute(samples1=result, samples2=next_latent, dim=dim)
            ),
        )
    return result


def latent_cut_to_batch(latent: dict[str, Any], start: int, length: int) -> dict[str, Any]:
    """Extract a contiguous segment from the latent batch dimension."""
    return latent_from_batch(latent=latent, batch_index=start, length=length)


def latent_cut(
    latent: dict[str, Any],
    dim: str = "t",
    index: int = 0,
    amount: int = 1,
) -> dict[str, Any]:
    """Extract a contiguous segment from a LATENT dict along the given dimension.

    Wraps ComfyUI's ``LatentCut`` node.  The ``dim`` parameter accepts ``"x"``,
    ``"y"``, or ``"t"`` for spatial-width, spatial-height, or temporal dimensions
    respectively.

    Args:
        latent: Source LATENT dict.
        dim: Dimension to cut along — ``"x"`` (width), ``"y"`` (height), or
            ``"t"`` (temporal).  Default ``"t"``.
        index: Start position along the chosen dimension (0-indexed).  Negative
            values count from the end.  Default ``0``.
        amount: Number of frames/slices to extract.  Default ``1``.

    Returns:
        A LATENT dict containing only the extracted segment.
    """
    latent_cut_type = _get_latent_cut_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_cut_type.execute(samples=latent, dim=dim, index=index, amount=amount)
        ),
    )


def replace_video_latent_frames(
    latent: dict[str, Any], replacement: dict[str, Any], start_frame: int
) -> dict[str, Any]:
    """Replace latent video frames from `start_frame` using replacement samples."""
    replace_video_latent_frames_type = _get_replace_video_latent_frames_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            replace_video_latent_frames_type.execute(
                destination=latent,
                source=replacement,
                index=start_frame,
            )
        ),
    )


def latent_composite(
    destination: dict[str, Any], source: dict[str, Any], x: int, y: int
) -> dict[str, Any]:
    """Composite source LATENT onto destination LATENT at pixel-space coordinates."""
    latent_composite_type = _get_latent_composite_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_composite_type().composite(
                samples_to=destination,
                samples_from=source,
                x=x,
                y=y,
                feather=0,
            )
        ),
    )


def latent_composite_masked(
    destination: dict[str, Any], source: dict[str, Any], mask: Any, x: int = 0, y: int = 0
) -> dict[str, Any]:
    """Composite source LATENT onto destination LATENT with mask-guided blending."""
    latent_composite_masked_type = _get_latent_composite_masked_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            latent_composite_masked_type().composite(
                destination=destination,
                source=source,
                x=x,
                y=y,
                resize_source=False,
                mask=mask,
            )
        ),
    )


def ltxv_empty_latent_video(
    width: int,
    height: int,
    length: int = 97,
    batch_size: int = 1,
    fps: int = 24,
) -> dict[str, Any]:
    """Create empty LTXV video latents compatible with the LTX-Video model.

    Parameters
    ----------
    width : int
        Frame width in pixels (must be divisible by 32).
    height : int
        Frame height in pixels (must be divisible by 32).
    length : int, optional
        Number of video frames (must satisfy ``(length - 1) % 8 == 0``).
        Default ``97``.
    batch_size : int, optional
        Batch size. Default ``1``.
    fps : int, optional
        Target frame rate. Accepted for API consistency and future scheduler
        use; does not affect the latent tensor shape. Default ``24``.
    """
    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management

    latent = torch.zeros(
        [batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": latent}


def _get_empty_flux2_latent_image_type() -> Any:
    """Resolve ComfyUI EmptyFlux2LatentImage node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage

    return EmptyFlux2LatentImage


def _get_empty_qwen_image_layered_latent_image_type() -> Any:
    """Resolve ComfyUI EmptyQwenImageLayeredLatentImage node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_qwen import EmptyQwenImageLayeredLatentImage

    return EmptyQwenImageLayeredLatentImage


def _load_latent_upscale_model() -> Any:
    """Resolve LTXVLatentUpsampler node at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

    return LTXVLatentUpsampler


def _upsample_latent(samples: dict[str, Any], upscale_model: Any, vae: Any) -> dict[str, Any]:
    """Delegate to LTXVLatentUpsampler.upsample_latent."""
    ltxv_upsampler_type = _load_latent_upscale_model()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            ltxv_upsampler_type().upsample_latent(
                samples=samples,
                upscale_model=upscale_model,
                vae=vae,
            )
        ),
    )


def ltxv_latent_upsample(
    samples: dict[str, Any], upscale_model: Any, vae: Any
) -> dict[str, Any]:
    """Upsample a video latent by factor 2 in latent space using LTXVLatentUpsampler."""
    return _upsample_latent(samples=samples, upscale_model=upscale_model, vae=vae)


def set_latent_noise_mask(latent: dict[str, Any], mask: Any) -> dict[str, Any]:
    """Return a LATENT dict with noise mask metadata for inpainting sampling."""
    torch_tensor_type = _get_torch_tensor_type()
    if not isinstance(mask, torch_tensor_type):
        raise TypeError("mask must be a torch.Tensor")

    latent_with_noise_mask = dict(latent)
    latent_with_noise_mask["noise_mask"] = mask
    return latent_with_noise_mask


def inpaint_model_conditioning(
    model: Any,
    latent: dict[str, Any],
    vae: Any,
    positive: Any,
    negative: Any,
) -> tuple[Any, Any, Any]:
    """Patch model and conditioning with inpaint latent metadata."""
    node_helpers = _get_node_helpers_module()

    latent_image = latent["samples"]
    mask = latent.get("noise_mask")

    if mask is None:
        torch = _get_torch_module()
        downscale_ratio = vae.spacial_compression_encode()
        mask = torch.ones(
            (
                latent_image.shape[0],
                1,
                latent_image.shape[-2] * downscale_ratio,
                latent_image.shape[-1] * downscale_ratio,
            ),
            dtype=latent_image.dtype,
            device=latent_image.device,
        )

    values = {"concat_latent_image": latent_image, "concat_mask": mask}
    patched_positive = node_helpers.conditioning_set_values(positive, values)
    patched_negative = node_helpers.conditioning_set_values(negative, values)

    patched_model = model.clone() if hasattr(model, "clone") else model
    return patched_model, patched_positive, patched_negative


def empty_wan_latent_video(
    width: int,
    height: int,
    length: int = 33,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Create empty WAN video latents compatible with WAN models.

    Parameters
    ----------
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    length : int, optional
        Number of video frames. Default ``33``.
    batch_size : int, optional
        Batch size. Default ``1``.

    Returns
    -------
    dict
        ``{"samples": tensor}`` with shape
        ``[batch_size, 16, ((length-1)//4)+1, height//8, width//8]``.
    """
    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management

    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": latent}


def wan22_image_to_video_latent(
    vae: Any,
    width: int,
    height: int,
    length: int = 49,
    batch_size: int = 1,
    *,
    start_image: Any = None,
) -> dict[str, Any]:
    """Create TI2V latents for WAN 2.2 image-to-video conditioning.

    Parameters
    ----------
    vae : Any
        ComfyUI VAE model used to encode the start image.
    width : int
        Frame width in pixels.
    height : int
        Frame height in pixels.
    length : int, optional
        Number of video frames. Default ``49``.
    batch_size : int, optional
        Batch size. Default ``1``.
    start_image : Any, optional
        Optional image tensor to use as the first frame.

    Returns
    -------
    dict
        ``{"samples": tensor}`` with shape
        ``[batch_size, 48, ((length-1)//4)+1, height//16, width//16]``.
        When *start_image* is provided, also includes ``"noise_mask"``.
    """
    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.latent_formats
    import comfy.model_management
    import comfy.utils

    latent = torch.zeros(
        [1, 48, ((length - 1) // 4) + 1, height // 16, width // 16],
        device=comfy.model_management.intermediate_device(),
    )

    if start_image is None:
        return {"samples": latent}

    mask = torch.ones(
        [latent.shape[0], 1, ((length - 1) // 4) + 1, latent.shape[-2], latent.shape[-1]],
        device=comfy.model_management.intermediate_device(),
    )

    start_image = comfy.utils.common_upscale(
        start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    latent_temp = vae.encode(start_image)
    latent[:, :, : latent_temp.shape[-3]] = latent_temp
    mask[:, :, : latent_temp.shape[-3]] *= 0.0

    latent_format = comfy.latent_formats.Wan22()
    latent = latent_format.process_out(latent) * mask + latent * (1.0 - mask)

    out_latent: dict[str, Any] = {}
    out_latent["samples"] = latent.repeat((batch_size,) + (1,) * (latent.ndim - 1))
    out_latent["noise_mask"] = mask.repeat((batch_size,) + (1,) * (mask.ndim - 1))
    return out_latent


def empty_sd3_latent_image(width: int, height: int, batch_size: int = 1) -> dict[str, Any]:
    """Create empty SD3/AuraFlow/Lumina2 image latents (16-channel, factor-8 spatial).

    Mirrors ``EmptySD3LatentImage.execute()`` from ComfyUI.  The returned dict
    contains ``"samples"`` (a zero-filled 16-channel tensor) and the
    ``"downscale_ratio_spacial"`` metadata key expected by SD3-family models.

    Parameters
    ----------
    width : int
        Target image width in pixels.  Must be divisible by 8.
    height : int
        Target image height in pixels.  Must be divisible by 8.
    batch_size : int, optional
        Batch size.  Default ``1``.
    """
    import torch

    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.model_management

    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def empty_flux2_latent_image(width: int, height: int, batch_size: int = 1) -> dict[str, Any]:
    """Create empty Flux.2 Klein image latents (128-channel, factor-16 spatial).

    Wraps ``EmptyFlux2LatentImage.execute()`` from ``comfy_extras.nodes_flux``.

    Parameters
    ----------
    width : int
        Target image width in pixels.  Must be divisible by 16.
    height : int
        Target image height in pixels.  Must be divisible by 16.
    batch_size : int, optional
        Batch size.  Default ``1``.

    Returns
    -------
    dict
        ``{"samples": tensor}`` with shape
        ``[batch_size, 128, height // 16, width // 16]``.
    """
    node_type = _get_empty_flux2_latent_image_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(node_type.execute(width=width, height=height, batch_size=batch_size)),
    )


def empty_qwen_image_layered_latent_image(
    width: int, height: int, layers: int, batch_size: int = 1
) -> dict[str, Any]:
    """Create empty Qwen Image Layered latents for Qwen Image Layered models.

    Wraps ``EmptyQwenImageLayeredLatentImage.execute()`` from
    ``comfy_extras.nodes_qwen``.

    Parameters
    ----------
    width : int
        Target image width in pixels.  Must be divisible by 8.
    height : int
        Target image height in pixels.  Must be divisible by 8.
    layers : int
        Number of extra layers.  The tensor time dimension is ``layers + 1``.
    batch_size : int, optional
        Batch size.  Default ``1``.

    Returns
    -------
    dict
        ``{"samples": tensor}`` with shape
        ``[batch_size, 16, layers + 1, height // 8, width // 8]``.
    """
    node_type = _get_empty_qwen_image_layered_latent_image_type()
    return cast(
        dict[str, Any],
        _unwrap_node_output(
            node_type.execute(width=width, height=height, layers=layers, batch_size=batch_size)
        ),
    )


def trim_video_latent(latent: dict[str, Any], n_latent_frames: int) -> dict[str, Any]:
    """Trim the first *n_latent_frames* along the time axis of a 5-D video latent.

    Use this after sampling with :func:`wan_infinite_talk_to_video` to remove the
    motion-frame prefix.  Convert video frames to latent frames with
    ``((n_video_frames - 1) // temporal_compression) + 1`` before calling.
    """
    trimmed: dict[str, Any] = {"samples": latent["samples"][:, :, n_latent_frames:]}
    if "noise_mask" in latent:
        trimmed["noise_mask"] = latent["noise_mask"][:, :, n_latent_frames:]
    return trimmed


__all__ = [
    "empty_flux2_latent_image",
    "empty_latent_image",
    "empty_qwen_image_layered_latent_image",
    "empty_sd3_latent_image",
    "empty_wan_latent_video",
    "ltxv_empty_latent_video",
    "latent_from_batch",
    "latent_cut_to_batch",
    "latent_cut",
    "repeat_latent_batch",
    "latent_concat",
    "replace_video_latent_frames",
    "latent_upscale",
    "latent_upscale_by",
    "latent_crop",
    "latent_composite",
    "latent_composite_masked",
    "set_latent_noise_mask",
    "inpaint_model_conditioning",
    "ltxv_latent_upsample",
    "trim_video_latent",
    "wan22_image_to_video_latent",
]
