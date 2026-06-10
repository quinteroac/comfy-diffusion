"""Image loading helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from PIL import Image as PILImage


def _get_load_image_dependencies() -> tuple[Any, Any, Any]:
    import torch
    from PIL import Image, ImageOps

    return Image, ImageOps, torch


def _get_torch_module() -> Any:
    import torch

    return torch


def _get_image_upscale_with_model_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

    return ImageUpscaleWithModel


def _get_repeat_image_batch_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_images import RepeatImageBatch

    return RepeatImageBatch


def _get_image_from_batch_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_images import ImageFromBatch

    return ImageFromBatch


def _get_image_composite_masked_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_mask import ImageCompositeMasked

    return ImageCompositeMasked


def _get_resize_image_mask_node_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_post_processing import ResizeImageMaskNode

    return ResizeImageMaskNode


def _get_empty_image_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from nodes import EmptyImage

    return EmptyImage


def _get_canny_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_canny import Canny

    return Canny


def _get_image_invert_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from nodes import ImageInvert

    return ImageInvert


def _get_math_expression_node_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_math import MathExpressionNode

    return MathExpressionNode


def _get_resize_images_by_longer_edge_node_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_dataset import ResizeImagesByLongerEdgeNode

    return ResizeImagesByLongerEdgeNode


def _get_ltxv_preprocess_dependencies() -> tuple[Any, Any]:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.utils
    from comfy_extras.nodes_lt import LTXVPreprocess

    return comfy.utils, LTXVPreprocess


def _get_comfy_utils() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.utils

    return comfy.utils


def _get_image_scale_to_total_pixels_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels

    return ImageScaleToTotalPixels


def _get_image_scale_to_max_dimension_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_images import ImageScaleToMaxDimension

    return ImageScaleToMaxDimension


def _get_get_image_size_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_images import GetImageSize

    return GetImageSize


def _get_dw_preprocessor_deps() -> tuple[Any, Any, Any]:
    try:
        from controlnet_aux import DWposeDetector
    except ImportError as exc:
        raise ImportError(
            "controlnet_aux is required for dw_preprocessor. "
            "Install it with: pip install controlnet-aux"
        ) from exc
    import numpy as np
    import torch

    return DWposeDetector, torch, np


def _unwrap_node_output(output: Any) -> Any:
    result = getattr(output, "result", output)
    return result[0]


def _pixels_to_float_rows(image: Any, *, channels: int) -> list[list[list[float]]]:
    width, height = image.size
    pixels = image.load()
    if pixels is None:
        raise ValueError("unable to access image pixels")

    rows: list[list[list[float]]] = []
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            pixel = cast(tuple[int, ...], pixels[x, y])
            row.append([channel / 255.0 for channel in pixel[:channels]])
        rows.append(row)
    return rows


def _opaque_mask(height: int, width: int) -> list[list[float]]:
    return [[0.0 for _ in range(width)] for _ in range(height)]


def _alpha_to_mask_rows(rgba_image: Any) -> list[list[float]]:
    width, height = rgba_image.size
    pixels = rgba_image.load()
    if pixels is None:
        raise ValueError("unable to access image pixels")

    rows: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        for x in range(width):
            alpha = cast(tuple[int, int, int, int], pixels[x, y])[3]
            row.append((255 - alpha) / 255.0)
        rows.append(row)
    return rows


def load_image(path: str | Path) -> tuple[Any, Any]:
    """Load an image file into ComfyUI-compatible IMAGE/MASK tensors."""
    image_path = Path(path)
    Image, ImageOps, torch = _get_load_image_dependencies()

    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        rgb = oriented.convert("RGB")
        width, height = rgb.size
        image_rows = _pixels_to_float_rows(rgb, channels=3)

        has_alpha = "A" in oriented.getbands() or (
            oriented.mode == "P" and "transparency" in oriented.info
        )
        if has_alpha:
            mask_values = _alpha_to_mask_rows(oriented.convert("RGBA"))
        else:
            mask_values = _opaque_mask(height, width)

    image_tensor = torch.tensor([image_rows], dtype=torch.float32)
    mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    return image_tensor, mask_tensor


def image_pad_for_outpaint(
    image: Any, left: int, top: int, right: int, bottom: int, feathering: int
) -> tuple[Any, Any]:
    """Pad an image and return a corresponding outpaint mask."""
    if left < 0 or top < 0 or right < 0 or bottom < 0:
        raise ValueError("left, top, right, and bottom must be non-negative integers")
    if feathering < 0:
        raise ValueError("feathering must be a non-negative integer")

    torch = _get_torch_module()

    batch, height, width, channels = image.size()

    padded_image = torch.ones(
        (batch, height + top + bottom, width + left + right, channels),
        dtype=torch.float32,
    ) * 0.5
    padded_image[:, top : top + height, left : left + width, :] = image

    padded_mask = torch.ones(
        (height + top + bottom, width + left + right),
        dtype=torch.float32,
    )
    inner_mask = torch.zeros((height, width), dtype=torch.float32)

    if feathering > 0 and feathering * 2 < height and feathering * 2 < width:
        for row in range(height):
            for col in range(width):
                distance_top = row if top != 0 else height
                distance_bottom = height - row if bottom != 0 else height
                distance_left = col if left != 0 else width
                distance_right = width - col if right != 0 else width

                distance = min(distance_top, distance_bottom, distance_left, distance_right)
                if distance >= feathering:
                    continue

                blend = (feathering - distance) / feathering
                inner_mask[row, col] = blend * blend

    padded_mask[top : top + height, left : left + width] = inner_mask
    return padded_image, padded_mask.unsqueeze(0)


def image_upscale_with_model(upscale_model: Any, image: Any) -> Any:
    """Upscale a BHWC image tensor using a ComfyUI-compatible upscale model."""
    image_upscale_with_model_type = _get_image_upscale_with_model_type()
    return _unwrap_node_output(image_upscale_with_model_type.execute(upscale_model, image))


def image_from_batch(image: Any, batch_index: int, length: int = 1) -> Any:
    """Extract a contiguous subset from the image batch dimension."""
    image_from_batch_type = _get_image_from_batch_type()
    return _unwrap_node_output(
        image_from_batch_type.execute(image=image, batch_index=batch_index, length=length)
    )


def repeat_image_batch(image: Any, amount: int) -> Any:
    """Repeat image tensors along the batch dimension."""
    repeat_image_batch_type = _get_repeat_image_batch_type()
    return _unwrap_node_output(repeat_image_batch_type.execute(image=image, amount=amount))


def image_composite_masked(destination: Any, source: Any, mask: Any, x: int, y: int) -> Any:
    """Composite source image onto destination image with mask-guided blending."""
    image_composite_masked_type = _get_image_composite_masked_type()
    return _unwrap_node_output(
        image_composite_masked_type.execute(
            destination=destination,
            source=source,
            x=x,
            y=y,
            resize_source=False,
            mask=mask,
        )
    )


def image_to_tensor(image: PILImage.Image) -> Any:
    """Convert a PIL Image to a BHWC float32 tensor with shape (1, H, W, 3)."""
    torch = _get_torch_module()
    rgb = image.convert("RGB")
    rows = _pixels_to_float_rows(rgb, channels=3)
    return torch.tensor([rows], dtype=torch.float32)


def images_to_tensor(images: list[PILImage.Image]) -> Any:
    """Convert PIL images to a BHWC float32 tensor."""
    if not images:
        raise ValueError("images must not be empty")
    torch = _get_torch_module()
    rows = [_pixels_to_float_rows(image.convert("RGB"), channels=3) for image in images]
    return torch.tensor(rows, dtype=torch.float32)


def ltxv_preprocess(image: Any, width: int, height: int, img_compression: int = 35) -> Any:
    """Preprocess image batch for LTXV img2vid with center resize and node compression."""
    comfy_utils, ltxv_preprocess_type = _get_ltxv_preprocess_dependencies()
    resized = comfy_utils.common_upscale(
        image.movedim(-1, 1), width, height, "bilinear", "center"
    ).movedim(1, -1)
    return _unwrap_node_output(ltxv_preprocess_type.execute(resized, img_compression=img_compression))


def resize_image_mask(
    image: Any,
    mask: Any,
    width: int,
    height: int,
    interpolation: str = "bilinear",
) -> tuple[Any, Any]:
    """Resize an image and its mask to the given dimensions in a single call."""
    resize_image_mask_node_type = _get_resize_image_mask_node_type()
    result = resize_image_mask_node_type.execute(
        image=image,
        mask=mask,
        width=width,
        height=height,
        interpolation=interpolation,
    )
    raw = getattr(result, "result", result)
    return raw[0], raw[1]


def resize_images_by_longer_edge(images: Any, size: int) -> Any:
    """Resize images so the longer dimension equals `size`, preserving aspect ratio."""
    resize_images_by_longer_edge_node_type = _get_resize_images_by_longer_edge_node_type()
    return _unwrap_node_output(
        resize_images_by_longer_edge_node_type.execute(images=images, size=size)
    )


def empty_image(width: int, height: int, batch_size: int = 1, color: int = 0) -> Any:
    """Create a solid-color blank image tensor with shape (batch_size, height, width, 3)."""
    empty_image_type = _get_empty_image_type()
    return _unwrap_node_output(
        empty_image_type.execute(width=width, height=height, batch_size=batch_size, color=color)
    )


def canny(image: Any, low_threshold: int = 100, high_threshold: int = 200) -> Any:
    """Apply Canny edge detection and return an image tensor of the same spatial dimensions."""
    canny_type = _get_canny_type()
    return _unwrap_node_output(
        canny_type.execute(
            image=image,
            low_threshold=low_threshold / 255.0,
            high_threshold=high_threshold / 255.0,
        )
    )


def image_invert(image: Any) -> Any:
    """Invert pixel values (1 − pixel) of an image tensor."""
    image_invert_type = _get_image_invert_type()
    return _unwrap_node_output(image_invert_type.execute(image=image))


def math_expression(expression: str, **kwargs: float) -> int | float:
    """Evaluate a parameterised math expression via ComfyMathExpression."""
    math_expression_node_type = _get_math_expression_node_type()
    result = math_expression_node_type.execute(expression=expression, values=kwargs)
    raw = getattr(result, "result", result)
    return cast("int | float", raw[0])


def image_scale_by(image: Any, upscale_method: str = "lanczos", scale_by: float = 1.0) -> Any:
    """Scale an IMAGE tensor by a float factor using the specified upscale method.

    Output dimensions are ``floor(input_h * scale_by)`` × ``floor(input_w * scale_by)``.
    """
    comfy_utils = _get_comfy_utils()
    samples = image.movedim(-1, 1)
    width = math.floor(samples.shape[3] * scale_by)
    height = math.floor(samples.shape[2] * scale_by)
    scaled = comfy_utils.common_upscale(samples, width, height, upscale_method, "disabled")
    return scaled.movedim(1, -1)


def image_resize_kj(
    image: Any,
    width: int,
    height: int,
    upscale_method: str = "lanczos",
    keep_proportion: str = "crop",
    pad_color: str = "0, 0, 0",
    crop_position: str = "center",
    divisible_by: int = 2,
    device: str = "cpu",
) -> tuple[Any, int, int]:
    """Resize an image using KJ-style resizing.

    Mirrors ``ImageResizeKJv2.resize()`` from KJNodes (comfyui-kjnodes).

    ``keep_proportion`` options:

    - ``"stretch"`` — direct resize, ignoring aspect ratio.
    - ``"crop"`` — center-crop to match target aspect ratio, then scale.
    - ``"resize"`` — scale to fit within target bounds (letterbox, no pad).
    - ``"total_pixels"`` — scale to match ``width * height`` total pixels.
    - ``"pad"`` / ``"pad_edge"`` / ``"pad_edge_pixel"`` — scale to fit, then
      fill remaining space with ``pad_color`` / edge pixels respectively.

    The MASK (4th output of the node) is discarded.

    Returns ``(IMAGE, out_width: int, out_height: int)``.
    """
    comfy_utils = _get_comfy_utils()
    torch = _get_torch_module()

    B, H, W, C = image.shape

    target_w = width
    target_h = height
    pad_left = pad_right = pad_top = pad_bottom = 0

    is_pad_mode = keep_proportion.startswith("pad") or keep_proportion == "pillarbox_blur"

    if keep_proportion in ("resize", "total_pixels") or is_pad_mode:
        if keep_proportion == "total_pixels":
            total_pixels = target_w * target_h
            aspect = W / H
            new_h = int(math.sqrt(total_pixels / aspect))
            new_w = int(math.sqrt(total_pixels * aspect))
        elif target_w == 0 and target_h == 0:
            new_w, new_h = W, H
        elif target_w == 0:
            ratio = target_h / H
            new_w = round(W * ratio)
            new_h = target_h
        elif target_h == 0:
            ratio = target_w / W
            new_w = target_w
            new_h = round(H * ratio)
        else:
            ratio = min(target_w / W, target_h / H)
            new_w = round(W * ratio)
            new_h = round(H * ratio)

        if is_pad_mode:
            if crop_position == "center":
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
            elif crop_position == "top":
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                pad_top = 0
                pad_bottom = target_h - new_h
            elif crop_position == "bottom":
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                pad_top = target_h - new_h
                pad_bottom = 0
            elif crop_position == "left":
                pad_left = 0
                pad_right = target_w - new_w
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
            elif crop_position == "right":
                pad_left = target_w - new_w
                pad_right = 0
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top

        target_w = new_w
        target_h = new_h
    elif keep_proportion == "crop":
        pass  # cropping is applied below after divisible_by
    else:  # "stretch"
        if target_w == 0:
            target_w = W
        if target_h == 0:
            target_h = H

    if divisible_by > 1:
        target_w = target_w - (target_w % divisible_by)
        target_h = target_h - (target_h % divisible_by)

    out = image

    if keep_proportion == "crop":
        old_aspect = W / H
        new_aspect = target_w / target_h
        if old_aspect > new_aspect:
            crop_w = round(H * new_aspect)
            crop_h = H
        else:
            crop_w = W
            crop_h = round(W / new_aspect)
        if crop_position == "center":
            x = (W - crop_w) // 2
            y = (H - crop_h) // 2
        elif crop_position == "top":
            x = (W - crop_w) // 2
            y = 0
        elif crop_position == "bottom":
            x = (W - crop_w) // 2
            y = H - crop_h
        elif crop_position == "left":
            x = 0
            y = (H - crop_h) // 2
        elif crop_position == "right":
            x = W - crop_w
            y = (H - crop_h) // 2
        else:
            x = (W - crop_w) // 2
            y = (H - crop_h) // 2
        out = out.narrow(1, y, crop_h).narrow(2, x, crop_w)

    out = comfy_utils.common_upscale(
        out.movedim(-1, 1), target_w, target_h, upscale_method, "disabled"
    ).movedim(1, -1)

    if is_pad_mode and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
        # Ensure padded dims are also divisible_by-aligned
        if divisible_by > 1:
            final_w = target_w + pad_left + pad_right
            final_h = target_h + pad_top + pad_bottom
            w_rem = final_w % divisible_by
            h_rem = final_h % divisible_by
            if w_rem > 0:
                pad_right += divisible_by - w_rem
            if h_rem > 0:
                pad_bottom += divisible_by - h_rem

        color_vals = [int(v.strip()) for v in pad_color.split(",")]
        color_f = [c / 255.0 for c in color_vals[:3]]
        final_w = target_w + pad_left + pad_right
        final_h = target_h + pad_top + pad_bottom
        color_t = torch.tensor(color_f, dtype=out.dtype, device=out.device)
        padded = color_t.view(1, 1, 1, 3).expand(B, final_h, final_w, 3).clone()
        padded[:, pad_top : pad_top + target_h, pad_left : pad_left + target_w, :] = out
        out = padded

    return out, int(out.shape[2]), int(out.shape[1])


def image_batch_extend_with_overlap(
    source_images: Any,
    new_images: Any = None,
    overlap: int = 13,
    overlap_side: str = "source",
    overlap_mode: str = "filmic_crossfade",
) -> Any:
    """Stitch video extension frames with cross-fade overlap.

    Mirrors ``ImageBatchExtendWithOverlap.imagesfrombatch()`` from KJNodes
    (comfyui-kjnodes).

    Returns the ``extended_images`` IMAGE tensor (the 3rd of 3 outputs —
    ``source_images`` and ``start_images`` passthroughs are discarded).

    ``overlap_mode`` options: ``"linear_blend"``, ``"filmic_crossfade"``,
    ``"ease_in_out"``, ``"cut"``, ``"perceptual_crossfade"``.
    The ``"perceptual_crossfade"`` mode requires ``kornia``.
    """
    torch = _get_torch_module()

    if overlap > len(source_images):
        return source_images

    if new_images is None:
        return torch.zeros((1, 64, 64, 3))

    if source_images.shape[1:3] != new_images.shape[1:3]:
        raise ValueError(
            f"Source and new images must have the same spatial dimensions: "
            f"{tuple(source_images.shape[1:3])} vs {tuple(new_images.shape[1:3])}"
        )

    prefix = source_images[:-overlap]

    if overlap_side == "source":
        blend_src = source_images[-overlap:]
        blend_dst = new_images[:overlap]
    else:  # "new_images"
        blend_src = new_images[:overlap]
        blend_dst = source_images[-overlap:]

    suffix = new_images[overlap:]

    if overlap_mode == "linear_blend":
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        alpha = alpha.view(-1, 1, 1, 1)
        blended = (1 - alpha) * blend_src + alpha * blend_dst

    elif overlap_mode == "filmic_crossfade":
        gamma = 2.2
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        alpha = alpha.view(-1, 1, 1, 1)
        lin_src = torch.pow(blend_src, gamma)
        lin_dst = torch.pow(blend_dst, gamma)
        blended = torch.pow((1 - alpha) * lin_src + alpha * lin_dst, 1.0 / gamma)

    elif overlap_mode == "ease_in_out":
        t = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        eased = 3 * t * t - 2 * t * t * t
        eased = eased.view(-1, 1, 1, 1)
        blended = (1 - eased) * blend_src + eased * blend_dst

    elif overlap_mode == "cut":
        if overlap_side == "new_images":
            return torch.cat((source_images, new_images[overlap:]), dim=0)
        else:
            return torch.cat((source_images[:-overlap], new_images), dim=0)

    elif overlap_mode == "perceptual_crossfade":
        try:
            import kornia
        except ImportError as exc:
            raise ImportError(
                "kornia is required for perceptual_crossfade overlap mode. "
                "Install it with: pip install kornia"
            ) from exc
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        src_nchw = blend_src.movedim(-1, 1)
        dst_nchw = blend_dst.movedim(-1, 1)
        lab_src = kornia.color.rgb_to_lab(src_nchw)
        lab_dst = kornia.color.rgb_to_lab(dst_nchw)
        alpha = alpha.view(-1, 1, 1, 1)
        blended_lab = (1 - alpha) * lab_src + alpha * lab_dst
        blended = kornia.color.lab_to_rgb(blended_lab).movedim(1, -1)

    else:
        raise ValueError(f"Unknown overlap_mode: {overlap_mode!r}")

    return torch.cat((prefix, blended, suffix), dim=0)


def dw_preprocessor(
    image: Any,
    detect_hand: bool = True,
    detect_body: bool = True,
    detect_face: bool = True,
    resolution: int = 512,
) -> Any:
    """Estimate human pose from an IMAGE tensor using DWposeDetector.

    Processes each frame in the batch through the DWpose model and returns an
    IMAGE tensor (BHWC float32) with the same batch dimension as the input.

    Requires ``controlnet-aux`` to be installed::

        pip install controlnet-aux
    """
    DWposeDetector, torch, np = _get_dw_preprocessor_deps()

    detector = DWposeDetector()
    batch_size = image.shape[0]
    pose_frames = []
    for i in range(batch_size):
        frame_np = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        result = detector(
            frame_np,
            detect_resolution=resolution,
            image_resolution=resolution,
            include_body=detect_body,
            include_hand=detect_hand,
            include_face=detect_face,
            return_pil=False,
        )
        pose_frames.append(torch.from_numpy(result).float() / 255.0)
    return torch.stack(pose_frames, dim=0)


def image_scale_to_total_pixels(
    image: Any,
    upscale_method: str,
    megapixels: float,
    smallest_side: int,
) -> Any:
    """Scale an IMAGE tensor to a target total pixel count.

    Wraps ``ImageScaleToTotalPixels`` from ``comfy_extras.nodes_post_processing``.
    ``smallest_side`` maps to ``resolution_steps`` — output dimensions are
    rounded to multiples of this value.
    """
    node_type = _get_image_scale_to_total_pixels_type()
    return _unwrap_node_output(
        node_type.execute(
            image=image,
            upscale_method=upscale_method,
            megapixels=megapixels,
            resolution_steps=smallest_side,
        )
    )


def image_scale_to_max_dimension(
    image: Any,
    upscale_method: str,
    max_dimension: int,
) -> Any:
    """Scale an IMAGE tensor so its largest dimension equals ``max_dimension``.

    Wraps ``ImageScaleToMaxDimension`` from ``comfy_extras.nodes_images``.
    """
    node_type = _get_image_scale_to_max_dimension_type()
    return _unwrap_node_output(
        node_type.execute(
            image=image,
            upscale_method=upscale_method,
            largest_size=max_dimension,
        )
    )


def get_image_size(image: Any) -> tuple[int, int]:
    """Return ``(width, height)`` of an IMAGE tensor.

    Wraps ``GetImageSize`` from ``comfy_extras.nodes_images``.
    """
    node_type = _get_get_image_size_type()
    result = node_type.execute(image=image)
    raw = getattr(result, "result", result)
    return int(raw[0]), int(raw[1])


def _get_flux_kontext_dependencies() -> Any:
    """Resolve comfy.utils and PREFERED_KONTEXT_RESOLUTIONS at call time."""
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    import comfy.utils
    from comfy_extras.nodes_flux import PREFERED_KONTEXT_RESOLUTIONS

    return comfy.utils, PREFERED_KONTEXT_RESOLUTIONS


def flux_kontext_image_scale(image: Any) -> Any:
    """Resize an image to the nearest optimal Flux Kontext resolution.

    Mirrors ``FluxKontextImageScale.execute()`` from ``comfy_extras/nodes_flux.py``.
    Selects the resolution from a fixed list that best preserves the input
    aspect ratio, then upscales or downscales with Lanczos + centre-crop.

    Args:
        image: BHWC image tensor.

    Returns:
        BHWC image tensor resized to the best-matching Kontext resolution.
    """
    comfy_utils, preferred_resolutions = _get_flux_kontext_dependencies()
    width = image.shape[2]
    height = image.shape[1]
    aspect_ratio = width / height
    _, out_width, out_height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in preferred_resolutions
    )
    return comfy_utils.common_upscale(
        image.movedim(-1, 1), out_width, out_height, "lanczos", "center"
    ).movedim(1, -1)


__all__ = [
    "load_image",
    "image_to_tensor",
    "images_to_tensor",
    "image_pad_for_outpaint",
    "image_upscale_with_model",
    "image_from_batch",
    "repeat_image_batch",
    "image_composite_masked",
    "ltxv_preprocess",
    "resize_image_mask",
    "resize_images_by_longer_edge",
    "empty_image",
    "math_expression",
    "canny",
    "image_invert",
    "image_scale_by",
    "dw_preprocessor",
    "image_resize_kj",
    "image_batch_extend_with_overlap",
    "image_scale_to_total_pixels",
    "image_scale_to_max_dimension",
    "get_image_size",
    "flux_kontext_image_scale",
]
