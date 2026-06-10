"""Segmentation helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["sam3_detect"]


def _get_sam3_detect_type() -> Any:
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    from comfy_extras.nodes_sam3 import SAM3_Detect

    return SAM3_Detect


def _unwrap_node_output(output: Any) -> tuple[Any, ...]:
    result = getattr(output, "result", output)
    return tuple(result)


def sam3_detect(
    model: Any,
    image: Any,
    *,
    conditioning: Any | None = None,
    bboxes: Any | None = None,
    positive_coords: str | None = None,
    negative_coords: str | None = None,
    threshold: float = 0.5,
    refine_iterations: int = 2,
    individual_masks: bool = False,
) -> tuple[Any, Any]:
    """Detect and segment objects with ComfyUI's SAM3 node.

    ``image`` is a ComfyUI IMAGE tensor with shape ``BHWC``. When text
    prompting is used, pass conditioning from :func:`conditioning.encode_prompt`
    using the SAM3 checkpoint's CLIP output.
    """
    sam3_detect_type = _get_sam3_detect_type()
    masks, detected_bboxes = _unwrap_node_output(
        sam3_detect_type.execute(
            model=model,
            image=image,
            conditioning=conditioning,
            bboxes=bboxes,
            positive_coords=positive_coords,
            negative_coords=negative_coords,
            threshold=threshold,
            refine_iterations=refine_iterations,
            individual_masks=individual_masks,
        )
    )
    return masks, detected_bboxes
