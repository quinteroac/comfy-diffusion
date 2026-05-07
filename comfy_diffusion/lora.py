"""LoRA application helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast


def apply_lora(
    model: Any,
    clip: Any,
    path: str | Path,
    strength_model: float,
    strength_clip: float,
) -> tuple[Any, Any]:
    """Apply a LoRA file to a model/CLIP pair and return patched copies.

    The returned pair can be passed back into ``apply_lora`` to stack
    multiple LoRAs by chaining calls.
    """
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()

    import comfy.sd
    import comfy.utils

    lora_path = str(Path(path))
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    patched = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    return cast(tuple[Any, Any], patched)


def apply_ic_lora_model_only(
    model: Any,
    path: str | Path,
    strength_model: float = 1.0,
) -> tuple[Any, float]:
    """Apply an IC-LoRA to a model and return its reference downscale factor.

    Mirrors Lightricks' ``LTXICLoRALoaderModelOnly`` node: the LoRA is applied
    to the diffusion model only, and ``reference_downscale_factor`` is read
    from safetensors metadata. Missing or invalid metadata falls back to ``1.0``.
    """
    from ._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()

    import comfy.sd
    import comfy.utils

    lora_path = str(Path(path))
    loaded = comfy.utils.load_torch_file(
        lora_path,
        safe_load=True,
        return_metadata=True,
    )
    lora, metadata = loaded

    try:
        latent_downscale_factor = float(metadata["reference_downscale_factor"])
    except (KeyError, TypeError, ValueError):
        latent_downscale_factor = 1.0

    if strength_model == 0:
        return model, latent_downscale_factor

    patched_model, _ = comfy.sd.load_lora_for_models(
        model,
        None,
        lora,
        strength_model,
        0.0,
    )
    return patched_model, latent_downscale_factor


__all__ = ["apply_lora", "apply_ic_lora_model_only"]
