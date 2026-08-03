"""Video generation pipeline sub-packages for comfy-diffusion.

Each sub-package groups pipelines by model family, mirroring the layout of
``comfyui_official_workflows/video/``.

Import directly from the specific pipeline module, e.g.::

    from comfy_diffusion.pipelines.video.ltx.ltx2.t2v import manifest, run
"""

__all__ = ["ltx", "minimax", "wan"]
