"""LTX-Video 2.3 (22B dev fp8) pipeline modules.

All four pipelines use the dev fp8 checkpoint (``ltx-2.3-22b-dev-fp8.safetensors``)
with the distilled LoRA and Gemma text-encoder LoRA applied at runtime.  They
follow a two-pass sampling chain: a full denoising pass followed by spatial
upscaling and a short refinement pass.

Available pipelines
-------------------
t2v
    Text-to-video-with-audio.  Mirrors ``video_ltx2_3_t2v.json``.
i2v
    Image-to-video-with-audio.  Mirrors ``video_ltx2_3_i2v.json``.
flf2v
    First-last-frame-to-video-with-audio.  Generates a smooth video transition
    between two guide images.  Mirrors ``video_ltx2_3_flf2v.json``.
ia2v
    Image+audio-to-video.  Jointly conditioned on a reference image and an
    audio file.  The audio is VAE-encoded and injected with a zero noise mask
    so the sampler uses it as structural guidance.
    Mirrors ``video_ltx2_3_ia2v.json``.
motion_track
    Motion-track IC-LoRA pipeline.  Uses a rendered sparse-track control video
    to guide object or region motion.

Usage::

    from comfy_diffusion.pipelines.video.ltx.ltx23.t2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.i2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.flf2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.ia2v import manifest, run
    from comfy_diffusion.pipelines.video.ltx.ltx23.motion_track import manifest, run
"""

__all__ = ["t2v", "i2v", "flf2v", "ia2v", "motion_track"]
