"""Tests for LTX-2.3 motion-track IC-LoRA pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock


def test_manifest_declares_motion_track_models() -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23.motion_track import manifest

    entries = manifest()
    assert len(entries) == 4
    assert [str(entry.dest) for entry in entries] == [
        "checkpoints/ltx-2.3-22b-dev.safetensors",
        "text_encoders/gemma_3_12B_it.safetensors",
        "loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "loras/ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
    ]


def test_ltx23_init_exports_motion_track() -> None:
    from comfy_diffusion.pipelines.video.ltx import ltx23

    assert "motion_track" in ltx23.__all__


def test_run_wires_ic_lora_downscale_into_video_guide(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    from comfy_diffusion.pipelines.video.ltx.ltx23 import motion_track

    calls: list[tuple[str, Any]] = []

    class FakeModelManager:
        def __init__(self, models_dir: Path) -> None:
            calls.append(("ModelManager", models_dir))

        def load_checkpoint_from_path(self, path: Path) -> Any:
            calls.append(("load_checkpoint", path))
            return Mock(model="model", vae="vae")

        def load_ltxv_audio_vae(self, path: Path) -> str:
            calls.append(("load_ltxv_audio_vae", path))
            return "audio_vae"

        def load_ltxav_text_encoder(self, text_encoder_path: Path, checkpoint_path: Path) -> str:
            calls.append(("load_ltxav_text_encoder", (text_encoder_path, checkpoint_path)))
            return "clip"

    monkeypatch.setattr(
        "comfy_diffusion.runtime.check_runtime",
        lambda: {"python_version": "3.12"},
    )
    monkeypatch.setattr("comfy_diffusion.models.ModelManager", FakeModelManager)
    monkeypatch.setattr(
        "comfy_diffusion.lora.apply_lora",
        lambda model, clip, path, sm, sc: ("distilled_model", clip),
    )
    monkeypatch.setattr(
        "comfy_diffusion.lora.apply_ic_lora_model_only",
        lambda model, path, strength_model: ("ic_model", 2.0),
    )
    monkeypatch.setattr(
        "comfy_diffusion.conditioning.encode_prompt",
        lambda clip, prompt, negative_prompt: ("positive", "negative"),
    )
    monkeypatch.setattr(
        "comfy_diffusion.conditioning.ltxv_conditioning",
        lambda positive, negative, frame_rate: ("positive_fps", "negative_fps"),
    )

    def fake_ic_guide(*args: Any, **kwargs: Any) -> tuple[str, str, dict[str, str]]:
        calls.append(("ltx_add_video_ic_lora_guide", kwargs))
        return "positive_ic", "negative_ic", {"samples": "video_latent_ic"}

    monkeypatch.setattr(
        "comfy_diffusion.conditioning.ltx_add_video_ic_lora_guide",
        fake_ic_guide,
    )
    monkeypatch.setattr(
        "comfy_diffusion.latent.ltxv_empty_latent_video",
        lambda **kwargs: {"samples": "video_latent"},
    )
    monkeypatch.setattr(
        "comfy_diffusion.audio.ltxv_empty_latent_audio",
        lambda audio_vae, frames_number, frame_rate: {"samples": "audio_latent"},
    )
    monkeypatch.setattr(
        "comfy_diffusion.audio.ltxv_concat_av_latent",
        lambda video_latent, audio_latent: {"samples": "av_latent"},
    )
    monkeypatch.setattr(
        "comfy_diffusion.sampling.cfg_guider",
        lambda model, positive, negative, cfg: "guider",
    )
    monkeypatch.setattr("comfy_diffusion.sampling.random_noise", lambda seed: "noise")
    monkeypatch.setattr("comfy_diffusion.sampling.manual_sigmas", lambda sigmas: "sigmas")
    monkeypatch.setattr("comfy_diffusion.sampling.get_sampler", lambda name: "sampler")
    monkeypatch.setattr(
        "comfy_diffusion.sampling.sample_custom",
        lambda noise, guider, sampler, sigmas, latent: ({"samples": "sampled"}, None),
    )
    monkeypatch.setattr(
        "comfy_diffusion.audio.ltxv_separate_av_latent",
        lambda latent: ({"samples": "video_out"}, {"samples": "audio_out"}),
    )
    monkeypatch.setattr(
        "comfy_diffusion.conditioning.ltxv_crop_guides",
        lambda positive, negative, latent: (positive, negative, latent),
    )
    monkeypatch.setattr(
        "comfy_diffusion.vae.vae_decode_batch_tiled",
        lambda vae, latent: ["frame"],
    )
    monkeypatch.setattr(
        "comfy_diffusion.audio.ltxv_audio_vae_decode",
        lambda audio_vae, latent: {"waveform": "audio"},
    )

    result = motion_track.run(
        models_dir=tmp_path,
        motion_track_video=object(),
        prompt="a subject follows a drawn path",
    )

    guide_call = next(call for call in calls if call[0] == "ltx_add_video_ic_lora_guide")
    assert guide_call[1]["latent_downscale_factor"] == 2.0
    assert guide_call[1]["frame_idx"] == 0
    assert guide_call[1]["strength"] == 1.0
    assert result == {"frames": ["frame"], "audio": {"waveform": "audio"}}
