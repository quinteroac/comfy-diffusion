from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


def test_manifest_returns_void_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.void import manifest

    entries = manifest()

    assert len(entries) == 5
    assert all(isinstance(entry, HFModelEntry) for entry in entries)
    assert {entry.repo_id for entry in entries} == {
        "Comfy-Org/void-model",
        "comfyanonymous/flux_text_encoders",
    }

    filenames = [entry.filename for entry in entries]
    dests = [str(entry.dest) for entry in entries]

    assert "diffusion_models/void_pass1.safetensors" in filenames
    assert "diffusion_models/void_pass2.safetensors" in filenames
    assert "optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.safetensors" in filenames
    assert "t5xxl_fp16.safetensors" in filenames
    assert "vae/cogvideox_vae.safetensors" in filenames

    assert "diffusion_models/void_pass1.safetensors" in dests
    assert "diffusion_models/void_pass2.safetensors" in dests
    assert "optical_flow/raft_large_C_T_SKHT_V2-ff5fadd5.safetensors" in dests
    assert "text_encoders/t5xxl_fp16.safetensors" in dests
    assert "vae/cogvideox_vae.safetensors" in dests


def test_manifest_can_include_sam3_checkpoint() -> None:
    from comfy_diffusion.pipelines.video.void import manifest

    entries = manifest(include_sam3=True)

    assert len(entries) == 6
    assert entries[-1].repo_id == "Comfy-Org/sam3.1"
    assert entries[-1].filename == "checkpoints/sam3.1_multiplex_fp16.safetensors"
    assert str(entries[-1].dest) == "checkpoints/sam3.1_multiplex_fp16.safetensors"


def test_run_wires_two_pass_void_pipeline(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video import void as pipeline_mod

    (tmp_path / "diffusion_models").mkdir()
    (tmp_path / "optical_flow").mkdir()
    (tmp_path / "text_encoders").mkdir()
    (tmp_path / "vae").mkdir()

    mm = MagicMock()
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    pass1_model = MagicMock(name="pass1_model")
    pass2_model = MagicMock(name="pass2_model")
    mm.load_unet.side_effect = [pass1_model, pass2_model]
    mm.load_optical_flow.return_value = MagicMock(name="optical_flow")

    pass1_frames = [MagicMock(name="pass1_frame")]
    final_frames = [MagicMock(name="final_frame")]

    with (
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12.0"}),
        patch("comfy_diffusion.models.ModelManager", return_value=mm),
        patch("comfy_diffusion.conditioning.encode_prompt", return_value=("positive", "negative")),
        patch(
            "comfy_diffusion.conditioning.void_quadmask_preprocess",
            return_value="quadmask",
        ) as quadmask,
        patch(
            "comfy_diffusion.conditioning.void_inpaint_conditioning",
            return_value=("positive_void", "negative_void", {"samples": "latent"}),
        ) as inpaint_conditioning,
        patch("comfy_diffusion.sampling.void_sampler", return_value="void_sampler") as sampler,
        patch(
            "comfy_diffusion.sampling.basic_scheduler",
            side_effect=["sigmas1", "sigmas2"],
        ) as scheduler,
        patch(
            "comfy_diffusion.sampling.cfg_guider",
            side_effect=["guider1", "guider2"],
        ) as guider,
        patch("comfy_diffusion.sampling.random_noise", return_value="noise1") as random_noise,
        patch(
            "comfy_diffusion.sampling.sample_custom",
            side_effect=[("sampled1", "denoised1"), ("sampled2", "denoised2")],
        ) as sample_custom,
        patch(
            "comfy_diffusion.vae.vae_decode_batch",
            side_effect=[pass1_frames, final_frames],
        ) as decode,
        patch(
            "comfy_diffusion.image.images_to_tensor",
            return_value="pass1_video_tensor",
        ) as images_to_tensor,
        patch(
            "comfy_diffusion.video.void_warped_noise",
            return_value={"samples": "warped"},
        ) as warped_noise,
        patch(
            "comfy_diffusion.sampling.void_warped_noise_source",
            return_value="warped_noise_source",
        ) as warped_noise_source,
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            video=object(),
            mask=object(),
            prompt="remove the object",
            negative_prompt="blur",
            seed=123,
            steps=25,
            cfg=5.5,
        )

    assert result == {"frames": final_frames, "pass1_frames": pass1_frames}
    mm.load_clip.assert_called_once()
    assert mm.load_clip.call_args.kwargs == {"clip_type": "cogvideox"}
    mm.load_vae.assert_called_once()
    mm.load_unet.assert_has_calls(
        [
            call(tmp_path / "diffusion_models" / "void_pass1.safetensors"),
            call(tmp_path / "diffusion_models" / "void_pass2.safetensors"),
        ]
    )
    mm.load_optical_flow.assert_called_once_with(
        tmp_path / "optical_flow" / "raft_large_C_T_SKHT_V2-ff5fadd5.safetensors"
    )

    quadmask.assert_called_once()
    inpaint_conditioning.assert_called_once()
    sampler.assert_called_once()
    scheduler.assert_has_calls(
        [
            call(pass1_model, "simple", 25, denoise=1.0),
            call(pass2_model, "simple", 25, denoise=1.0),
        ]
    )
    assert scheduler.call_args_list[0].args[1:] == ("simple", 25)
    assert scheduler.call_args_list[1].args[1:] == ("simple", 25)
    guider.assert_has_calls(
        [
            call(pass1_model, "positive_void", "negative_void", 5.5),
            call(pass2_model, "positive_void", "negative_void", 5.5),
        ]
    )
    random_noise.assert_called_once_with(123)
    sample_custom.assert_has_calls(
        [
            call("noise1", "guider1", "void_sampler", "sigmas1", {"samples": "latent"}),
            call(
                "warped_noise_source",
                "guider2",
                "void_sampler",
                "sigmas2",
                {"samples": "latent"},
            ),
        ]
    )
    decode.assert_has_calls(
        [
            call(mm.load_vae.return_value, "sampled1"),
            call(mm.load_vae.return_value, "sampled2"),
        ]
    )
    images_to_tensor.assert_called_once_with(pass1_frames)
    warped_noise.assert_called_once()
    warped_noise_source.assert_called_once_with({"samples": "warped"})


def test_run_refine_false_skips_pass2(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video import void as pipeline_mod

    tmp_path.mkdir(exist_ok=True)
    mm = MagicMock()
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_unet.return_value = MagicMock(name="pass1_model")
    pass1_frames = [MagicMock(name="pass1_frame")]

    with (
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12.0"}),
        patch("comfy_diffusion.models.ModelManager", return_value=mm),
        patch("comfy_diffusion.conditioning.encode_prompt", return_value=("positive", "negative")),
        patch("comfy_diffusion.conditioning.void_quadmask_preprocess", return_value="quadmask"),
        patch(
            "comfy_diffusion.conditioning.void_inpaint_conditioning",
            return_value=("positive_void", "negative_void", {"samples": "latent"}),
        ),
        patch("comfy_diffusion.sampling.void_sampler", return_value="void_sampler"),
        patch("comfy_diffusion.sampling.cfg_guider", return_value="guider1"),
        patch("comfy_diffusion.sampling.random_noise", return_value="noise1") as random_noise,
        patch("comfy_diffusion.sampling.sample_custom", return_value=("sampled1", "denoised1")),
        patch("comfy_diffusion.vae.vae_decode_batch", return_value=pass1_frames) as decode,
        patch("comfy_diffusion.video.void_warped_noise") as warped_noise,
        patch("comfy_diffusion.sampling.basic_scheduler", return_value="sigmas1") as scheduler,
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            video=object(),
            mask=object(),
            prompt="remove the object",
            refine=False,
        )

    assert result == {"frames": pass1_frames, "pass1_frames": pass1_frames}
    scheduler.assert_called_once_with(mm.load_unet.return_value, "simple", 30, denoise=1.0)
    random_noise.assert_called_once_with(43)
    decode.assert_called_once_with(mm.load_vae.return_value, "sampled1")
    mm.load_optical_flow.assert_not_called()
    warped_noise.assert_not_called()


def test_run_generates_video_mask_from_sam3_prompt(tmp_path: Path) -> None:
    from comfy_diffusion.models import CheckpointResult
    from comfy_diffusion.pipelines.video import void as pipeline_mod

    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "sam3.1_multiplex_fp16.safetensors").write_text("stub")

    source_video = MagicMock(name="source_video")
    sam3_model = MagicMock(name="sam3_model")
    sam3_clip = MagicMock(name="sam3_clip")
    mm = MagicMock()
    mm.load_checkpoint.return_value = CheckpointResult(
        model=sam3_model,
        clip=sam3_clip,
        vae=None,
    )
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_unet.return_value = MagicMock(name="pass1_model")
    pass1_frames = [MagicMock(name="pass1_frame")]

    with (
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12.0"}),
        patch("comfy_diffusion.models.ModelManager", return_value=mm),
        patch("comfy_diffusion.conditioning.encode_prompt") as encode_prompt,
        patch(
            "comfy_diffusion.segmentation.sam3_detect",
            return_value=("sam3_mask", []),
        ) as detect,
        patch(
            "comfy_diffusion.conditioning.void_quadmask_preprocess",
            return_value="quadmask",
        ) as quadmask,
        patch(
            "comfy_diffusion.conditioning.void_inpaint_conditioning",
            return_value=("positive_void", "negative_void", {"samples": "latent"}),
        ),
        patch("comfy_diffusion.sampling.void_sampler", return_value="void_sampler"),
        patch("comfy_diffusion.sampling.basic_scheduler", return_value="sigmas1"),
        patch("comfy_diffusion.sampling.cfg_guider", return_value="guider1"),
        patch("comfy_diffusion.sampling.random_noise", return_value="noise1"),
        patch("comfy_diffusion.sampling.sample_custom", return_value=("sampled1", "denoised1")),
        patch("comfy_diffusion.vae.vae_decode_batch", return_value=pass1_frames),
    ):
        encode_prompt.side_effect = ["sam3_conditioning", ("positive", "negative")]
        result = pipeline_mod.run(
            models_dir=tmp_path,
            video=source_video,
            mask_prompt="person",
            prompt="remove the person",
            refine=False,
            sam3_threshold=0.7,
            sam3_refine_iterations=3,
        )

    assert result == {"frames": pass1_frames, "pass1_frames": pass1_frames}
    mm.load_checkpoint.assert_called_once_with("sam3.1_multiplex_fp16.safetensors")
    encode_prompt.assert_has_calls(
        [
            call(sam3_clip, "person"),
            call(mm.load_clip.return_value, "remove the person", ""),
        ]
    )
    detect.assert_called_once_with(
        sam3_model,
        source_video,
        conditioning="sam3_conditioning",
        threshold=0.7,
        refine_iterations=3,
        individual_masks=False,
    )
    quadmask.assert_called_once_with("sam3_mask", dilate_width=0)


def test_run_slices_source_and_mask_by_start_and_duration(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video import void as pipeline_mod

    tmp_path.mkdir(exist_ok=True)
    mm = MagicMock()
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    mm.load_unet.return_value = MagicMock(name="pass1_model")
    pass1_frames = [MagicMock(name="pass1_frame")]

    with (
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12.0"}),
        patch("comfy_diffusion.models.ModelManager", return_value=mm),
        patch("comfy_diffusion.conditioning.encode_prompt", return_value=("positive", "negative")),
        patch(
            "comfy_diffusion.conditioning.void_quadmask_preprocess",
            return_value="quadmask",
        ) as quadmask,
        patch(
            "comfy_diffusion.conditioning.void_inpaint_conditioning",
            return_value=("positive_void", "negative_void", {"samples": "latent"}),
        ) as inpaint_conditioning,
        patch("comfy_diffusion.sampling.void_sampler", return_value="void_sampler"),
        patch("comfy_diffusion.sampling.basic_scheduler", return_value="sigmas1"),
        patch("comfy_diffusion.sampling.cfg_guider", return_value="guider1"),
        patch("comfy_diffusion.sampling.random_noise", return_value="noise1"),
        patch("comfy_diffusion.sampling.sample_custom", return_value=("sampled1", "denoised1")),
        patch("comfy_diffusion.vae.vae_decode_batch", return_value=pass1_frames),
    ):
        result = pipeline_mod.run(
            models_dir=tmp_path,
            video=list(range(10)),
            mask=list(range(10, 20)),
            prompt="empty snowy mountain view",
            start_frame_index=2,
            duration_seconds=2,
            fps=3,
            refine=False,
        )

    assert result == {"frames": pass1_frames, "pass1_frames": pass1_frames}
    quadmask.assert_called_once_with([12, 13, 14, 15, 16, 17], dilate_width=0)
    assert inpaint_conditioning.call_args.kwargs["video"] == [2, 3, 4, 5, 6, 7]
    assert inpaint_conditioning.call_args.kwargs["length"] == 6


def test_run_requires_exactly_one_mask_source(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video import void as pipeline_mod

    tmp_path.mkdir(exist_ok=True)
    mm = MagicMock()

    with (
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12.0"}),
        patch("comfy_diffusion.models.ModelManager", return_value=mm),
        pytest.raises(ValueError, match="either mask or mask_prompt is required"),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            video=object(),
            prompt="remove the person",
            refine=False,
        )

    with (
        patch("comfy_diffusion.runtime.check_runtime", return_value={"python_version": "3.12.0"}),
        patch("comfy_diffusion.models.ModelManager", return_value=mm),
        pytest.raises(ValueError, match="pass either mask or mask_prompt"),
    ):
        pipeline_mod.run(
            models_dir=tmp_path,
            video=object(),
            mask=object(),
            mask_prompt="person",
            prompt="remove the person",
            refine=False,
        )
