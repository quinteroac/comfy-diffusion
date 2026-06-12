"""Tests for the WAN 2.2 Bernini reference-guided video editing pipeline."""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT
    / "comfy_diffusion"
    / "pipelines"
    / "video"
    / "wan"
    / "wan22"
    / "bernini.py"
)

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_APPLY_LORA_PATCH = "comfy_diffusion.lora.apply_lora"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_BERNINI_CONDITIONING_PATCH = (
    "comfy_diffusion.pipelines.video.wan.wan22.bernini._run_bernini_conditioning"
)
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_GET_SAMPLER_PATCH = "comfy_diffusion.sampling.get_sampler"
_BASIC_SCHEDULER_PATCH = "comfy_diffusion.sampling.basic_scheduler"
_SPLIT_SIGMAS_PATCH = "comfy_diffusion.sampling.split_sigmas"
_SAMPLE_CUSTOM_SIMPLE_PATCH = "comfy_diffusion.sampling.sample_custom_simple"
_VAE_DECODE_BATCH_PATCH = "comfy_diffusion.vae.vae_decode_batch"


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file()


def test_pipeline_parses_without_syntax_errors() -> None:
    tree = ast.parse(_PIPELINE_FILE.read_text(encoding="utf-8"))
    assert isinstance(tree, ast.Module)


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), f"Top-level comfy import at line {i}: {line!r}"


def test_all_is_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '__all__ = ["manifest", "run"]' in source


def test_manifest_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import manifest

    entries = manifest()
    assert len(entries) == 5
    assert all(isinstance(entry, HFModelEntry) for entry in entries)

    filenames = [entry.filename for entry in entries]
    dests = [str(entry.dest) for entry in entries]
    repos = [entry.repo_id for entry in entries]

    assert "Bernini/Wan22_Bernini_HIGH_fp8_e4m3fn_scaled.safetensors" in filenames
    assert "Bernini/Wan22_Bernini_LOW_fp8_e4m3fn_scaled.safetensors" in filenames
    assert (
        "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
        in filenames
    )
    assert "nsfw_wan_umt5-xxl_fp8_scaled.safetensors" in filenames
    assert "split_files/vae/wan_2.1_vae.safetensors" in filenames

    assert any("diffusion_models" in dest and "HIGH" in dest for dest in dests)
    assert any("diffusion_models" in dest and "LOW" in dest for dest in dests)
    assert any("loras/wan22" in dest for dest in dests)
    assert "Kijai/WanVideo_comfy_fp8_scaled" in repos
    assert "Kijai/WanVideo_comfy" in repos
    assert "NSFW-API/NSFW-Wan-UMT5-XXL" in repos


def test_run_signature_defaults_match_workflow() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    sig = inspect.signature(run)
    assert sig.parameters["source_video"].default is None
    assert sig.parameters["reference_image"].default is None
    assert sig.parameters["width"].default == 832
    assert sig.parameters["height"].default == 480
    assert sig.parameters["length"].default == 81
    assert sig.parameters["models_dir"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["seed"].default == 3
    assert sig.parameters["steps"].default == 8
    assert sig.parameters["split_step"].default == 4
    assert sig.parameters["cfg"].default == 1.0
    assert sig.parameters["high_lora_strength"].default == 3.0
    assert sig.parameters["low_lora_strength"].default == 1.5
    assert sig.parameters["sampler_name"].default == "res_multistep"
    assert sig.parameters["scheduler"].default == "simple"
    assert sig.parameters["ref_max_size"].default == 848


def test_bernini_conditioning_uses_vendored_node_autogrow(monkeypatch: object) -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import _run_bernini_conditioning

    class FakeNodeOutput:
        def __init__(self, *values: object) -> None:
            self.result = values

    class FakeBerniniConditioning:
        call_kwargs: dict[str, object]

        @classmethod
        def execute(cls, *args: object, **kwargs: object) -> FakeNodeOutput:
            cls.call_kwargs = kwargs
            return FakeNodeOutput("bernini_positive", "bernini_negative", {"samples": "latent"})

    fake_module = types.SimpleNamespace(BerniniConditioning=FakeBerniniConditioning)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_bernini", fake_module)

    reference_one = object()
    reference_two = object()
    result = _run_bernini_conditioning(
        "positive",
        "negative",
        "vae",
        width=832,
        height=480,
        length=81,
        batch_size=1,
        source_video="source",
        reference_images=[reference_one, reference_two],
        ref_max_size=848,
    )

    assert result == ("bernini_positive", "bernini_negative", {"samples": "latent"})
    assert FakeBerniniConditioning.call_kwargs["source_video"] == "source"
    assert FakeBerniniConditioning.call_kwargs["reference_images"] == {
        "reference_image_0": reference_one,
        "reference_image_1": reference_two,
    }
    assert FakeBerniniConditioning.call_kwargs["ref_max_size"] == 848


def test_run_wires_bernini_workflow() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    source_video = object()
    reference_image = object()
    model_high = object()
    model_low = object()
    patched_high = object()
    patched_low = object()
    clip = object()
    vae = object()
    latent0 = {"samples": object()}
    latent1 = {"samples": object()}
    latent2 = {"samples": object()}
    frames = [object()]

    manager = MagicMock()
    manager.load_unet.side_effect = [model_high, model_low]
    manager.load_clip.return_value = clip
    manager.load_vae.return_value = vae

    with (
        patch(_RUNTIME_PATCH, return_value={}),
        patch(_MM_PATCH, return_value=manager) as model_manager_type,
        patch(
            _APPLY_LORA_PATCH,
            side_effect=[(patched_high, None), (patched_low, None)],
        ) as apply_lora,
        patch(_ENCODE_PATCH, return_value=("positive", "negative")) as encode_prompt,
        patch(
            _BERNINI_CONDITIONING_PATCH,
            return_value=("bernini_positive", "bernini_negative", latent0),
        ) as bernini_conditioning,
        patch(_IMAGE_TO_TENSOR_PATCH) as image_to_tensor,
        patch(_GET_SAMPLER_PATCH, return_value="sampler") as get_sampler,
        patch(_BASIC_SCHEDULER_PATCH, return_value="sigmas") as basic_scheduler,
        patch(_SPLIT_SIGMAS_PATCH, return_value=("high_sigmas", "low_sigmas")) as split_sigmas,
        patch(_SAMPLE_CUSTOM_SIMPLE_PATCH, side_effect=[latent1, latent2]) as sample_custom_simple,
        patch(_VAE_DECODE_BATCH_PATCH, return_value=frames) as vae_decode_batch,
    ):
        result = run(
            source_video,
            reference_image,
            "replace subject",
            models_dir="/models",
        )

    assert result == frames
    model_manager_type.assert_called_once_with(Path("/models"))
    assert manager.load_unet.call_count == 2
    manager.load_clip.assert_called_once()
    manager.load_vae.assert_called_once()
    apply_lora.assert_has_calls(
        [
            call(
                model_high,
                clip,
                Path(
                    "/models/loras/wan22/"
                    "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
                ),
                3.0,
                0.0,
            ),
            call(
                model_low,
                clip,
                Path(
                    "/models/loras/wan22/"
                    "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
                ),
                1.5,
                0.0,
            ),
        ]
    )
    encode_prompt.assert_called_once()
    assert encode_prompt.call_args.args[0] is clip
    assert encode_prompt.call_args.args[1] == "replace subject"
    assert isinstance(encode_prompt.call_args.args[2], str)
    image_to_tensor.assert_not_called()
    bernini_conditioning.assert_called_once_with(
        "positive",
        "negative",
        vae,
        width=832,
        height=480,
        length=81,
        batch_size=1,
        source_video=source_video,
        reference_images=[reference_image],
        ref_max_size=848,
    )
    get_sampler.assert_called_once_with("res_multistep")
    basic_scheduler.assert_called_once_with(patched_low, "simple", 8, denoise=1.0)
    split_sigmas.assert_called_once_with("sigmas", 4)
    sample_custom_simple.assert_has_calls(
        [
            call(
                patched_high,
                add_noise=True,
                noise_seed=3,
                cfg=1.0,
                positive="bernini_positive",
                negative="bernini_negative",
                sampler="sampler",
                sigmas="high_sigmas",
                latent_image=latent0,
            ),
            call(
                patched_low,
                add_noise=False,
                noise_seed=0,
                cfg=1.0,
                positive="bernini_positive",
                negative="bernini_negative",
                sampler="sampler",
                sigmas="low_sigmas",
                latent_image=latent1,
            ),
        ]
    )
    vae_decode_batch.assert_called_once_with(vae, latent2)


def test_run_accepts_multiple_reference_images() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    class FakePilImage:
        def convert(self, mode: str) -> object:
            return object()

    source_video = object()
    pil_reference = FakePilImage()
    tensor_reference = object()
    converted_reference = object()

    manager = MagicMock()
    manager.load_unet.side_effect = [object(), object()]
    manager.load_clip.return_value = object()
    manager.load_vae.return_value = object()

    with (
        patch(_RUNTIME_PATCH, return_value={}),
        patch(_MM_PATCH, return_value=manager),
        patch(_APPLY_LORA_PATCH, side_effect=[(object(), None), (object(), None)]),
        patch(_ENCODE_PATCH, return_value=("positive", "negative")),
        patch(
            _BERNINI_CONDITIONING_PATCH,
            return_value=("bernini_positive", "bernini_negative", {"samples": object()}),
        ) as bernini_conditioning,
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=converted_reference) as image_to_tensor,
        patch(_GET_SAMPLER_PATCH, return_value="sampler"),
        patch(_BASIC_SCHEDULER_PATCH, return_value="sigmas"),
        patch(_SPLIT_SIGMAS_PATCH, return_value=("high_sigmas", "low_sigmas")),
        patch(
            _SAMPLE_CUSTOM_SIMPLE_PATCH,
            side_effect=[{"samples": object()}, {"samples": object()}],
        ),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        run(
            source_video,
            [pil_reference, tensor_reference],
            models_dir="/models",
        )

    image_to_tensor.assert_called_once_with(pil_reference)
    assert bernini_conditioning.call_args.kwargs["reference_images"] == [
        converted_reference,
        tensor_reference,
    ]


def test_run_allows_text_only_reference() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    manager = MagicMock()
    manager.load_unet.side_effect = [object(), object()]
    manager.load_clip.return_value = object()
    manager.load_vae.return_value = object()

    with (
        patch(_RUNTIME_PATCH, return_value={}),
        patch(_MM_PATCH, return_value=manager),
        patch(_APPLY_LORA_PATCH, side_effect=[(object(), None), (object(), None)]),
        patch(_ENCODE_PATCH, return_value=("positive", "negative")),
        patch(
            _BERNINI_CONDITIONING_PATCH,
            return_value=("bernini_positive", "bernini_negative", {"samples": object()}),
        ) as bernini_conditioning,
        patch(_IMAGE_TO_TENSOR_PATCH) as image_to_tensor,
        patch(_GET_SAMPLER_PATCH, return_value="sampler"),
        patch(_BASIC_SCHEDULER_PATCH, return_value="sigmas"),
        patch(_SPLIT_SIGMAS_PATCH, return_value=("high_sigmas", "low_sigmas")),
        patch(
            _SAMPLE_CUSTOM_SIMPLE_PATCH,
            side_effect=[{"samples": object()}, {"samples": object()}],
        ),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        run(
            object(),
            None,
            "turn the person into a chrome robot",
            models_dir="/models",
        )

    image_to_tensor.assert_not_called()
    assert bernini_conditioning.call_args.kwargs["reference_images"] is None


def test_run_allows_reference_images_without_source_video() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    reference_one = object()
    reference_two = object()

    manager = MagicMock()
    manager.load_unet.side_effect = [object(), object()]
    manager.load_clip.return_value = object()
    manager.load_vae.return_value = object()

    with (
        patch(_RUNTIME_PATCH, return_value={}),
        patch(_MM_PATCH, return_value=manager),
        patch(_APPLY_LORA_PATCH, side_effect=[(object(), None), (object(), None)]),
        patch(_ENCODE_PATCH, return_value=("positive", "negative")),
        patch(
            _BERNINI_CONDITIONING_PATCH,
            return_value=("bernini_positive", "bernini_negative", {"samples": object()}),
        ) as bernini_conditioning,
        patch(_IMAGE_TO_TENSOR_PATCH) as image_to_tensor,
        patch(_GET_SAMPLER_PATCH, return_value="sampler"),
        patch(_BASIC_SCHEDULER_PATCH, return_value="sigmas"),
        patch(_SPLIT_SIGMAS_PATCH, return_value=("high_sigmas", "low_sigmas")),
        patch(
            _SAMPLE_CUSTOM_SIMPLE_PATCH,
            side_effect=[{"samples": object()}, {"samples": object()}],
        ),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        run(
            reference_image=[reference_one, reference_two],
            prompt="make a video from these robot references",
            models_dir="/models",
        )

    image_to_tensor.assert_not_called()
    assert bernini_conditioning.call_args.kwargs["source_video"] is None
    assert bernini_conditioning.call_args.kwargs["reference_images"] == [
        reference_one,
        reference_two,
    ]


def test_run_requires_source_or_reference() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    manager = MagicMock()
    manager.load_unet.side_effect = [object(), object()]
    manager.load_clip.return_value = object()
    manager.load_vae.return_value = object()

    with (
        patch(_RUNTIME_PATCH, return_value={}),
        patch(_MM_PATCH, return_value=manager),
        patch(_APPLY_LORA_PATCH, side_effect=[(object(), None), (object(), None)]),
        patch(_ENCODE_PATCH, return_value=("positive", "negative")),
        patch(_BERNINI_CONDITIONING_PATCH) as bernini_conditioning,
    ):
        try:
            run(prompt="make a video", models_dir="/models")
        except ValueError as exc:
            assert "source_video or reference_image" in str(exc)
        else:
            raise AssertionError("run() should require a source video or reference image")

    bernini_conditioning.assert_not_called()


def test_run_raises_when_runtime_unavailable() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.bernini import run

    with patch(_RUNTIME_PATCH, return_value={"error": "missing ComfyUI"}):
        try:
            run(source_video=object(), reference_image=object(), models_dir="/models")
        except RuntimeError as exc:
            assert "missing ComfyUI" in str(exc)
        else:
            raise AssertionError("run() should raise RuntimeError when runtime is unavailable")


def test_wan22_package_exports_bernini() -> None:
    from comfy_diffusion.pipelines.video.wan import wan22

    assert "bernini" in wan22.__all__
