"""Tests for comfy_diffusion/pipelines/video/wan/wan22/i2v.py — WAN 2.2 I2V pipeline.

Covers:
  - File exists, parses, has future annotations, module docstring
  - Exports manifest and run; no top-level comfy imports
  - __all__ = ["manifest", "run"]
  - manifest() returns exactly 6 HFModelEntry items with correct dest paths
  - All manifest entries use the Comfy-Org/Wan_2.2_ComfyUI_Repackaged HF repo
  - run() signature: image, prompt, negative_prompt, width, height, length,
    *, models_dir, seed, steps, cfg
  - run() behaviour: WanImageToVideo conditioning followed by dual two-pass
    KSamplerAdvanced (high-noise first, low-noise second) with ModelSamplingSD3
    shift=5 applied; no LoRA (switch=False default)
  - run() raises RuntimeError when check_runtime returns an error
  - wan22 sub-package exports i2v
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE_FILE = (
    _REPO_ROOT
    / "comfy_diffusion"
    / "pipelines"
    / "video"
    / "wan"
    / "wan22"
    / "i2v.py"
)

# ---------------------------------------------------------------------------
# Patch constants (source module paths for lazy imports)
# ---------------------------------------------------------------------------

_RUNTIME_PATCH = "comfy_diffusion.runtime.check_runtime"
_MM_PATCH = "comfy_diffusion.models.ModelManager"
_MODEL_SAMPLING_SD3_PATCH = "comfy_diffusion.models.model_sampling_sd3"
_ENCODE_PATCH = "comfy_diffusion.conditioning.encode_prompt"
_WAN_I2V_PATCH = "comfy_diffusion.conditioning.wan_image_to_video"
_IMAGE_TO_TENSOR_PATCH = "comfy_diffusion.image.image_to_tensor"
_SAMPLE_ADVANCED_PATCH = "comfy_diffusion.sampling.sample_advanced"
_VAE_DECODE_BATCH_PATCH = "comfy_diffusion.vae.vae_decode_batch"


# ---------------------------------------------------------------------------
# File-level checks
# ---------------------------------------------------------------------------


def test_pipeline_file_exists() -> None:
    assert _PIPELINE_FILE.is_file(), "i2v.py must exist under wan22/"


def test_pipeline_parses_without_syntax_errors() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    assert isinstance(tree, ast.Module)


def test_pipeline_has_future_annotations() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source


def test_pipeline_has_module_docstring() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_PIPELINE_FILE))
    docstring = ast.get_docstring(tree)
    assert docstring, "i2v.py must have a module-level docstring"


def test_pipeline_all_is_manifest_and_run() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    assert '__all__ = ["manifest", "run"]' in source


def test_no_top_level_comfy_imports() -> None:
    source = _PIPELINE_FILE.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("import comfy.") or stripped.startswith("from comfy."):
            assert line.startswith("    "), (
                f"Top-level comfy import at line {i}: {line!r}. "
                "Use lazy imports inside functions."
            )


def test_import_manifest_and_run() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest, run  # noqa: F401

    assert callable(manifest)
    assert callable(run)


# ---------------------------------------------------------------------------
# Manifest checks — exactly 6 HFModelEntry items (AC-01, AC-03)
# ---------------------------------------------------------------------------


def test_manifest_returns_exactly_six_entries() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    result = manifest()
    assert isinstance(result, list)
    assert len(result) == 6, f"manifest() must return exactly 6 entries, got {len(result)}"


def test_manifest_entries_are_hf_model_entries() -> None:
    from comfy_diffusion.downloader import HFModelEntry
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    for entry in manifest():
        assert isinstance(entry, HFModelEntry), (
            f"manifest() entry {entry!r} must be an HFModelEntry"
        )


def test_manifest_unet_high_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "high_noise" in d for d in dests), (
        "manifest() must include a diffusion_models/...high_noise... UNet entry"
    )


def test_manifest_unet_low_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("diffusion_models" in d and "low_noise" in d for d in dests), (
        "manifest() must include a diffusion_models/...low_noise... UNet entry"
    )


def test_manifest_lora_high_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "high_noise" in d for d in dests), (
        "manifest() must include a loras/...high_noise... LoRA entry"
    )


def test_manifest_lora_low_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("loras" in d and "low_noise" in d for d in dests), (
        "manifest() must include a loras/...low_noise... LoRA entry"
    )


def test_manifest_text_encoder_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("text_encoders" in d and "umt5_xxl" in d for d in dests), (
        "manifest() must include a text_encoders/umt5_xxl entry"
    )


def test_manifest_vae_dest_path() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    dests = [str(e.dest) for e in manifest()]
    assert any("vae" in d and "wan_2.1_vae" in d for d in dests), (
        "manifest() must include a vae/wan_2.1_vae entry"
    )


def test_manifest_all_from_comfy_org_wan22_repo() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    for entry in manifest():
        assert entry.repo_id == "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", (
            f"Entry {entry!r} must use repo_id 'Comfy-Org/Wan_2.2_ComfyUI_Repackaged'"
        )


def test_manifest_has_no_bypassed_node_entries() -> None:
    """All 6 entries correspond to active nodes — no bypassed nodes present (AC-03)."""
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    result = manifest()
    assert len(result) == 6, (
        "manifest() must have exactly 6 entries (no bypassed nodes)"
    )


# ---------------------------------------------------------------------------
# run() signature checks (AC-02)
# ---------------------------------------------------------------------------


def test_run_signature_includes_required_params() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    required = {
        "image", "prompt", "negative_prompt", "width", "height", "length",
        "models_dir", "seed", "steps", "cfg",
    }
    assert required <= set(sig.parameters.keys()), (
        f"run() is missing parameters: {required - set(sig.parameters.keys())}"
    )


def test_run_models_dir_is_keyword_only() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    param = sig.parameters["models_dir"]
    assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
        "models_dir must be keyword-only"
    )


def test_run_default_width_640() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["width"].default == 640


def test_run_default_height_640() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["height"].default == 640


def test_run_default_length_81() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["length"].default == 81


def test_run_default_steps_20() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["steps"].default == 20


def test_run_default_cfg_3_5() -> None:
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import run

    sig = inspect.signature(run)
    assert sig.parameters["cfg"].default == 3.5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_mock_mm() -> MagicMock:
    mm = MagicMock()
    mm.load_unet.return_value = MagicMock(name="model")
    mm.load_clip.return_value = MagicMock(name="clip")
    mm.load_vae.return_value = MagicMock(name="vae")
    return mm


def _run_with_mocks(
    tmp_path: Path,
    *,
    call_order: list[str] | None = None,
    fake_frames: list[Any] | None = None,
) -> list[Any]:
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    if fake_frames is None:
        fake_frames = [MagicMock(name="frame0")]

    fake_latent = MagicMock(name="latent")
    fake_i2v_latent = MagicMock(name="i2v_latent")
    fake_sampled_1 = MagicMock(name="sampled_1")
    fake_sampled_2 = MagicMock(name="sampled_2")
    fake_patched_model = MagicMock(name="patched_model")
    fake_image_tensor = MagicMock(name="image_tensor")

    mm = _build_mock_mm()
    sample_advanced_call_count: list[int] = [0]

    def _track(name: str, rv: Any) -> Any:
        if call_order is not None:
            call_order.append(name)
        return rv

    def fake_sample_advanced(model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any) -> Any:
        sample_advanced_call_count[0] += 1
        n = sample_advanced_call_count[0]
        name = f"sample_advanced_{n}"
        if call_order is not None:
            call_order.append(name)
        return fake_sampled_1 if n == 1 else fake_sampled_2

    fake_image = MagicMock(name="pil_image")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, side_effect=lambda img: (
            _track("image_to_tensor", fake_image_tensor)
        )),
        patch(_WAN_I2V_PATCH, side_effect=lambda pos, neg, vae, **kw: (
            _track("wan_image_to_video", (pos, neg, fake_i2v_latent))
        )),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=lambda m, shift: (
            _track("model_sampling_sd3", fake_patched_model)
        )),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=fake_sample_advanced),
        patch(_VAE_DECODE_BATCH_PATCH, side_effect=lambda v, s: (
            _track("vae_decode_batch", fake_frames)
        )),
    ):
        return pipeline_mod.run(
            fake_image,
            "a dragon flying through clouds",
            models_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# run() behaviour tests (AC-02)
# ---------------------------------------------------------------------------


def test_run_returns_list_of_frames(tmp_path: Path) -> None:
    result = _run_with_mocks(tmp_path)
    assert isinstance(result, list)


def test_run_frames_value(tmp_path: Path) -> None:
    fake_frames = [MagicMock(name="f0"), MagicMock(name="f1")]
    result = _run_with_mocks(tmp_path, fake_frames=fake_frames)
    assert result is fake_frames


def test_run_calls_wan_image_to_video(tmp_path: Path) -> None:
    """WanImageToVideo must be called to build I2V conditioning."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "wan_image_to_video" in call_order


def test_run_calls_image_to_tensor(tmp_path: Path) -> None:
    """Input image must be converted to tensor before WanImageToVideo."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "image_to_tensor" in call_order


def test_run_does_not_call_apply_lora(tmp_path: Path) -> None:
    """No LoRA should be applied when switches are at default False."""
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    fake_image = MagicMock(name="pil_image")
    mm = _build_mock_mm()

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
        patch("comfy_diffusion.lora.apply_lora") as mock_apply_lora,
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    mock_apply_lora.assert_not_called()


def test_run_calls_model_sampling_sd3_twice(tmp_path: Path) -> None:
    """ModelSamplingSD3 must be applied to both models."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    sd3_calls = [c for c in call_order if c == "model_sampling_sd3"]
    assert len(sd3_calls) == 2, f"model_sampling_sd3 must be called twice, got {sd3_calls}"


def test_run_calls_sample_advanced_twice(tmp_path: Path) -> None:
    """Both KSamplerAdvanced passes must be executed."""
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    sa_calls = [c for c in call_order if c.startswith("sample_advanced")]
    assert len(sa_calls) == 2, f"sample_advanced must be called twice, got {sa_calls}"


def test_run_samples_high_noise_before_low_noise(tmp_path: Path) -> None:
    """First sampler pass must use high-noise; second pass must use low-noise."""
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    fake_image = MagicMock(name="pil_image")
    model_high = MagicMock(name="model_high")
    model_low = MagicMock(name="model_low")
    patched_high = MagicMock(name="patched_high")
    patched_low = MagicMock(name="patched_low")
    mm = _build_mock_mm()
    mm.load_unet.side_effect = [model_high, model_low]

    sampled_models: list[Any] = []

    def capture_sd3(model: Any, shift: float) -> Any:
        if model is model_high:
            return patched_high
        if model is model_low:
            return patched_low
        raise AssertionError(f"Unexpected model passed to model_sampling_sd3: {model!r}")

    def capture_sample_advanced(
        model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any
    ) -> Any:
        sampled_models.append(model)
        return MagicMock(name="sampled")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=capture_sd3),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=capture_sample_advanced),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    assert sampled_models == [patched_high, patched_low]


def test_run_calls_vae_decode_batch(tmp_path: Path) -> None:
    call_order: list[str] = []
    _run_with_mocks(tmp_path, call_order=call_order)
    assert "vae_decode_batch" in call_order


def test_run_pass1_add_noise_true(tmp_path: Path) -> None:
    """First KSamplerAdvanced pass must have add_noise=True."""
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    mm = _build_mock_mm()
    captured_calls: list[dict[str, Any]] = []
    fake_image = MagicMock(name="pil_image")

    def capture_sample_advanced(
        model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any
    ) -> Any:
        captured_calls.append(kwargs)
        return MagicMock(name="sampled")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=capture_sample_advanced),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    assert len(captured_calls) == 2
    assert captured_calls[0].get("add_noise") is True, (
        f"Pass 1 must have add_noise=True, got {captured_calls[0].get('add_noise')}"
    )
    assert captured_calls[0].get("return_with_leftover_noise") is True, (
        "Pass 1 must have return_with_leftover_noise=True"
    )


def test_run_pass2_add_noise_false(tmp_path: Path) -> None:
    """Second KSamplerAdvanced pass must have add_noise=False."""
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    mm = _build_mock_mm()
    captured_calls: list[dict[str, Any]] = []
    fake_image = MagicMock(name="pil_image")

    def capture_sample_advanced(
        model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any
    ) -> Any:
        captured_calls.append(kwargs)
        return MagicMock(name="sampled")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=capture_sample_advanced),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    assert len(captured_calls) == 2
    assert captured_calls[1].get("add_noise") is False, (
        f"Pass 2 must have add_noise=False, got {captured_calls[1].get('add_noise')}"
    )
    assert captured_calls[1].get("return_with_leftover_noise") is False, (
        "Pass 2 must have return_with_leftover_noise=False"
    )


def test_run_model_sampling_sd3_shift_is_5(tmp_path: Path) -> None:
    """ModelSamplingSD3 must be called with shift=5.0 for both models."""
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    mm = _build_mock_mm()
    captured_shifts: list[float] = []
    fake_image = MagicMock(name="pil_image")

    def capture_sd3(model: Any, shift: float) -> Any:
        captured_shifts.append(shift)
        return MagicMock(name="patched_model")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, side_effect=capture_sd3),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    assert all(s == 5.0 for s in captured_shifts), (
        f"model_sampling_sd3 must be called with shift=5.0 for both models, got {captured_shifts}"
    )


def test_run_uses_euler_simple_sampler(tmp_path: Path) -> None:
    """Sampler must be 'euler' with 'simple' scheduler as in the workflow."""
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    mm = _build_mock_mm()
    captured_calls: list[dict[str, Any]] = []
    fake_image = MagicMock(name="pil_image")

    def capture_sample_advanced(
        model: Any, pos: Any, neg: Any, lat: Any, **kwargs: Any
    ) -> Any:
        captured_calls.append(kwargs)
        return MagicMock(name="sampled")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, side_effect=capture_sample_advanced),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    for i, c in enumerate(captured_calls):
        assert c.get("sampler_name") == "euler", (
            f"Pass {i+1}: sampler_name must be 'euler', got {c.get('sampler_name')}"
        )
        assert c.get("scheduler") == "simple", (
            f"Pass {i+1}: scheduler must be 'simple', got {c.get('scheduler')}"
        )


def test_run_uses_load_clip_with_wan_type(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    mm = _build_mock_mm()
    fake_image = MagicMock(name="pil_image")

    with (
        patch(_RUNTIME_PATCH, return_value={"python_version": "3.12.0"}),
        patch(_MM_PATCH, return_value=mm),
        patch(_ENCODE_PATCH, return_value=(MagicMock(), MagicMock())),
        patch(_IMAGE_TO_TENSOR_PATCH, return_value=MagicMock()),
        patch(_WAN_I2V_PATCH, return_value=(MagicMock(), MagicMock(), MagicMock())),
        patch(_MODEL_SAMPLING_SD3_PATCH, return_value=MagicMock()),
        patch(_SAMPLE_ADVANCED_PATCH, return_value=MagicMock()),
        patch(_VAE_DECODE_BATCH_PATCH, return_value=[]),
    ):
        pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)

    mm.load_clip.assert_called_once()
    _, kwargs = mm.load_clip.call_args
    assert kwargs.get("clip_type") == "wan", (
        "load_clip must be called with clip_type='wan'"
    )


def test_run_raises_on_runtime_error(tmp_path: Path) -> None:
    from comfy_diffusion.pipelines.video.wan.wan22 import i2v as pipeline_mod

    fake_image = MagicMock(name="pil_image")

    with patch(
        _RUNTIME_PATCH,
        return_value={"error": "ComfyUI submodule not initialized", "python_version": "3.12.0"},
    ):
        with pytest.raises(RuntimeError, match="ComfyUI runtime not available"):
            pipeline_mod.run(fake_image, "test prompt", models_dir=tmp_path)


# ---------------------------------------------------------------------------
# wan22 sub-package wiring
# ---------------------------------------------------------------------------


def test_wan22_exports_i2v() -> None:
    import comfy_diffusion.pipelines.video.wan.wan22 as wan22_pkg

    assert "i2v" in wan22_pkg.__all__, (
        f"wan22/__init__.py __all__ must contain 'i2v', got {wan22_pkg.__all__!r}"
    )


def test_wan22_i2v_module_importable() -> None:
    import importlib

    mod = importlib.import_module("comfy_diffusion.pipelines.video.wan.wan22.i2v")
    assert callable(mod.manifest)
    assert callable(mod.run)


# ---------------------------------------------------------------------------
# download_models idempotent — 6 entries (AC-04)
# ---------------------------------------------------------------------------


def test_download_models_idempotent_all_present(tmp_path: Path) -> None:
    from comfy_diffusion.downloader import download_models
    from comfy_diffusion.pipelines.video.wan.wan22.i2v import manifest

    entries = manifest()
    assert len(entries) == 6

    for entry in entries:
        dest = tmp_path / entry.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00" * 16)

    download_models(entries, models_dir=tmp_path, quiet=True)
