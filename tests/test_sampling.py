"""Tests for sampling helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import comfy_diffusion.sampling as sampling_module
from comfy_diffusion.sampling import (
    ays_scheduler,
    basic_guider,
    basic_scheduler,
    cfg_override,
    cfg_guider,
    disable_noise,
    dual_model_guider,
    flux2_scheduler,
    get_sampler,
    ideogram4_scheduler,
    karras_scheduler,
    ltxv_scheduler,
    manual_sigmas,
    random_noise,
    sample,
    sample_advanced,
    sample_custom,
    set_first_sigma,
    split_sigmas,
    split_sigmas_denoise,
    video_linear_cfg_guidance,
    video_triangle_cfg_guidance,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    repo_root = str(_repo_root())
    env["PYTHONPATH"] = repo_root if not existing else f"{repo_root}{os.pathsep}{existing}"

    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
        cwd=_repo_root(),
    )


def test_sampling_public_api_exports_all_entrypoints() -> None:
    assert sample.__name__ == "sample"
    assert sample_advanced.__name__ == "sample_advanced"
    assert sample_custom.__name__ == "sample_custom"
    assert basic_guider.__name__ == "basic_guider"
    assert cfg_guider.__name__ == "cfg_guider"
    assert dual_model_guider.__name__ == "dual_model_guider"
    assert cfg_override.__name__ == "cfg_override"
    assert random_noise.__name__ == "random_noise"
    assert disable_noise.__name__ == "disable_noise"
    assert basic_scheduler.__name__ == "basic_scheduler"
    assert karras_scheduler.__name__ == "karras_scheduler"
    assert ays_scheduler.__name__ == "ays_scheduler"
    assert flux2_scheduler.__name__ == "flux2_scheduler"
    assert ideogram4_scheduler.__name__ == "ideogram4_scheduler"
    assert ltxv_scheduler.__name__ == "ltxv_scheduler"
    assert split_sigmas.__name__ == "split_sigmas"
    assert split_sigmas_denoise.__name__ == "split_sigmas_denoise"
    assert get_sampler.__name__ == "get_sampler"
    assert video_linear_cfg_guidance.__name__ == "video_linear_cfg_guidance"
    assert video_triangle_cfg_guidance.__name__ == "video_triangle_cfg_guidance"
    assert sampling_module.__all__ == [
        "sample",
        "sample_advanced",
        "sample_custom",
        "sample_custom_simple",
        "basic_guider",
        "cfg_guider",
        "dual_model_guider",
        "cfg_override",
        "video_linear_cfg_guidance",
        "video_triangle_cfg_guidance",
        "random_noise",
        "disable_noise",
        "basic_scheduler",
        "karras_scheduler",
        "ays_scheduler",
        "flux2_scheduler",
        "ideogram4_scheduler",
        "ltxv_scheduler",
        "sd_turbo_scheduler",
        "split_sigmas",
        "split_sigmas_denoise",
        "get_sampler",
        "set_first_sigma",
        "manual_sigmas",
        "flux_kv_cache",
    ]


def test_sample_signature_matches_contract() -> None:
    signature = inspect.signature(sample)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', latent: 'Any', "
        "steps: 'Any', cfg: 'Any', sampler_name: 'str', scheduler: 'str', "
        "seed: 'int', *, denoise: 'float' = 1.0) -> 'Any'"
    )


def test_sample_advanced_signature_matches_contract() -> None:
    signature = inspect.signature(sample_advanced)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', latent: 'Any', "
        "steps: 'Any', cfg: 'Any', sampler_name: 'str', scheduler: 'str', "
        "noise_seed: 'int', *, add_noise: 'bool' = True, "
        "return_with_leftover_noise: 'bool' = False, denoise: 'float' = 1.0, "
        "start_at_step: 'int' = 0, end_at_step: 'int' = 10000) -> 'Any'"
    )


def test_sample_custom_signature_matches_contract() -> None:
    signature = inspect.signature(sample_custom)

    assert str(signature) == (
        "(noise: 'Any', guider: 'Any', sampler: 'Any', sigmas: 'Any', "
        "latent_image: 'Any') -> 'tuple[Any, Any]'"
    )


def test_basic_guider_signature_matches_contract() -> None:
    signature = inspect.signature(basic_guider)

    assert str(signature) == "(model: 'Any', conditioning: 'Any') -> 'Any'"


def test_cfg_guider_signature_matches_contract() -> None:
    signature = inspect.signature(cfg_guider)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', negative: 'Any', cfg: 'Any') -> 'Any'"
    )


def test_dual_model_guider_signature_matches_contract() -> None:
    signature = inspect.signature(dual_model_guider)

    assert str(signature) == (
        "(model: 'Any', positive: 'Any', cfg: 'float', model_negative: 'Any', "
        "negative: 'Any | None' = None) -> 'Any'"
    )


def test_cfg_override_signature_matches_contract() -> None:
    signature = inspect.signature(cfg_override)

    assert str(signature) == (
        "(model: 'Any', cfg: 'float', start_percent: 'float' = 0.0, "
        "end_percent: 'float' = 1.0) -> 'Any'"
    )


def test_random_noise_signature_matches_contract() -> None:
    signature = inspect.signature(random_noise)

    assert str(signature) == "(noise_seed: 'int') -> 'Any'"


def test_disable_noise_signature_matches_contract() -> None:
    signature = inspect.signature(disable_noise)

    assert str(signature) == "() -> 'Any'"


def test_basic_scheduler_signature_matches_contract() -> None:
    signature = inspect.signature(basic_scheduler)

    assert str(signature) == (
        "(model: 'Any', scheduler_name: 'str', steps: 'int', "
        "denoise: 'float' = 1.0) -> 'Any'"
    )


def test_karras_scheduler_signature_matches_contract() -> None:
    signature = inspect.signature(karras_scheduler)

    assert str(signature) == (
        "(steps: 'int', sigma_max: 'float', sigma_min: 'float', rho: 'float' = 7.0) -> 'Any'"
    )


def test_ays_scheduler_signature_matches_contract() -> None:
    signature = inspect.signature(ays_scheduler)

    assert str(signature) == (
        "(model_type: 'str', steps: 'int', denoise: 'float' = 1.0) -> 'Any'"
    )


def test_flux2_scheduler_signature_matches_contract() -> None:
    signature = inspect.signature(flux2_scheduler)

    assert str(signature) == "(steps: 'int', width: 'int', height: 'int') -> 'Any'"


def test_ideogram4_scheduler_signature_matches_contract() -> None:
    signature = inspect.signature(ideogram4_scheduler)

    assert str(signature) == (
        "(steps: 'int', width: 'int', height: 'int', mu: 'float' = 0.0, "
        "std: 'float' = 1.75) -> 'Any'"
    )


def test_ltxv_scheduler_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_scheduler)

    assert str(signature) == (
        "(steps: 'int', max_shift: 'float', base_shift: 'float', *, "
        "stretch: 'bool' = True, terminal: 'float' = 0.1, latent: 'Any' = None) -> 'Any'"
    )


def test_split_sigmas_signature_matches_contract() -> None:
    signature = inspect.signature(split_sigmas)

    assert str(signature) == "(sigmas: 'Any', step: 'int') -> 'tuple[Any, Any]'"


def test_split_sigmas_denoise_signature_matches_contract() -> None:
    signature = inspect.signature(split_sigmas_denoise)

    assert str(signature) == "(sigmas: 'Any', denoise: 'float') -> 'tuple[Any, Any]'"


def test_get_sampler_signature_matches_contract() -> None:
    signature = inspect.signature(get_sampler)

    assert str(signature) == "(sampler_name: 'str') -> 'Any'"


def test_video_linear_cfg_guidance_signature_matches_contract() -> None:
    signature = inspect.signature(video_linear_cfg_guidance)

    assert str(signature) == "(model: 'Any', min_cfg: 'float') -> 'Any'"


def test_video_triangle_cfg_guidance_signature_matches_contract() -> None:
    signature = inspect.signature(video_triangle_cfg_guidance)

    assert str(signature) == "(model: 'Any', min_cfg: 'float') -> 'Any'"


def test_basic_guider_wraps_basic_guider_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    conditioning = object()

    class FakeBasicGuider:
        def __init__(self, received_model: Any) -> None:
            self.received_model = received_model
            self.received_conditioning: Any = None

        def set_conds(self, received_conditioning: Any) -> None:
            self.received_conditioning = received_conditioning

    monkeypatch.setattr(sampling_module, "_get_basic_guider_type", lambda: FakeBasicGuider)

    guider = basic_guider(model, conditioning)

    assert isinstance(guider, FakeBasicGuider)
    assert guider.received_model is model
    assert guider.received_conditioning is conditioning


def test_cfg_guider_wraps_cfg_guider_type(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object()
    positive = object()
    negative = object()
    cfg = 6.5

    class FakeCFGGuider:
        def __init__(self, received_model: Any) -> None:
            self.received_model = received_model
            self.received_positive: Any = None
            self.received_negative: Any = None
            self.received_cfg: Any = None

        def set_conds(self, received_positive: Any, received_negative: Any) -> None:
            self.received_positive = received_positive
            self.received_negative = received_negative

        def set_cfg(self, received_cfg: Any) -> None:
            self.received_cfg = received_cfg

    monkeypatch.setattr(sampling_module, "_get_cfg_guider_type", lambda: FakeCFGGuider)

    guider = cfg_guider(model, positive, negative, cfg)

    assert isinstance(guider, FakeCFGGuider)
    assert guider.received_model is model
    assert guider.received_positive is positive
    assert guider.received_negative is negative
    assert guider.received_cfg == cfg


def test_dual_model_guider_wraps_dual_model_guider_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    model_negative = object()
    positive = object()

    class FakeDualModelGuider:
        @classmethod
        def execute(
            cls,
            received_model: Any,
            received_positive: Any,
            received_cfg: float,
            *,
            model_negative: Any = None,
            negative: Any = None,
        ) -> tuple[Any]:
            return (
                {
                    "model": received_model,
                    "positive": received_positive,
                    "cfg": received_cfg,
                    "model_negative": model_negative,
                    "negative": negative,
                },
            )

    monkeypatch.setattr(
        sampling_module,
        "_get_dual_model_guider_type",
        lambda: FakeDualModelGuider,
    )

    guider = dual_model_guider(model, positive, 7.0, model_negative)

    assert guider == {
        "model": model,
        "positive": positive,
        "cfg": 7.0,
        "model_negative": model_negative,
        "negative": None,
    }


def test_cfg_override_wraps_cfg_override_type(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object()

    class FakeCFGOverride:
        @classmethod
        def execute(
            cls,
            received_model: Any,
            received_cfg: float,
            received_start_percent: float,
            received_end_percent: float,
        ) -> tuple[Any]:
            return (
                {
                    "model": received_model,
                    "cfg": received_cfg,
                    "start_percent": received_start_percent,
                    "end_percent": received_end_percent,
                },
            )

    monkeypatch.setattr(sampling_module, "_get_cfg_override_type", lambda: FakeCFGOverride)

    assert cfg_override(model, 3.0, 0.7, 1.0) == {
        "model": model,
        "cfg": 3.0,
        "start_percent": 0.7,
        "end_percent": 1.0,
    }


def test_ideogram4_scheduler_wraps_ideogram4_scheduler_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeIdeogram4Scheduler:
        @classmethod
        def execute(
            cls,
            steps: int,
            width: int,
            height: int,
            mu: float,
            std: float,
        ) -> tuple[Any]:
            return (("sigmas", steps, width, height, mu, std),)

    monkeypatch.setattr(
        sampling_module,
        "_get_ideogram4_scheduler_type",
        lambda: FakeIdeogram4Scheduler,
    )

    assert ideogram4_scheduler(48, 1024, 768, 0.25, 1.5) == (
        "sigmas",
        48,
        1024,
        768,
        0.25,
        1.5,
    )


class _FakeVectorTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.shape = (len(values), 1, 1, 1)
        self.created_scale_values: list[float] | None = None

    def new_tensor(self, values: list[float]) -> _FakeScaleTensor:
        self.created_scale_values = list(values)
        return _FakeScaleTensor(values)

    def __sub__(self, other: _FakeVectorTensor) -> _FakeVectorTensor:
        return _FakeVectorTensor(
            [left - right for left, right in zip(self.values, other.values, strict=True)]
        )

    def __add__(self, other: _FakeVectorTensor) -> _FakeVectorTensor:
        return _FakeVectorTensor(
            [left + right for left, right in zip(self.values, other.values, strict=True)]
        )


class _FakeScaleTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.reshape_shape: tuple[int, int, int, int] | None = None

    def reshape(self, shape: tuple[int, int, int, int]) -> _FakeScaleTensor:
        self.reshape_shape = shape
        return self

    def __mul__(self, other: _FakeVectorTensor) -> _FakeVectorTensor:
        return _FakeVectorTensor(
            [scale * value for scale, value in zip(self.values, other.values, strict=True)]
        )


def test_video_linear_cfg_guidance_returns_patched_model_clone() -> None:
    class FakePatchedModel:
        def __init__(self) -> None:
            self.cfg_function: Any = None

        def set_model_sampler_cfg_function(self, cfg_function: Any) -> None:
            self.cfg_function = cfg_function

    class FakeModel:
        def __init__(self) -> None:
            self.clone_calls = 0
            self.patched = FakePatchedModel()

        def clone(self) -> FakePatchedModel:
            self.clone_calls += 1
            return self.patched

    model = FakeModel()

    patched = video_linear_cfg_guidance(model, min_cfg=1.25)

    assert model.clone_calls == 1
    assert patched is model.patched
    assert callable(patched.cfg_function)


def test_video_linear_cfg_guidance_scales_from_full_cfg_to_min_cfg_linearly() -> None:
    class FakePatchedModel:
        def __init__(self) -> None:
            self.cfg_function: Any = None

        def set_model_sampler_cfg_function(self, cfg_function: Any) -> None:
            self.cfg_function = cfg_function

    class FakeModel:
        def __init__(self) -> None:
            self.patched = FakePatchedModel()

        def clone(self) -> FakePatchedModel:
            return self.patched

    cond = _FakeVectorTensor([10.0, 20.0, 30.0, 40.0])
    uncond = _FakeVectorTensor([2.0, 2.0, 2.0, 2.0])

    patched = video_linear_cfg_guidance(FakeModel(), min_cfg=1.0)
    assert callable(patched.cfg_function)

    output = patched.cfg_function({"cond": cond, "uncond": uncond, "cond_scale": 4.0})

    assert cond.created_scale_values == [4.0, 3.0, 2.0, 1.0]
    assert output.values == [34.0, 56.0, 58.0, 40.0]


def test_video_triangle_cfg_guidance_returns_patched_model_clone() -> None:
    class FakePatchedModel:
        def __init__(self) -> None:
            self.cfg_function: Any = None

        def set_model_sampler_cfg_function(self, cfg_function: Any) -> None:
            self.cfg_function = cfg_function

    class FakeModel:
        def __init__(self) -> None:
            self.clone_calls = 0
            self.patched = FakePatchedModel()

        def clone(self) -> FakePatchedModel:
            self.clone_calls += 1
            return self.patched

    model = FakeModel()

    patched = video_triangle_cfg_guidance(model, min_cfg=1.25)

    assert model.clone_calls == 1
    assert patched is model.patched
    assert callable(patched.cfg_function)


def test_video_triangle_cfg_guidance_scales_from_min_cfg_to_middle_peak() -> None:
    class FakePatchedModel:
        def __init__(self) -> None:
            self.cfg_function: Any = None

        def set_model_sampler_cfg_function(self, cfg_function: Any) -> None:
            self.cfg_function = cfg_function

    class FakeModel:
        def __init__(self) -> None:
            self.patched = FakePatchedModel()

        def clone(self) -> FakePatchedModel:
            return self.patched

    cond = _FakeVectorTensor([12.0, 14.0, 16.0, 18.0, 20.0])
    uncond = _FakeVectorTensor([2.0, 2.0, 2.0, 2.0, 2.0])

    patched = video_triangle_cfg_guidance(FakeModel(), min_cfg=1.0)
    assert callable(patched.cfg_function)

    output = patched.cfg_function({"cond": cond, "uncond": uncond, "cond_scale": 4.0})

    assert cond.created_scale_values == [1.0, 2.5, 4.0, 2.5, 1.0]
    assert output.values == [12.0, 32.0, 58.0, 42.0, 20.0]


def test_random_noise_wraps_random_noise_type(monkeypatch: pytest.MonkeyPatch) -> None:
    noise_seed = 1_234_567_890

    class _FakeNoiseObj:
        def __init__(self, seed: int) -> None:
            self.received_noise_seed = seed

    fake_obj = _FakeNoiseObj(noise_seed)

    class FakeRandomNoise:
        @classmethod
        def execute(cls, seed: int) -> tuple[Any, ...]:
            return (fake_obj,)

    monkeypatch.setattr(sampling_module, "_get_random_noise_type", lambda: FakeRandomNoise)

    noise = random_noise(noise_seed)

    assert noise is fake_obj
    assert noise.received_noise_seed == noise_seed


def test_disable_noise_wraps_disable_noise_type(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeNoiseObj:
        created = True

    fake_obj = _FakeNoiseObj()

    class FakeDisableNoise:
        @classmethod
        def execute(cls) -> tuple[Any, ...]:
            return (fake_obj,)

    monkeypatch.setattr(sampling_module, "_get_disable_noise_type", lambda: FakeDisableNoise)

    noise = disable_noise()

    assert noise is fake_obj
    assert noise.created is True


def test_basic_scheduler_wraps_basic_scheduler_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    expected_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeBasicScheduler:
        @classmethod
        def execute(
            cls,
            received_model: Any,
            received_scheduler: str,
            received_steps: int,
            received_denoise: float,
        ) -> tuple[Any]:
            recorded["args"] = (
                received_model,
                received_scheduler,
                received_steps,
                received_denoise,
            )
            return (expected_sigmas,)

    monkeypatch.setattr(sampling_module, "_get_basic_scheduler_type", lambda: FakeBasicScheduler)
    result = basic_scheduler(model, "normal", 20, 0.6)

    assert recorded["args"] == (model, "normal", 20, 0.6)
    assert result is expected_sigmas


def test_karras_scheduler_wraps_karras_scheduler_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeKarrasScheduler:
        @classmethod
        def execute(
            cls,
            received_steps: int,
            received_sigma_max: float,
            received_sigma_min: float,
            received_rho: float,
        ) -> tuple[Any]:
            recorded["args"] = (
                received_steps,
                received_sigma_max,
                received_sigma_min,
                received_rho,
            )
            return (expected_sigmas,)

    monkeypatch.setattr(
        sampling_module, "_get_karras_scheduler_type", lambda: FakeKarrasScheduler
    )
    result = karras_scheduler(25, 14.61, 0.03)

    assert recorded["args"] == (25, 14.61, 0.03, 7.0)
    assert result is expected_sigmas


def test_ays_scheduler_wraps_ays_scheduler_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeAysScheduler:
        @classmethod
        def execute(
            cls, received_model_type: str, received_steps: int, received_denoise: float
        ) -> tuple[Any]:
            recorded["args"] = (received_model_type, received_steps, received_denoise)
            return (expected_sigmas,)

    monkeypatch.setattr(sampling_module, "_get_ays_scheduler_type", lambda: FakeAysScheduler)
    result = ays_scheduler("SDXL", 18, 0.75)

    assert recorded["args"] == ("SDXL", 18, 0.75)
    assert result is expected_sigmas


def test_ays_scheduler_rejects_invalid_model_type() -> None:
    with pytest.raises(ValueError, match="model_type must be one of"):
        ays_scheduler("FLUX", 20)


def test_flux2_scheduler_wraps_flux2_scheduler_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeFlux2Scheduler:
        @classmethod
        def execute(
            cls, received_steps: int, received_width: int, received_height: int
        ) -> tuple[Any]:
            recorded["args"] = (received_steps, received_width, received_height)
            return (expected_sigmas,)

    monkeypatch.setattr(sampling_module, "_get_flux2_scheduler_type", lambda: FakeFlux2Scheduler)
    result = flux2_scheduler(12, 1024, 768)

    assert recorded["args"] == (12, 1024, 768)
    assert result is expected_sigmas


def test_ltxv_scheduler_wraps_ltxv_scheduler_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_sigmas = object()
    recorded: dict[str, Any] = {}
    latent = {"samples": object()}

    class FakeLtxvScheduler:
        @classmethod
        def execute(
            cls,
            received_steps: int,
            received_max_shift: float,
            received_base_shift: float,
            received_stretch: bool,
            received_terminal: float,
            received_latent: Any,
        ) -> tuple[Any]:
            recorded["args"] = (
                received_steps,
                received_max_shift,
                received_base_shift,
                received_stretch,
                received_terminal,
                received_latent,
            )
            return (expected_sigmas,)

    monkeypatch.setattr(sampling_module, "_get_ltxv_scheduler_type", lambda: FakeLtxvScheduler)
    result = ltxv_scheduler(
        30,
        2.05,
        0.95,
        stretch=False,
        terminal=0.2,
        latent=latent,
    )

    assert recorded["args"] == (30, 2.05, 0.95, False, 0.2, latent)
    assert result is expected_sigmas


def test_scheduler_wrappers_extract_from_nodeoutput_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNodeOutput:
        def __init__(self, value: Any) -> None:
            self.result = (value,)

    basic_value = object()
    karras_value = object()
    ays_value = object()
    flux2_value = object()
    ltxv_value = object()

    class FakeBasicScheduler:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(basic_value)

    class FakeKarrasScheduler:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(karras_value)

    class FakeAysScheduler:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(ays_value)

    class FakeFlux2Scheduler:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(flux2_value)

    class FakeLtxvScheduler:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(ltxv_value)

    monkeypatch.setattr(sampling_module, "_get_basic_scheduler_type", lambda: FakeBasicScheduler)
    monkeypatch.setattr(sampling_module, "_get_karras_scheduler_type", lambda: FakeKarrasScheduler)
    monkeypatch.setattr(sampling_module, "_get_ays_scheduler_type", lambda: FakeAysScheduler)
    monkeypatch.setattr(sampling_module, "_get_flux2_scheduler_type", lambda: FakeFlux2Scheduler)
    monkeypatch.setattr(sampling_module, "_get_ltxv_scheduler_type", lambda: FakeLtxvScheduler)

    assert basic_scheduler(object(), "normal", 10) is basic_value
    assert karras_scheduler(10, 14.6, 0.03) is karras_value
    assert ays_scheduler("SD1", 10) is ays_value
    assert flux2_scheduler(10, 1024, 1024) is flux2_value
    assert ltxv_scheduler(10, 2.05, 0.95) is ltxv_value


def test_split_sigmas_wraps_split_sigmas_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sigmas = object()
    high_sigmas = object()
    low_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeSplitSigmas:
        @classmethod
        def execute(cls, received_sigmas: Any, received_step: int) -> tuple[Any, Any]:
            recorded["args"] = (received_sigmas, received_step)
            return high_sigmas, low_sigmas

    monkeypatch.setattr(sampling_module, "_get_split_sigmas_type", lambda: FakeSplitSigmas)
    first, second = split_sigmas(sigmas, 6)

    assert recorded["args"] == (sigmas, 6)
    assert first is high_sigmas
    assert second is low_sigmas


def test_split_sigmas_denoise_wraps_split_sigmas_denoise_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sigmas = object()
    high_sigmas = object()
    low_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeSplitSigmasDenoise:
        @classmethod
        def execute(cls, received_sigmas: Any, received_denoise: float) -> tuple[Any, Any]:
            recorded["args"] = (received_sigmas, received_denoise)
            return high_sigmas, low_sigmas

    monkeypatch.setattr(
        sampling_module,
        "_get_split_sigmas_denoise_type",
        lambda: FakeSplitSigmasDenoise,
    )
    first, second = split_sigmas_denoise(sigmas, 0.55)

    assert recorded["args"] == (sigmas, 0.55)
    assert first is high_sigmas
    assert second is low_sigmas


def test_get_sampler_wraps_ksampler_select_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = object()
    recorded: dict[str, Any] = {}

    class FakeKSamplerSelect:
        @classmethod
        def execute(cls, received_sampler_name: str) -> tuple[Any]:
            recorded["args"] = (received_sampler_name,)
            return (sampler,)

    monkeypatch.setattr(
        sampling_module,
        "_get_ksampler_select_type",
        lambda: FakeKSamplerSelect,
    )
    result = get_sampler("dpmpp_2m")

    assert recorded["args"] == ("dpmpp_2m",)
    assert result is sampler


def test_split_and_sampler_wrappers_extract_from_nodeoutput_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_first = object()
    split_second = object()
    denoise_first = object()
    denoise_second = object()
    sampler = object()

    class FakeNodeOutput:
        def __init__(self, *values: Any) -> None:
            self.result = values

    class FakeSplitSigmas:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(split_first, split_second)

    class FakeSplitSigmasDenoise:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(denoise_first, denoise_second)

    class FakeKSamplerSelect:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(sampler)

    monkeypatch.setattr(sampling_module, "_get_split_sigmas_type", lambda: FakeSplitSigmas)
    monkeypatch.setattr(
        sampling_module,
        "_get_split_sigmas_denoise_type",
        lambda: FakeSplitSigmasDenoise,
    )
    monkeypatch.setattr(sampling_module, "_get_ksampler_select_type", lambda: FakeKSamplerSelect)

    assert split_sigmas(object(), 3) == (split_first, split_second)
    assert split_sigmas_denoise(object(), 0.8) == (denoise_first, denoise_second)
    assert get_sampler("euler") is sampler


def test_sample_returns_raw_denoised_latent_without_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_latent = {"samples": object(), "batch_index": [0]}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        return (raw_latent,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        seed=123,
    )

    assert result is raw_latent


def test_sample_uses_common_ksampler_call_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    positive = object()
    negative = object()
    latent = {"samples": object()}
    expected = object()
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return (expected,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent,
        steps=30,
        cfg=5.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        seed=42,
        denoise=0.65,
    )

    assert result is expected
    assert recorded["args"] == (
        model,
        42,
        30,
        5.5,
        "dpmpp_2m",
        "karras",
        positive,
        negative,
        latent,
    )
    assert recorded["kwargs"] == {"denoise": 0.65}


def test_sample_advanced_returns_raw_denoised_latent_without_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_latent = {"samples": object(), "batch_index": [0]}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        return (raw_latent,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample_advanced(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        noise_seed=123,
    )

    assert result is raw_latent


def test_sample_advanced_uses_common_ksampler_call_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    positive = object()
    negative = object()
    latent = {"samples": object()}
    expected = object()
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return (expected,)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    result = sample_advanced(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent,
        steps=30,
        cfg=5.5,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        noise_seed=42,
        denoise=0.65,
        start_at_step=3,
        end_at_step=20,
    )

    assert result is expected
    assert recorded["args"] == (
        model,
        42,
        30,
        5.5,
        "dpmpp_2m",
        "karras",
        positive,
        negative,
        latent,
    )
    assert recorded["kwargs"] == {
        "denoise": 0.65,
        "disable_noise": False,
        "start_step": 3,
        "last_step": 20,
        "force_full_denoise": True,
    }


def test_sample_advanced_disables_noise_when_add_noise_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["kwargs"] = kwargs
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample_advanced(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name="euler",
        scheduler="normal",
        noise_seed=999,
        add_noise=False,
    )

    assert recorded["kwargs"]["disable_noise"] is True


def test_sample_advanced_returns_with_leftover_noise_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["kwargs"] = kwargs
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample_advanced(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name="euler",
        scheduler="normal",
        noise_seed=999,
        return_with_leftover_noise=True,
    )

    assert recorded["kwargs"]["force_full_denoise"] is False


def test_sample_custom_wraps_sampler_custom_advanced_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    noise = object()
    guider = object()
    sampler = object()
    sigmas = object()
    latent_image = {"samples": object()}
    output_latent = {"samples": object()}
    denoised_latent = {"samples": object()}
    recorded: dict[str, Any] = {}

    class FakeSamplerCustomAdvanced:
        @classmethod
        def execute(
            cls,
            received_noise: Any,
            received_guider: Any,
            received_sampler: Any,
            received_sigmas: Any,
            received_latent_image: Any,
        ) -> tuple[Any, Any]:
            recorded["args"] = (
                received_noise,
                received_guider,
                received_sampler,
                received_sigmas,
                received_latent_image,
            )
            return output_latent, denoised_latent

    monkeypatch.setattr(
        sampling_module,
        "_get_sampler_custom_advanced_type",
        lambda: FakeSamplerCustomAdvanced,
    )

    output, denoised = sample_custom(noise, guider, sampler, sigmas, latent_image)

    assert recorded["args"] == (noise, guider, sampler, sigmas, latent_image)
    assert output is output_latent
    assert denoised is denoised_latent
    assert isinstance(output, dict)
    assert isinstance(denoised, dict)


def test_sample_custom_extracts_from_nodeoutput_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_latent = {"samples": object()}
    denoised_latent = {"samples": object()}

    class FakeNodeOutput:
        def __init__(self) -> None:
            self.result = (output_latent, denoised_latent)

    class FakeSamplerCustomAdvanced:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput()

    monkeypatch.setattr(
        sampling_module,
        "_get_sampler_custom_advanced_type",
        lambda: FakeSamplerCustomAdvanced,
    )

    output, denoised = sample_custom(object(), object(), object(), object(), object())

    assert output is output_latent
    assert denoised is denoised_latent


def test_sample_custom_end_to_end_dummy_latent_basic_guider_basic_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeNoiseObj:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    class FakeRandomNoise:
        @classmethod
        def execute(cls, noise_seed: int) -> tuple[Any, ...]:
            return (_FakeNoiseObj(noise_seed),)

    class FakeBasicGuider:
        def __init__(self, model: Any) -> None:
            self.model = model
            self.conditioning: Any = None

        def set_conds(self, conditioning: Any) -> None:
            self.conditioning = conditioning

    class FakeBasicScheduler:
        @classmethod
        def execute(
            cls, model: Any, scheduler_name: str, steps: int, denoise: float
        ) -> tuple[list[float]]:
            return ([1.0, 0.5, 0.0],)

    class FakeSamplerCustomAdvanced:
        @classmethod
        def execute(
            cls, noise: Any, guider: Any, sampler: Any, sigmas: Any, latent_image: Any
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            output_latent = dict(latent_image)
            denoised_latent = dict(latent_image)
            output_latent["noise_seed"] = noise.seed
            output_latent["sigmas"] = sigmas
            output_latent["guider_model"] = guider.model
            output_latent["sampler"] = sampler
            denoised_latent["samples"] = latent_image["samples"]
            return output_latent, denoised_latent

    monkeypatch.setattr(sampling_module, "_get_random_noise_type", lambda: FakeRandomNoise)
    monkeypatch.setattr(sampling_module, "_get_basic_guider_type", lambda: FakeBasicGuider)
    monkeypatch.setattr(sampling_module, "_get_basic_scheduler_type", lambda: FakeBasicScheduler)
    monkeypatch.setattr(
        sampling_module,
        "_get_sampler_custom_advanced_type",
        lambda: FakeSamplerCustomAdvanced,
    )

    model = object()
    conditioning = object()
    sampler = object()
    latent_image = {"samples": object(), "batch_index": [0]}
    noise = random_noise(1234)
    guider = basic_guider(model, conditioning)
    sigmas = basic_scheduler(model, "normal", 3)

    output_latent, denoised_latent = sample_custom(
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent_image,
    )

    assert isinstance(output_latent, dict)
    assert isinstance(denoised_latent, dict)
    assert output_latent["noise_seed"] == 1234
    assert output_latent["sigmas"] == [1.0, 0.5, 0.0]
    assert output_latent["guider_model"] is model
    assert output_latent["sampler"] is sampler
    assert denoised_latent["samples"] is latent_image["samples"]


def test_sample_passes_sampler_and_scheduler_strings_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    sampler_name = " Euler++ custom "
    scheduler = "Normal/alt schedule "

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=10,
        cfg=4.0,
        sampler_name=sampler_name,
        scheduler=scheduler,
        seed=999,
    )

    assert recorded["args"][4] == sampler_name
    assert recorded["args"][5] == scheduler


def test_sample_passes_seed_to_common_ksampler_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    seed = 4_294_967_295

    def common_ksampler(*args: Any, **kwargs: Any) -> tuple[Any]:
        recorded["args"] = args
        return (object(),)

    monkeypatch.setattr(sampling_module, "_get_common_ksampler", lambda: common_ksampler)

    sample(
        model=object(),
        positive=object(),
        negative=object(),
        latent={"samples": object()},
        steps=20,
        cfg=6.5,
        sampler_name="euler",
        scheduler="normal",
        seed=seed,
    )

    assert recorded["args"][1] == seed


def test_uv_run_python_imports_sample_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import sample; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_sample_advanced_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import sample_advanced; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_sample_custom_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import sample_custom; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_basic_guider_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import basic_guider; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_cfg_guider_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import cfg_guider; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_video_linear_cfg_guidance_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import video_linear_cfg_guidance; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_video_triangle_cfg_guidance_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import video_triangle_cfg_guidance; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_random_noise_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import random_noise; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_disable_noise_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import disable_noise; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_basic_scheduler_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import basic_scheduler; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_karras_scheduler_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import karras_scheduler; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_ays_scheduler_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import ays_scheduler; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_flux2_scheduler_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import flux2_scheduler; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_ltxv_scheduler_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import ltxv_scheduler; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_split_sigmas_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import split_sigmas; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_split_sigmas_denoise_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import split_sigmas_denoise; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_uv_run_python_imports_get_sampler_on_cpu_only_machine_smoke() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            "from comfy_diffusion.sampling import get_sampler; print('ok')",
        ],
        cwd=_repo_root(),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_import_comfy_diffusion_sampling_has_no_additional_heavy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_path = list(sys.path)\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.sampling import "
        "ays_scheduler, basic_guider, basic_scheduler, cfg_guider, disable_noise, "
        "flux2_scheduler, get_sampler, karras_scheduler, ltxv_scheduler, manual_sigmas, "
        "random_noise, sample, sample_advanced, sample_custom, split_sigmas, "
        "split_sigmas_denoise, video_linear_cfg_guidance, video_triangle_cfg_guidance\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'ays_name': ays_scheduler.__name__,\n"
        "  'basic_scheduler_name': basic_scheduler.__name__,\n"
        "  'flux2_name': flux2_scheduler.__name__,\n"
        "  'func_name': sample.__name__,\n"
        "  'advanced_name': sample_advanced.__name__,\n"
        "  'custom_name': sample_custom.__name__,\n"
        "  'basic_name': basic_guider.__name__,\n"
        "  'cfg_name': cfg_guider.__name__,\n"
        "  'karras_name': karras_scheduler.__name__,\n"
        "  'ltxv_name': ltxv_scheduler.__name__,\n"
        "  'split_name': split_sigmas.__name__,\n"
        "  'split_denoise_name': split_sigmas_denoise.__name__,\n"
        "  'sampler_name': get_sampler.__name__,\n"
        "  'video_linear_cfg_name': video_linear_cfg_guidance.__name__,\n"
        "  'video_triangle_cfg_name': video_triangle_cfg_guidance.__name__,\n"
        "  'random_name': random_noise.__name__,\n"
        "  'disable_name': disable_noise.__name__,\n"
        "  'manual_sigmas_name': manual_sigmas.__name__,\n"
        "  'path_unchanged': baseline_path == list(sys.path),\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'nodes_loaded': 'nodes' in sys.modules,\n"
        "  'folder_paths_loaded': 'folder_paths' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["ays_name"] == "ays_scheduler"
    assert payload["basic_scheduler_name"] == "basic_scheduler"
    assert payload["flux2_name"] == "flux2_scheduler"
    assert payload["func_name"] == "sample"
    assert payload["advanced_name"] == "sample_advanced"
    assert payload["custom_name"] == "sample_custom"
    assert payload["basic_name"] == "basic_guider"
    assert payload["cfg_name"] == "cfg_guider"
    assert payload["karras_name"] == "karras_scheduler"
    assert payload["ltxv_name"] == "ltxv_scheduler"
    assert payload["split_name"] == "split_sigmas"
    assert payload["split_denoise_name"] == "split_sigmas_denoise"
    assert payload["sampler_name"] == "get_sampler"
    assert payload["video_linear_cfg_name"] == "video_linear_cfg_guidance"
    assert payload["video_triangle_cfg_name"] == "video_triangle_cfg_guidance"
    assert payload["random_name"] == "random_noise"
    assert payload["disable_name"] == "disable_noise"
    assert payload["manual_sigmas_name"] == "manual_sigmas"
    assert payload["path_unchanged"] is True
    assert payload["torch_loaded"] is False
    assert payload["nodes_loaded"] is False
    assert payload["folder_paths_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module == "nodes"
        or module.startswith(("torch", "folder_paths", "comfy.", "numpy"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"


def test_manual_sigmas_signature_matches_contract() -> None:
    signature = inspect.signature(manual_sigmas)

    assert str(signature) == "(sigmas: 'str') -> 'Any'"


def test_manual_sigmas_parses_space_separated_integers() -> None:
    import torch

    result = manual_sigmas("14 7 3 0")

    assert isinstance(result, torch.FloatTensor)
    assert list(result) == [14.0, 7.0, 3.0, 0.0]


def test_manual_sigmas_parses_decimal_values() -> None:
    import torch

    result = manual_sigmas("3.5, 1.75, 0.875")

    assert isinstance(result, torch.FloatTensor)
    assert result.tolist() == pytest.approx([3.5, 1.75, 0.875])


def test_manual_sigmas_parses_negative_values() -> None:
    import torch

    result = manual_sigmas("-1.5 0.0 1.5")

    assert isinstance(result, torch.FloatTensor)
    assert result.tolist() == pytest.approx([-1.5, 0.0, 1.5])


def test_manual_sigmas_returns_float_tensor() -> None:
    import torch

    result = manual_sigmas("1 2 3")

    assert result.dtype == torch.float32


def test_manual_sigmas_is_in_all() -> None:
    assert "manual_sigmas" in sampling_module.__all__


def test_manual_sigmas_torch_import_is_deferred() -> None:
    result = _run_python(
        "import sys\n"
        "import json\n"
        "baseline = set(sys.modules)\n"
        "from comfy_diffusion.sampling import manual_sigmas\n"
        "post = set(sys.modules)\n"
        "new = sorted(post - baseline)\n"
        "torch_loaded = 'torch' in sys.modules\n"
        "print(json.dumps({'torch_loaded': torch_loaded, 'func_name': manual_sigmas.__name__}))\n"
    )

    import json as _json

    payload = _json.loads(result.stdout)
    assert payload["func_name"] == "manual_sigmas"
    assert payload["torch_loaded"] is False


def test_set_first_sigma_signature_matches_contract() -> None:
    signature = inspect.signature(set_first_sigma)

    assert str(signature) == "(sigmas: 'Any', sigma_override: 'float') -> 'Any'"


def test_set_first_sigma_wraps_set_first_sigma_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sigmas = object()
    modified_sigmas = object()
    recorded: dict[str, Any] = {}

    class FakeSetFirstSigma:
        @classmethod
        def execute(cls, received_sigmas: Any, received_sigma_override: float) -> tuple[Any]:
            recorded["args"] = (received_sigmas, received_sigma_override)
            return (modified_sigmas,)

    monkeypatch.setattr(
        sampling_module,
        "_get_set_first_sigma_type",
        lambda: FakeSetFirstSigma,
    )
    result = set_first_sigma(sigmas, 14.5)

    assert recorded["args"] == (sigmas, 14.5)
    assert result is modified_sigmas


def test_set_first_sigma_extracts_from_nodeoutput_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modified_sigmas = object()

    class FakeNodeOutput:
        def __init__(self, value: Any) -> None:
            self.result = (value,)

    class FakeSetFirstSigma:
        @classmethod
        def execute(cls, *args: Any) -> FakeNodeOutput:
            return FakeNodeOutput(modified_sigmas)

    monkeypatch.setattr(
        sampling_module,
        "_get_set_first_sigma_type",
        lambda: FakeSetFirstSigma,
    )
    result = set_first_sigma(object(), 7.25)

    assert result is modified_sigmas


def test_set_first_sigma_lazy_import_does_not_load_comfy_at_module_level() -> None:
    result = _run_python(
        "import json, sys\n"
        "from comfy_diffusion.sampling import set_first_sigma\n"
        "comfy_extras_loaded = 'comfy_extras' in sys.modules\n"
        "print(json.dumps({'comfy_extras_loaded': comfy_extras_loaded, 'func_name': set_first_sigma.__name__}))\n"
    )

    import json as _json

    payload = _json.loads(result.stdout)
    assert payload["func_name"] == "set_first_sigma"
    assert payload["comfy_extras_loaded"] is False



def test_flux_kv_cache_in_all() -> None:
    assert "flux_kv_cache" in sampling_module.__all__


def test_flux_kv_cache_accepts_mock_model_and_returns_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_model = object()
    patched_model = object()

    class FakeFluxKVCache:
        @classmethod
        def execute(cls, model: Any) -> tuple[Any]:
            return (patched_model,)

    monkeypatch.setattr(
        sampling_module,
        "_get_flux_kv_cache_type",
        lambda: FakeFluxKVCache,
    )
    from comfy_diffusion.sampling import flux_kv_cache

    result = flux_kv_cache(mock_model)
    assert result is patched_model


def test_flux_kv_cache_lazy_import_does_not_load_comfy_at_module_level() -> None:
    result = _run_python(
        "import json, sys\n"
        "from comfy_diffusion.sampling import flux_kv_cache\n"
        "comfy_extras_loaded = 'comfy_extras' in sys.modules\n"
        "print(json.dumps({'comfy_extras_loaded': comfy_extras_loaded, 'func_name': flux_kv_cache.__name__}))\n"
    )

    import json as _json

    payload = _json.loads(result.stdout)
    assert payload["func_name"] == "flux_kv_cache"
    assert payload["comfy_extras_loaded"] is False
