"""Tests for LTXV audio VAE helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import comfy_diffusion.audio as audio_module
from comfy_diffusion.audio import (
    audio_crop,
    audio_encoder_encode,
    audio_separation,
    empty_ace_step_15_latent_audio,
    encode_ace_step_15_audio,
    load_audio,
    ltxv_audio_vae_decode,
    ltxv_audio_vae_encode,
    ltxv_concat_av_latent,
    ltxv_empty_latent_audio,
    ltxv_separate_av_latent,
    trim_audio_duration,
    vae_decode_audio,
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


def test_audio_module_exports_ltxv_audio_vae_helpers() -> None:
    assert audio_module.__all__ == [
        "ltxv_audio_vae_encode",
        "ltxv_audio_vae_decode",
        "ltxv_empty_latent_audio",
        "encode_ace_step_15_audio",
        "empty_ace_step_15_latent_audio",
        "ltxv_concat_av_latent",
        "ltxv_separate_av_latent",
        "load_audio",
        "audio_crop",
        "audio_separation",
        "trim_audio_duration",
        "ltxv_audio_video_mask",
        "audio_encoder_encode",
        "vae_decode_audio",
    ]


def test_ltxv_audio_vae_encode_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_audio_vae_encode)
    assert str(signature) == "(vae: '_LtxvAudioVaeEncoder', audio: 'Any') -> 'dict[str, Any]'"


def test_ltxv_audio_vae_encode_calls_vae_encode_and_returns_latent_dict() -> None:
    expected_audio = object()
    expected_latent = object()

    class FakeAudioVae:
        sample_rate = 44100

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def encode(self, audio: Any) -> Any:
            self.calls.append(audio)
            return expected_latent

    vae = FakeAudioVae()
    result = ltxv_audio_vae_encode(vae, expected_audio)

    assert result == {"samples": expected_latent, "sample_rate": 44100, "type": "audio"}
    assert vae.calls == [expected_audio]


def test_ltxv_audio_vae_decode_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_audio_vae_decode)
    assert str(signature) == "(vae: '_LtxvAudioVaeDecoder', latent: 'Any') -> 'dict[str, Any]'"


def test_ltxv_audio_vae_decode_calls_vae_decode_and_returns_audio_dict() -> None:
    expected_device = "fake-device"

    class FakeAudioTensor:
        device = expected_device
        to_calls: list[Any] = []

        def to(self, device: Any) -> "FakeAudioTensor":
            FakeAudioTensor.to_calls.append(device)
            return self

    expected_raw_latent = object()
    fake_audio = FakeAudioTensor()

    class FakeLatentTensor:
        device = expected_device

    fake_latent_tensor = FakeLatentTensor()

    class FakeAudioVae:
        output_sample_rate = 22050

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def decode(self, latent: Any) -> FakeAudioTensor:
            self.calls.append(latent)
            return fake_audio

    # Test 1: raw tensor input
    vae = FakeAudioVae()
    result = ltxv_audio_vae_decode(vae, fake_latent_tensor)

    assert result["waveform"] is fake_audio
    assert result["sample_rate"] == 22050
    assert vae.calls == [fake_latent_tensor]

    # Test 2: dict latent input (as produced by ltxv_audio_vae_encode)
    vae2 = FakeAudioVae()
    FakeAudioTensor.to_calls.clear()
    latent_dict = {"samples": fake_latent_tensor, "sample_rate": 44100, "type": "audio"}
    result2 = ltxv_audio_vae_decode(vae2, latent_dict)

    assert result2["waveform"] is fake_audio
    assert result2["sample_rate"] == 22050
    assert vae2.calls == [fake_latent_tensor]


def test_ltxv_audio_vae_decode_supports_comfy_vae_wrapper() -> None:
    class FakeAudioTensor:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        def movedim(self, source: int, destination: int) -> "FakeAudioTensor":
            self.calls.append(("movedim", (source, destination)))
            return self

        def to(self, device: Any) -> "FakeAudioTensor":
            self.calls.append(("to", device))
            return self

    class FakeFirstStageModel:
        output_sample_rate = 48000

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def decode(self, latent: Any) -> FakeAudioTensor:
            self.calls.append(latent)
            return fake_audio

    class FakeLatentTensor:
        device = "cuda:0"

    fake_audio = FakeAudioTensor()
    fake_latent = FakeLatentTensor()

    class FakeComfyVae:
        def __init__(self) -> None:
            self.first_stage_model = FakeFirstStageModel()

    vae = FakeComfyVae()
    result = ltxv_audio_vae_decode(vae, {"samples": fake_latent})

    assert result == {"waveform": fake_audio, "sample_rate": 48000}
    assert vae.first_stage_model.calls == [fake_latent]
    assert fake_audio.calls == [("to", "cuda:0")]


def test_ltxv_empty_latent_audio_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_empty_latent_audio)
    assert (
        str(signature)
        == "(audio_vae: '_LtxvAudioVae', frames_number: 'int', "
        "frame_rate: 'int' = 25, batch_size: 'int' = 1) -> 'dict[str, Any]'"
    )


def test_ltxv_empty_latent_audio_wraps_node_execute_and_returns_audio_latent_dict(
    monkeypatch: Any,
) -> None:
    expected_samples = object()

    class FakeAudioVae:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self.sample_rate = 48000
            self.latent_channels = 32
            self.latent_frequency_bins = 128

        def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
            self.calls.append((frames_number, frame_rate))
            return 123

    class FakeLtxvEmptyLatentAudio:
        @classmethod
        def execute(
            cls,
            *,
            frames_number: int,
            frame_rate: int,
            batch_size: int,
            audio_vae: Any,
        ) -> tuple[dict[str, Any]]:
            num_latents = audio_vae.num_of_latents_from_frames(frames_number, frame_rate)
            return (
                {
                    "samples": expected_samples,
                    "sample_rate": int(audio_vae.sample_rate),
                    "type": "audio",
                    "num_latents": num_latents,
                    "batch_size": batch_size,
                },
            )

    monkeypatch.setattr(
        audio_module,
        "_get_ltxv_empty_latent_audio_type",
        lambda: FakeLtxvEmptyLatentAudio,
    )

    audio_vae = FakeAudioVae()
    result = ltxv_empty_latent_audio(
        audio_vae=audio_vae,
        frames_number=97,
        frame_rate=25,
        batch_size=2,
    )

    assert result["samples"] is expected_samples
    assert result["sample_rate"] == 48000
    assert result["type"] == "audio"
    assert result["num_latents"] == 123
    assert result["batch_size"] == 2
    assert audio_vae.calls == [(97, 25)]


def test_encode_ace_step_15_audio_signature_matches_contract() -> None:
    signature = inspect.signature(encode_ace_step_15_audio)
    assert (
        str(signature)
        == "(clip: '_AceStep15Clip', tags: 'str', lyrics: 'str' = '', seed: 'int' = 0, "
        "bpm: 'int' = 120, duration: 'float' = 120.0, timesignature: 'str' = '4', "
        "language: 'str' = 'en', keyscale: 'str' = 'C major', "
        "generate_audio_codes: 'bool' = True, cfg_scale: 'float' = 2.0, "
        "temperature: 'float' = 0.85, top_p: 'float' = 0.9, top_k: 'int' = 0, "
        "min_p: 'float' = 0.0) -> 'Any'"
    )
    assert "sample_rate" not in signature.parameters


def test_encode_ace_step_15_audio_wraps_tokenize_and_encode_from_tokens_scheduled() -> None:
    expected_tokens = object()
    expected_conditioning = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[tuple[str, dict[str, Any]]] = []
            self.encode_calls: list[Any] = []

        def tokenize(self, tags: str, **kwargs: Any) -> Any:
            self.tokenize_calls.append((tags, kwargs))
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: Any) -> Any:
            self.encode_calls.append(tokens)
            return expected_conditioning

    clip = FakeClip()
    result = encode_ace_step_15_audio(
        clip=clip,
        tags="electronic dance anthem",
        lyrics="we rise at dawn",
        seed=42,
        bpm=128,
        duration=91.5,
        timesignature="6",
        language="en",
        keyscale="A minor",
        generate_audio_codes=False,
        cfg_scale=3.5,
        temperature=0.7,
        top_p=0.8,
        top_k=10,
        min_p=0.05,
    )

    assert result is expected_conditioning
    assert clip.tokenize_calls == [
        (
            "electronic dance anthem",
            {
                "lyrics": "we rise at dawn",
                "bpm": 128,
                "duration": 91.5,
                "timesignature": 6,
                "language": "en",
                "keyscale": "A minor",
                "seed": 42,
                "generate_audio_codes": False,
                "cfg_scale": 3.5,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 10,
                "min_p": 0.05,
            },
        )
    ]
    assert clip.encode_calls == [expected_tokens]


def test_empty_ace_step_15_latent_audio_signature_matches_contract() -> None:
    signature = inspect.signature(empty_ace_step_15_latent_audio)
    assert str(signature) == "(seconds: 'float', batch_size: 'int' = 1) -> 'dict[str, Any]'"


def test_empty_ace_step_15_latent_audio_wraps_logic_and_returns_audio_latent_dict(
    monkeypatch: Any,
) -> None:
    expected_tensor = object()

    class FakeTorch:
        def __init__(self) -> None:
            self.zeros_calls: list[tuple[list[int], str]] = []

        def zeros(self, shape: list[int], *, device: str) -> Any:
            self.zeros_calls.append((shape, device))
            return expected_tensor

    class FakeModelManagement:
        @staticmethod
        def intermediate_device() -> str:
            return "fake-device"

    fake_torch = FakeTorch()

    monkeypatch.setattr(
        audio_module,
        "_get_ace_step_15_latent_audio_dependencies",
        lambda: (fake_torch, FakeModelManagement),
    )

    result = empty_ace_step_15_latent_audio(seconds=2.5, batch_size=3)

    assert result == {"samples": expected_tensor, "type": "audio"}
    assert fake_torch.zeros_calls == [([3, 64, 62], "fake-device")]


def test_ltxv_concat_av_latent_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_concat_av_latent)
    assert (
        str(signature)
        == "(video_latent: 'dict[str, Any]', audio_latent: 'dict[str, Any]') -> 'dict[str, Any]'"
    )


def test_ltxv_concat_av_latent_returns_nested_tensor_samples(monkeypatch: Any) -> None:
    video_samples = object()
    audio_samples = object()
    video_latent: dict[str, Any] = {"samples": video_samples, "type": "video"}
    audio_latent: dict[str, Any] = {"samples": audio_samples, "type": "audio"}

    nested_tensors_created: list[Any] = []

    class FakeNestedTensor:
        def __init__(self, tensors: Any) -> None:
            self.tensors = tensors
            nested_tensors_created.append(self)

    class FakeNestedTensorModule:
        NestedTensor = FakeNestedTensor

    class FakeTorch:
        pass

    monkeypatch.setattr(
        audio_module,
        "_get_concat_av_latent_dependencies",
        lambda: (FakeTorch, FakeNestedTensorModule),
    )

    result = ltxv_concat_av_latent(video_latent, audio_latent)

    assert isinstance(result["samples"], FakeNestedTensor)
    assert result["samples"].tensors == (video_samples, audio_samples)
    assert "noise_mask" not in result


def test_ltxv_concat_av_latent_merges_video_and_audio_keys(monkeypatch: Any) -> None:
    video_latent: dict[str, Any] = {"samples": object(), "type": "video", "extra_v": 1}
    audio_latent: dict[str, Any] = {"samples": object(), "type": "audio", "extra_a": 2}

    class FakeNestedTensor:
        def __init__(self, tensors: Any) -> None:
            self.tensors = tensors

    class FakeNestedTensorModule:
        NestedTensor = FakeNestedTensor

    class FakeTorch:
        pass

    monkeypatch.setattr(
        audio_module,
        "_get_concat_av_latent_dependencies",
        lambda: (FakeTorch, FakeNestedTensorModule),
    )

    result = ltxv_concat_av_latent(video_latent, audio_latent)

    assert result["extra_v"] == 1
    assert result["extra_a"] == 2
    assert result["type"] == "audio"  # audio_latent wins (updated last)


def test_ltxv_concat_av_latent_noise_mask_video_only(monkeypatch: Any) -> None:
    video_samples = object()
    audio_samples = object()
    video_mask = object()

    ones_calls: list[Any] = []

    class FakeTorch:
        @staticmethod
        def ones_like(t: Any) -> object:
            ones_calls.append(t)
            return object()

    class FakeNestedTensor:
        def __init__(self, tensors: Any) -> None:
            self.tensors = tensors

    class FakeNestedTensorModule:
        NestedTensor = FakeNestedTensor

    monkeypatch.setattr(
        audio_module,
        "_get_concat_av_latent_dependencies",
        lambda: (FakeTorch, FakeNestedTensorModule),
    )

    video_latent: dict[str, Any] = {"samples": video_samples, "noise_mask": video_mask}
    audio_latent: dict[str, Any] = {"samples": audio_samples}

    result = ltxv_concat_av_latent(video_latent, audio_latent)

    assert isinstance(result["noise_mask"], FakeNestedTensor)
    # video mask passed as-is, audio mask created via ones_like(audio_samples)
    assert result["noise_mask"].tensors[0] is video_mask
    assert ones_calls == [audio_samples]


def test_ltxv_concat_av_latent_noise_mask_audio_only(monkeypatch: Any) -> None:
    video_samples = object()
    audio_samples = object()
    audio_mask = object()

    ones_calls: list[Any] = []

    class FakeTorch:
        @staticmethod
        def ones_like(t: Any) -> object:
            ones_calls.append(t)
            return object()

    class FakeNestedTensor:
        def __init__(self, tensors: Any) -> None:
            self.tensors = tensors

    class FakeNestedTensorModule:
        NestedTensor = FakeNestedTensor

    monkeypatch.setattr(
        audio_module,
        "_get_concat_av_latent_dependencies",
        lambda: (FakeTorch, FakeNestedTensorModule),
    )

    video_latent: dict[str, Any] = {"samples": video_samples}
    audio_latent: dict[str, Any] = {"samples": audio_samples, "noise_mask": audio_mask}

    result = ltxv_concat_av_latent(video_latent, audio_latent)

    assert isinstance(result["noise_mask"], FakeNestedTensor)
    # video mask created via ones_like(video_samples), audio mask passed as-is
    assert ones_calls == [video_samples]
    assert result["noise_mask"].tensors[1] is audio_mask


def test_ltxv_concat_av_latent_noise_mask_both(monkeypatch: Any) -> None:
    video_samples = object()
    audio_samples = object()
    video_mask = object()
    audio_mask = object()

    class FakeTorch:
        @staticmethod
        def ones_like(t: Any) -> object:
            raise AssertionError("ones_like should not be called when both masks are present")

    class FakeNestedTensor:
        def __init__(self, tensors: Any) -> None:
            self.tensors = tensors

    class FakeNestedTensorModule:
        NestedTensor = FakeNestedTensor

    monkeypatch.setattr(
        audio_module,
        "_get_concat_av_latent_dependencies",
        lambda: (FakeTorch, FakeNestedTensorModule),
    )

    video_latent: dict[str, Any] = {"samples": video_samples, "noise_mask": video_mask}
    audio_latent: dict[str, Any] = {"samples": audio_samples, "noise_mask": audio_mask}

    result = ltxv_concat_av_latent(video_latent, audio_latent)

    assert isinstance(result["noise_mask"], FakeNestedTensor)
    assert result["noise_mask"].tensors == (video_mask, audio_mask)


def test_ltxv_concat_av_latent_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import ltxv_concat_av_latent; "
        "assert ltxv_concat_av_latent.__name__ == 'ltxv_concat_av_latent'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_ltxv_audio_vae_encode_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import ltxv_audio_vae_encode; "
        "assert ltxv_audio_vae_encode.__name__ == 'ltxv_audio_vae_encode'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_ltxv_audio_vae_decode_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import ltxv_audio_vae_decode; "
        "assert ltxv_audio_vae_decode.__name__ == 'ltxv_audio_vae_decode'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_ltxv_empty_latent_audio_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import ltxv_empty_latent_audio; "
        "assert ltxv_empty_latent_audio.__name__ == 'ltxv_empty_latent_audio'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_encode_ace_step_15_audio_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import encode_ace_step_15_audio; "
        "assert encode_ace_step_15_audio.__name__ == 'encode_ace_step_15_audio'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_empty_ace_step_15_latent_audio_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import empty_ace_step_15_latent_audio; "
        "assert empty_ace_step_15_latent_audio.__name__ == 'empty_ace_step_15_latent_audio'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_ltxv_separate_av_latent_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_separate_av_latent)
    assert (
        str(signature)
        == "(av_latent: 'dict[str, Any]') -> 'tuple[dict[str, Any], dict[str, Any]]'"
    )


def test_ltxv_separate_av_latent_unbinds_samples_into_video_and_audio() -> None:
    video_samples = object()
    audio_samples = object()

    class FakeNestedTensor:
        def __init__(self, tensors: tuple[Any, Any]) -> None:
            self.tensors = tensors

        def unbind(self) -> tuple[Any, Any]:
            return self.tensors

    av_latent: dict[str, Any] = {
        "samples": FakeNestedTensor((video_samples, audio_samples)),
    }

    video_latent, audio_latent = ltxv_separate_av_latent(av_latent)

    assert video_latent["samples"] is video_samples
    assert audio_latent["samples"] is audio_samples
    assert "noise_mask" not in video_latent
    assert "noise_mask" not in audio_latent


def test_ltxv_separate_av_latent_unbinds_noise_mask_when_present() -> None:
    video_samples = object()
    audio_samples = object()
    video_mask = object()
    audio_mask = object()

    class FakeNestedTensor:
        def __init__(self, tensors: tuple[Any, Any]) -> None:
            self.tensors = tensors

        def unbind(self) -> tuple[Any, Any]:
            return self.tensors

    av_latent: dict[str, Any] = {
        "samples": FakeNestedTensor((video_samples, audio_samples)),
        "noise_mask": FakeNestedTensor((video_mask, audio_mask)),
    }

    video_latent, audio_latent = ltxv_separate_av_latent(av_latent)

    assert video_latent["samples"] is video_samples
    assert audio_latent["samples"] is audio_samples
    assert video_latent["noise_mask"] is video_mask
    assert audio_latent["noise_mask"] is audio_mask


def test_ltxv_separate_av_latent_is_importable_from_comfy_diffusion_audio() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import ltxv_separate_av_latent; "
        "assert ltxv_separate_av_latent.__name__ == 'ltxv_separate_av_latent'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_import_comfy_diffusion_audio_has_no_torch_or_comfy_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.audio import (\n"
        "  empty_ace_step_15_latent_audio,\n"
        "  encode_ace_step_15_audio,\n"
        "  ltxv_audio_vae_decode,\n"
        "  ltxv_audio_vae_encode,\n"
        "  ltxv_concat_av_latent,\n"
        "  ltxv_empty_latent_audio,\n"
        "  ltxv_separate_av_latent,\n"
        ")\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'empty_ace_step_15_func_name': empty_ace_step_15_latent_audio.__name__,\n"
        "  'ace_step_15_func_name': encode_ace_step_15_audio.__name__,\n"
        "  'func_name': ltxv_audio_vae_encode.__name__,\n"
        "  'decode_func_name': ltxv_audio_vae_decode.__name__,\n"
        "  'concat_av_func_name': ltxv_concat_av_latent.__name__,\n"
        "  'empty_latent_func_name': ltxv_empty_latent_audio.__name__,\n"
        "  'separate_av_func_name': ltxv_separate_av_latent.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'comfy_loaded': 'comfy' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["empty_ace_step_15_func_name"] == "empty_ace_step_15_latent_audio"
    assert payload["ace_step_15_func_name"] == "encode_ace_step_15_audio"
    assert payload["func_name"] == "ltxv_audio_vae_encode"
    assert payload["decode_func_name"] == "ltxv_audio_vae_decode"
    assert payload["concat_av_func_name"] == "ltxv_concat_av_latent"
    assert payload["empty_latent_func_name"] == "ltxv_empty_latent_audio"
    assert payload["separate_av_func_name"] == "ltxv_separate_av_latent"
    assert payload["torch_loaded"] is False
    assert payload["comfy_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    heavy = [module for module in payload["new_modules"] if module.startswith(("torch", "comfy.")) or module == "comfy"]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"


# ---------------------------------------------------------------------------
# load_audio tests
# ---------------------------------------------------------------------------


def test_load_audio_signature_matches_contract() -> None:
    signature = inspect.signature(load_audio)
    params = signature.parameters
    assert "path" in params
    assert "start_time" in params
    assert params["start_time"].default == 0.0
    assert "duration" in params
    assert params["duration"].default is None


def _make_fake_torchaudio(sample_rate: int, num_channels: int, num_samples: int) -> Any:
    """Return a fake torchaudio module whose load() returns a fixed tensor."""
    import sys

    # Import torch inside helper so it is lazily available
    import torch

    fake_waveform = torch.zeros(num_channels, num_samples)

    class FakeTorchaudio:
        @staticmethod
        def load(path: Any, **kwargs: Any) -> tuple[Any, int]:
            return fake_waveform, sample_rate

    return FakeTorchaudio()


def test_load_audio_returns_waveform_dict(monkeypatch: Any) -> None:
    """load_audio returns dict with 'waveform' [1, C, N] tensor and 'sample_rate' int."""
    import sys

    sample_rate = 8000
    num_channels = 1
    num_samples = 4000  # 0.5 seconds

    fake_ta = _make_fake_torchaudio(sample_rate, num_channels, num_samples)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    result = load_audio("/fake/audio.wav")

    assert isinstance(result, dict)
    assert "waveform" in result
    assert "sample_rate" in result
    assert result["sample_rate"] == sample_rate
    waveform = result["waveform"]
    assert len(waveform.shape) == 3, f"Expected shape [1, C, N], got {waveform.shape}"
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == num_channels
    assert waveform.shape[2] == num_samples


def test_load_audio_start_time_trims_beginning(monkeypatch: Any) -> None:
    """start_time offsets the audio by the given number of seconds."""
    import sys

    sample_rate = 8000
    num_channels = 1
    num_samples = 8000  # 1 second

    fake_ta = _make_fake_torchaudio(sample_rate, num_channels, num_samples)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    result = load_audio("/fake/audio.wav", start_time=0.5)
    waveform = result["waveform"]
    # After 0.5s offset from 1s audio, 4000 samples remain
    assert waveform.shape[2] == 4000


def test_load_audio_duration_limits_output(monkeypatch: Any) -> None:
    """duration limits the number of audio frames loaded."""
    import math
    import sys

    sample_rate = 8000
    num_channels = 1
    num_samples = 8000  # 1 second

    fake_ta = _make_fake_torchaudio(sample_rate, num_channels, num_samples)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    result = load_audio("/fake/audio.wav", duration=0.25)
    waveform = result["waveform"]
    expected = math.ceil(0.25 * sample_rate)
    assert waveform.shape[2] == expected


def test_load_audio_start_and_duration(monkeypatch: Any) -> None:
    """start_time + duration correctly slices the waveform."""
    import math
    import sys

    sample_rate = 16000
    num_channels = 2
    num_samples = 32000  # 2 seconds

    fake_ta = _make_fake_torchaudio(sample_rate, num_channels, num_samples)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_ta)

    result = load_audio("/fake/audio.wav", start_time=0.5, duration=1.0)
    waveform = result["waveform"]
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == num_channels
    assert waveform.shape[2] == math.ceil(1.0 * sample_rate)


def test_load_audio_is_in_all() -> None:
    assert "load_audio" in audio_module.__all__


def test_load_audio_is_importable() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import load_audio; "
        "assert load_audio.__name__ == 'load_audio'; "
        "print('ok')"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# audio_crop tests
# ---------------------------------------------------------------------------


def test_audio_crop_signature_matches_contract() -> None:
    signature = inspect.signature(audio_crop)
    assert (
        str(signature)
        == "(audio: 'dict[str, Any]', start_time: 'float', end_time: 'float') -> 'dict[str, Any]'"
    )


def test_audio_crop_slices_waveform_by_time() -> None:
    import torch

    sample_rate = 8000
    num_samples = 8000  # 1 second
    waveform = torch.arange(num_samples, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_crop(audio, start_time=0.25, end_time=0.75)

    assert result["sample_rate"] == sample_rate
    expected_start = int(round(0.25 * sample_rate))
    expected_end = int(round(0.75 * sample_rate))
    assert result["waveform"].shape[-1] == expected_end - expected_start
    torch.testing.assert_close(result["waveform"], waveform[..., expected_start:expected_end])


def test_audio_crop_clamps_to_audio_bounds() -> None:
    import torch

    sample_rate = 4000
    num_samples = 4000  # 1 second
    waveform = torch.zeros(1, 1, num_samples)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_crop(audio, start_time=0.0, end_time=10.0)

    assert result["waveform"].shape[-1] == num_samples


def test_audio_crop_raises_for_invalid_range() -> None:
    import pytest
    import torch

    sample_rate = 8000
    waveform = torch.zeros(1, 1, 8000)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    with pytest.raises(ValueError, match="start_time"):
        audio_crop(audio, start_time=0.5, end_time=0.5)


def test_audio_crop_is_in_all() -> None:
    assert "audio_crop" in audio_module.__all__


# ---------------------------------------------------------------------------
# audio_separation tests
# ---------------------------------------------------------------------------


def test_audio_separation_signature_matches_contract() -> None:
    signature = inspect.signature(audio_separation)
    params = signature.parameters
    assert list(params.keys()) == ["audio", "mode", "fft_n", "win_length"]
    assert params["mode"].default == "harmonic"
    assert params["fft_n"].default == 2048
    assert params["win_length"].default is None


def test_audio_separation_returns_dict_with_waveform_and_sample_rate() -> None:
    import torch

    sample_rate = 16000
    num_samples = 16000  # 1 second
    waveform = torch.randn(1, 1, num_samples)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    result = audio_separation(audio, mode="harmonic", fft_n=512, win_length=512)

    assert "waveform" in result
    assert "sample_rate" in result
    assert result["sample_rate"] == sample_rate
    assert result["waveform"].shape == waveform.shape


def test_audio_separation_harmonic_and_percussive_differ() -> None:
    import torch

    sample_rate = 16000
    num_samples = 16000
    torch.manual_seed(0)
    waveform = torch.randn(1, 1, num_samples)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": sample_rate}

    harmonic = audio_separation(audio, mode="harmonic", fft_n=512, win_length=512)
    percussive = audio_separation(audio, mode="percussive", fft_n=512, win_length=512)

    assert not torch.allclose(harmonic["waveform"], percussive["waveform"])


def test_audio_separation_raises_on_invalid_mode() -> None:
    import pytest
    import torch

    waveform = torch.zeros(1, 1, 4096)
    audio: dict[str, Any] = {"waveform": waveform, "sample_rate": 8000}

    with pytest.raises(ValueError, match="mode"):
        audio_separation(audio, mode="invalid")


def test_audio_separation_is_in_all() -> None:
    assert "audio_separation" in audio_module.__all__


# ---------------------------------------------------------------------------
# trim_audio_duration tests
# ---------------------------------------------------------------------------


def test_trim_audio_duration_signature_matches_contract() -> None:
    signature = inspect.signature(trim_audio_duration)
    assert (
        str(signature)
        == "(audio: 'dict[str, Any]', start: 'float', duration: 'float') -> 'dict[str, Any]'"
    )


def test_trim_audio_duration_wraps_trim_audio_duration_node(monkeypatch: Any) -> None:
    expected_waveform = object()
    expected_output: dict[str, Any] = {"waveform": expected_waveform, "sample_rate": 44100}
    call_log: list[dict[str, Any]] = []

    class FakeTrimAudioDurationNode:
        @classmethod
        def execute(cls, *, audio: Any, start_index: float, duration: float) -> tuple[dict[str, Any], ...]:
            call_log.append({"audio": audio, "start_index": start_index, "duration": duration})
            return (expected_output,)

    monkeypatch.setattr(
        audio_module,
        "_get_trim_audio_duration_node",
        lambda: FakeTrimAudioDurationNode,
    )

    input_audio: dict[str, Any] = {"waveform": object(), "sample_rate": 44100}
    result = trim_audio_duration(input_audio, start=2.5, duration=10.0)

    assert result is expected_output
    assert len(call_log) == 1
    assert call_log[0]["audio"] is input_audio
    assert call_log[0]["start_index"] == 2.5
    assert call_log[0]["duration"] == 10.0


def test_trim_audio_duration_is_in_all() -> None:
    assert "trim_audio_duration" in audio_module.__all__


def test_trim_audio_duration_is_importable() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import trim_audio_duration; "
        "assert trim_audio_duration.__name__ == 'trim_audio_duration'; "
        "print('ok')"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_audio_preprocessing_has_no_top_level_comfy_or_torch_imports() -> None:
    result = _run_python(
        "import sys\n"
        "import comfy_diffusion\n"
        "from comfy_diffusion.audio import audio_crop, audio_separation, trim_audio_duration\n"
        "assert 'torch' not in sys.modules, 'torch was imported at module level'\n"
        "assert 'comfy' not in sys.modules, 'comfy was imported at module level'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# audio_encoder_encode tests
# ---------------------------------------------------------------------------


def test_audio_encoder_encode_is_in_all() -> None:
    assert "audio_encoder_encode" in audio_module.__all__


def test_audio_encoder_encode_signature_matches_contract() -> None:
    signature = inspect.signature(audio_encoder_encode)
    assert str(signature) == "(audio_encoder: 'Any', audio: 'dict') -> 'Any'"


def test_audio_encoder_encode_calls_encode_audio_with_waveform_and_sample_rate() -> None:
    expected_waveform = object()
    expected_sample_rate = 44100
    expected_output = object()
    call_log: list[dict[str, Any]] = []

    class FakeAudioEncoder:
        def encode_audio(self, waveform: Any, sample_rate: Any) -> Any:
            call_log.append({"waveform": waveform, "sample_rate": sample_rate})
            return expected_output

    encoder = FakeAudioEncoder()
    audio: dict[str, Any] = {"waveform": expected_waveform, "sample_rate": expected_sample_rate}

    result = audio_encoder_encode(encoder, audio)

    assert result is expected_output
    assert len(call_log) == 1
    assert call_log[0]["waveform"] is expected_waveform
    assert call_log[0]["sample_rate"] == expected_sample_rate


def test_audio_encoder_encode_is_importable() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import audio_encoder_encode; "
        "assert audio_encoder_encode.__name__ == 'audio_encoder_encode'; "
        "print('ok')"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_audio_encoder_encode_has_no_top_level_comfy_or_torch_imports() -> None:
    result = _run_python(
        "import sys\n"
        "import comfy_diffusion\n"
        "from comfy_diffusion.audio import audio_encoder_encode\n"
        "assert 'torch' not in sys.modules, 'torch was imported at module level'\n"
        "assert 'comfy' not in sys.modules, 'comfy was imported at module level'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# vae_decode_audio
# ---------------------------------------------------------------------------


def test_vae_decode_audio_is_in_all() -> None:
    assert "vae_decode_audio" in audio_module.__all__


def test_vae_decode_audio_calls_vae_decode_and_movedim() -> None:
    """AC-002: implementation calls vae.decode(latent["samples"]).movedim(-1, 1)."""
    movedim_calls: list[tuple[Any, Any]] = []

    class FakeWaveform:
        def movedim(self, src: Any, dst: Any) -> "FakeWaveform":
            movedim_calls.append((src, dst))
            return self

    fake_waveform = FakeWaveform()
    decode_calls: list[Any] = []

    class FakeVae:
        def decode(self, samples: Any) -> FakeWaveform:
            decode_calls.append(samples)
            return fake_waveform

    expected_samples = object()
    latent = {"samples": expected_samples}

    result = vae_decode_audio(FakeVae(), latent)

    assert decode_calls == [expected_samples]
    assert movedim_calls == [(-1, 1)]
    assert result is fake_waveform


def test_vae_decode_audio_is_importable() -> None:
    result = _run_python(
        "from comfy_diffusion.audio import vae_decode_audio; "
        "assert vae_decode_audio.__name__ == 'vae_decode_audio'; "
        "print('ok')"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_vae_decode_audio_has_no_top_level_comfy_imports() -> None:
    """AC-004: vae_decode_audio must not trigger top-level comfy.* imports."""
    result = _run_python(
        "import sys\n"
        "import comfy_diffusion\n"
        "from comfy_diffusion.audio import vae_decode_audio\n"
        "assert 'comfy' not in sys.modules, 'comfy was imported at module level'\n"
        "print('ok')\n"
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"
