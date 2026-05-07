"""Tests for prompt conditioning helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import comfy_diffusion.conditioning as conditioning
from comfy_diffusion.conditioning import (
    conditioning_combine,
    conditioning_set_mask,
    conditioning_set_timestep_range,
    encode_clip_vision,
    encode_prompt,
    encode_prompt_flux,
    flux_guidance,
    ltxv_conditioning,
    ltxv_crop_guides,
    ltxv_img_to_video,
    reference_latent,
    wan_first_last_frame_to_video,
    wan_image_to_video,
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


def test_encode_prompt_imports_without_models_on_cpu() -> None:
    result = _run_python(
        "from comfy_diffusion.conditioning import encode_prompt; "
        "assert encode_prompt.__name__ == 'encode_prompt'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_encode_prompt_returns_raw_clip_conditioning_output() -> None:
    sentinel_conditioning = object()
    expected_tokens = {"tokenized": "a portrait of a woman, studio lighting"}

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[object] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            self.encode_calls.append(tokens)
            return sentinel_conditioning

    clip = FakeClip()

    result = encode_prompt(clip, "a portrait of a woman, studio lighting")

    assert result is not None
    assert result is sentinel_conditioning
    assert clip.tokenize_calls == ["a portrait of a woman, studio lighting"]
    assert clip.encode_calls == [expected_tokens]


def test_encode_prompt_supports_negative_prompt_text() -> None:
    sentinel_conditioning = object()
    expected_tokens = {"tokenized": "ugly, blurry"}

    class FakeClip:
        def tokenize(self, text: str) -> object:
            assert text == "ugly, blurry"
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            assert tokens is expected_tokens
            return sentinel_conditioning

    result = encode_prompt(FakeClip(), "ugly, blurry")

    assert result is not None
    assert result is sentinel_conditioning


def test_encode_prompt_accepts_weighted_prompt_syntax_without_changes() -> None:
    weighted_prompt = "a portrait of a woman, (studio lighting:1.3)"
    sentinel_conditioning = object()
    expected_tokens = {"tokenized": weighted_prompt}

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[object] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            return expected_tokens

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            self.encode_calls.append(tokens)
            return sentinel_conditioning

    clip = FakeClip()
    result = encode_prompt(clip, weighted_prompt)

    assert result is not None
    assert result is sentinel_conditioning
    assert clip.tokenize_calls == [weighted_prompt]
    assert clip.encode_calls == [expected_tokens]


def test_encode_prompt_does_not_raise_for_valid_weighted_prompt_syntax() -> None:
    weighted_prompt = "a portrait of a woman, (studio lighting:1.3)"

    class FakeClip:
        def tokenize(self, text: str) -> object:
            assert text == weighted_prompt
            return ("tokens", text)

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            return {"conditioning": tokens}

    result = encode_prompt(FakeClip(), weighted_prompt)

    assert result is not None
    assert result == {"conditioning": ("tokens", weighted_prompt)}


def test_encode_prompt_accepts_empty_string_without_crashing() -> None:
    sentinel_conditioning = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            if text == "":
                raise AssertionError("empty prompt should be normalized before tokenization")
            return ("tokens", text)

        def encode_from_tokens_scheduled(self, tokens: object) -> object:
            assert tokens == ("tokens", " ")
            return sentinel_conditioning

    clip = FakeClip()
    result = encode_prompt(clip, "")

    assert result is not None
    assert result is sentinel_conditioning
    assert clip.tokenize_calls == [" "]


def test_encode_prompt_flux_imports_without_models_on_cpu() -> None:
    result = _run_python(
        "from comfy_diffusion.conditioning import encode_prompt_flux; "
        "assert encode_prompt_flux.__name__ == 'encode_prompt_flux'; "
        "print('ok')"
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_encode_prompt_flux_encodes_clip_l_and_t5xxl_with_guidance() -> None:
    expected_output = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[tuple[object, object]] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            if text == "clip-l detail":
                return {"clip_l": "clip-l detail"}
            if text == "t5 detail":
                return {"t5xxl": "t5 detail"}
            raise AssertionError(f"unexpected text: {text!r}")

        def encode_from_tokens_scheduled(
            self,
            tokens: object,
            add_dict: object = None,
        ) -> object:
            self.encode_calls.append((tokens, add_dict))
            return expected_output

    clip = FakeClip()

    result = encode_prompt_flux(
        clip=clip,
        text="t5 detail",
        clip_l_text="clip-l detail",
        guidance=7.0,
    )

    assert result is expected_output
    assert clip.tokenize_calls == ["clip-l detail", "t5 detail"]
    assert clip.encode_calls == [
        ({"clip_l": "clip-l detail", "t5xxl": "t5 detail"}, {"guidance": 7.0})
    ]


def test_encode_prompt_flux_uses_default_guidance_and_normalizes_empty_text() -> None:
    expected_output = object()

    class FakeClip:
        def __init__(self) -> None:
            self.tokenize_calls: list[str] = []
            self.encode_calls: list[tuple[object, object]] = []

        def tokenize(self, text: str) -> object:
            self.tokenize_calls.append(text)
            if text == " ":
                return {"t5xxl": " "}
            if text == "clip-l":
                return {"clip_l": "clip-l"}
            raise AssertionError(f"unexpected text: {text!r}")

        def encode_from_tokens_scheduled(
            self,
            tokens: object,
            add_dict: object = None,
        ) -> object:
            self.encode_calls.append((tokens, add_dict))
            return expected_output

    clip = FakeClip()

    result = encode_prompt_flux(clip=clip, text="", clip_l_text="clip-l")

    assert result is expected_output
    assert clip.tokenize_calls == ["clip-l", " "]
    assert clip.encode_calls == [
        ({"clip_l": "clip-l", "t5xxl": " "}, {"guidance": 3.5})
    ]


def test_conditioning_public_api_exports_expected_entrypoints() -> None:
    signature = inspect.signature(encode_prompt)
    flux_signature = inspect.signature(encode_prompt_flux)
    flux_guidance_signature = inspect.signature(flux_guidance)

    assert "is_negative" not in signature.parameters
    assert not hasattr(conditioning, "encode_negative_prompt")
    assert flux_signature.parameters["guidance"].default == 3.5
    assert flux_guidance_signature.parameters["guidance"].default == 3.5
    assert conditioning_combine.__name__ == "conditioning_combine"
    assert conditioning_set_mask.__name__ == "conditioning_set_mask"
    assert conditioning_set_timestep_range.__name__ == "conditioning_set_timestep_range"
    assert flux_guidance.__name__ == "flux_guidance"
    assert encode_clip_vision.__name__ == "encode_clip_vision"
    assert wan_image_to_video.__name__ == "wan_image_to_video"
    assert wan_first_last_frame_to_video.__name__ == "wan_first_last_frame_to_video"
    assert ltxv_img_to_video.__name__ == "ltxv_img_to_video"
    assert ltxv_conditioning.__name__ == "ltxv_conditioning"
    assert conditioning.__all__ == [
        "encode_prompt",
        "encode_prompt_flux",
        "encode_clip_vision",
        "wan_image_to_video",
        "wan_first_last_frame_to_video",
        "wan_vace_to_video",
        "wan_fun_control_to_video",
        "wan22_fun_control_to_video",
        "wan_fun_inpaint_to_video",
        "wan_camera_embedding",
        "wan_camera_image_to_video",
        "wan_phantom_subject_to_video",
        "wan_track_to_video",
        "wan_sound_image_to_video",
        "wan_sound_image_to_video_extend",
        "wan_humo_image_to_video",
        "wan_animate_to_video",
        "wan_infinite_talk_to_video",
        "wan_scail_to_video",
        "wan22_image_to_video_latent",
        "ltxv_img_to_video",
        "ltxv_conditioning",
        "ltxv_crop_guides",
        "ltx_add_video_ic_lora_guide",
        "conditioning_zero_out",
        "conditioning_combine",
        "conditioning_set_mask",
        "conditioning_set_timestep_range",
        "flux_guidance",
        "reference_latent",
        "encode_qwen_image_edit_plus",
        "apply_flux_kontext_multi_reference",
    ]


def test_conditioning_combine_merges_two_conditioning_lists() -> None:
    cond_a = [["a-token-1", {"meta": "a1"}], ["a-token-2", {"meta": "a2"}]]
    cond_b = [["b-token-1", {"meta": "b1"}]]

    merged = conditioning_combine(cond_a, cond_b)

    assert merged == [
        ["a-token-1", {"meta": "a1"}],
        ["a-token-2", {"meta": "a2"}],
        ["b-token-1", {"meta": "b1"}],
    ]


def test_conditioning_combine_supports_more_than_two_by_chaining() -> None:
    cond_a = [["a-token", {"w": 1.0}]]
    cond_b = [["b-token", {"w": 1.0}]]
    cond_c = [["c-token", {"w": 1.0}]]

    merged = conditioning_combine(conditioning_combine(cond_a, cond_b), cond_c)

    assert merged == [
        ["a-token", {"w": 1.0}],
        ["b-token", {"w": 1.0}],
        ["c-token", {"w": 1.0}],
    ]


def test_conditioning_combine_supports_list_input() -> None:
    cond_a = [["a-token", {"source": "base"}]]
    cond_b = [["b-token", {"source": "style"}]]
    cond_c = [["c-token", {"source": "detail"}]]

    merged = conditioning_combine([cond_a, cond_b, cond_c])

    assert merged == [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
        ["c-token", {"source": "detail"}],
    ]


def test_conditioning_set_mask_applies_default_strength_and_default_cond_area() -> None:
    conditioning_input = [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
    ]

    class FakeMask:
        def __init__(self) -> None:
            self.shape = (64, 64)
            self.unsqueeze_calls: list[int] = []
            self.batch_mask = object()

        def unsqueeze(self, dim: int) -> object:
            self.unsqueeze_calls.append(dim)
            assert dim == 0
            return self.batch_mask

    mask = FakeMask()

    output = conditioning_set_mask(conditioning_input, mask)

    assert output is not conditioning_input
    assert mask.unsqueeze_calls == [0]
    assert output[0][1]["source"] == "base"
    assert output[1][1]["source"] == "style"
    assert output[0][1]["mask"] is mask.batch_mask
    assert output[1][1]["mask"] is mask.batch_mask
    assert output[0][1]["mask_strength"] == 1.0
    assert output[1][1]["mask_strength"] == 1.0
    assert output[0][1]["set_area_to_bounds"] is False
    assert output[1][1]["set_area_to_bounds"] is False
    assert "mask" not in conditioning_input[0][1]
    assert "mask_strength" not in conditioning_input[0][1]


def test_conditioning_set_mask_supports_mask_bounds_and_custom_strength() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    class FakeMask:
        def __init__(self) -> None:
            self.shape = (1, 64, 64)
            self.unsqueeze_calls: list[int] = []

        def unsqueeze(self, dim: int) -> Any:
            self.unsqueeze_calls.append(dim)
            raise AssertionError("mask already has batch dimension")

    mask = FakeMask()

    output = conditioning_set_mask(
        conditioning_input,
        mask=mask,
        strength=0.25,
        set_cond_area="mask bounds",
    )

    assert mask.unsqueeze_calls == []
    assert output[0][1]["mask"] is mask
    assert output[0][1]["mask_strength"] == 0.25
    assert output[0][1]["set_area_to_bounds"] is True


def test_conditioning_set_timestep_range_sets_start_and_end_percent_metadata() -> None:
    conditioning_input = [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
    ]

    output = conditioning_set_timestep_range(conditioning_input, start=0.2, end=0.8)

    assert output is not conditioning_input
    assert output == [
        ["a-token", {"source": "base", "start_percent": 0.2, "end_percent": 0.8}],
        ["b-token", {"source": "style", "start_percent": 0.2, "end_percent": 0.8}],
    ]
    assert output[0][1] is not conditioning_input[0][1]
    assert output[1][1] is not conditioning_input[1][1]
    assert "start_percent" not in conditioning_input[0][1]
    assert "end_percent" not in conditioning_input[0][1]


def test_conditioning_set_timestep_range_rejects_non_float_values() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    with pytest.raises(TypeError, match="start must be a float"):
        conditioning_set_timestep_range(conditioning_input, start=0, end=1.0)

    with pytest.raises(TypeError, match="end must be a float"):
        conditioning_set_timestep_range(conditioning_input, start=0.0, end=1)


def test_conditioning_set_timestep_range_rejects_values_outside_percentage_bounds() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    with pytest.raises(ValueError, match="start must be between 0.0 and 1.0"):
        conditioning_set_timestep_range(conditioning_input, start=-0.01, end=1.0)

    with pytest.raises(ValueError, match="end must be between 0.0 and 1.0"):
        conditioning_set_timestep_range(conditioning_input, start=0.0, end=1.01)


def test_flux_guidance_applies_guidance_to_each_conditioning_item() -> None:
    conditioning_input = [
        ["a-token", {"source": "base"}],
        ["b-token", {"source": "style"}],
    ]

    output = flux_guidance(conditioning_input, guidance=5.25)

    assert output is not conditioning_input
    assert output == [
        ["a-token", {"source": "base", "guidance": 5.25}],
        ["b-token", {"source": "style", "guidance": 5.25}],
    ]
    assert output[0][1] is not conditioning_input[0][1]
    assert output[1][1] is not conditioning_input[1][1]
    assert "guidance" not in conditioning_input[0][1]
    assert "guidance" not in conditioning_input[1][1]


def test_flux_guidance_uses_default_guidance_value() -> None:
    conditioning_input = [["a-token", {"source": "base"}]]

    output = flux_guidance(conditioning_input)

    assert output == [["a-token", {"source": "base", "guidance": 3.5}]]


def test_ltxv_crop_guides_returns_inputs_unchanged_when_no_keyframe_idxs() -> None:
    """AC03: when num_keyframes == 0 (no keyframe_idxs in conditioning), inputs are returned unchanged."""
    positive = [["pos-token", {"frame_rate": 25.0}]]
    negative = [["neg-token", {"frame_rate": 25.0}]]

    class _FakeLatent(dict):  # type: ignore[misc]
        pass

    latent: dict[str, Any] = _FakeLatent({"samples": object()})

    out_pos, out_neg, out_latent = ltxv_crop_guides(positive, negative, latent)

    assert out_pos is positive
    assert out_neg is negative
    assert out_latent is latent


def test_ltxv_crop_guides_is_import_safe_without_gpu() -> None:
    """AC04: importing ltxv_crop_guides must not load torch or comfy.sd modules."""
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.conditioning import ltxv_crop_guides\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': ltxv_crop_guides.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "ltxv_crop_guides"
    assert payload["torch_loaded"] is False
    heavy = [
        m
        for m in payload["new_modules"]
        if m.startswith(("torch", "folder_paths", "comfy.sd"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"


def test_ltxv_crop_guides_crops_latent_and_clears_conditioning_when_keyframes_present() -> None:
    """AC02: when keyframe_idxs are present, crops latent frames and clears guide keys from conditioning."""

    class _FakeTensor:
        """Minimal tensor stub: supports clone, slicing, shape, and torch.unique."""

        def __init__(self, data: list[Any], shape: tuple[int, ...]) -> None:
            self.data = data
            self.shape = shape

        device = "cpu"

        def clone(self) -> "_FakeTensor":
            return _FakeTensor(list(self.data), self.shape)

        def __getitem__(self, idx: Any) -> "_FakeTensor":
            # Handle [:, :, :-n] slicing on the time dimension (dim 2).
            if isinstance(idx, tuple) and len(idx) == 3:
                _b, _c, t_slice = idx
                if isinstance(t_slice, slice) and t_slice.stop is not None and t_slice.stop < 0:
                    n = -t_slice.stop
                    new_len = self.shape[2] - n
                    new_shape = (self.shape[0], self.shape[1], new_len) + self.shape[3:]
                    return _FakeTensor(self.data, new_shape)
                if isinstance(t_slice, slice):
                    return _FakeTensor(self.data, self.shape)
            # keyframe_idxs[:, 0, :, 0] → return stub for torch.unique
            if isinstance(idx, tuple) and len(idx) == 4:
                return _FakeTensor(self.data, (2,))
            return _FakeTensor(self.data, self.shape)

    class _FakeTorch:
        """Torch stub providing only `unique` and `ones`."""

        float32 = "float32"

        def unique(self, tensor: _FakeTensor) -> _FakeTensor:
            # Simulate 2 unique keyframe start positions.
            return _FakeTensor([], (2,))

        def ones(self, shape: tuple[int, ...], **kwargs: Any) -> _FakeTensor:
            return _FakeTensor([], shape)

    # Build keyframe_idxs tensor: shape (1, 3, 2, 2) — 2 unique start values
    keyframe_idxs = _FakeTensor([], (1, 3, 2, 2))
    # latent with 7 time frames (num_keyframes=2 → 5 remain)
    latent_samples = _FakeTensor([], (1, 128, 7, 4, 4))

    positive_in = [["pos-token", {"keyframe_idxs": keyframe_idxs, "guide_attention_entries": "entries"}]]
    negative_in = [["neg-token", {"keyframe_idxs": keyframe_idxs, "guide_attention_entries": "entries"}]]
    latent_in: dict[str, Any] = {"samples": latent_samples}

    set_values_calls: list[tuple[Any, dict[str, Any]]] = []

    def _conditioning_set_values(cond: Any, values: dict[str, Any]) -> list[Any]:
        set_values_calls.append((cond, dict(values)))
        # Return a new conditioning with the values applied (mimic node_helpers behaviour).
        result = []
        for entry in cond:
            updated = [entry[0], {**entry[1], **values}]
            result.append(updated)
        return result

    import unittest.mock as mock

    fake_torch = _FakeTorch()

    with (
        mock.patch(
            "comfy_diffusion.conditioning._get_ltxv_conditioning_dependencies",
            return_value=(fake_torch, None, None, mock.Mock(conditioning_set_values=_conditioning_set_values)),
        ),
    ):
        out_pos, out_neg, out_latent = ltxv_crop_guides(positive_in, negative_in, latent_in)

    # Latent must be cropped by num_keyframes=2 frames → 5 remaining
    assert out_latent["samples"].shape[2] == 5
    assert "noise_mask" in out_latent
    assert out_latent["noise_mask"].shape[2] == 5

    # guide keys cleared in both conditioning objects
    assert len(set_values_calls) == 2
    for _cond, values in set_values_calls:
        assert values["keyframe_idxs"] is None
        assert values["guide_attention_entries"] is None

    # Original latent not mutated
    assert latent_in["samples"].shape[2] == 7


def test_ltxv_crop_guides_in_all() -> None:
    """AC05: ltxv_crop_guides is listed in conditioning.__all__."""
    assert "ltxv_crop_guides" in conditioning.__all__


def test_import_comfy_diffusion_conditioning_has_no_torch_or_loader_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.conditioning import encode_prompt\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': encode_prompt.__name__,\n"
        "  'torch_loaded': 'torch' in sys.modules,\n"
        "  'folder_paths_loaded': 'folder_paths' in sys.modules,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "encode_prompt"
    assert payload["torch_loaded"] is False
    assert payload["folder_paths_loaded"] is False
    assert payload["comfy_sd_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("torch", "folder_paths", "comfy.sd"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"


# ── reference_latent ─────────────────────────────────────────────────────────


def test_reference_latent_returns_list() -> None:
    """US-002-AC03: output is a list regardless of mock inputs."""
    conditioning_input = [
        ["token-a", {"source": "base"}],
    ]
    # Use a plain object as a stand-in for a torch tensor — node_helpers only
    # stores the reference, so no GPU or torch required.
    fake_samples = object()
    latent: dict[str, Any] = {"samples": fake_samples}

    result = reference_latent(conditioning_input, latent)

    assert isinstance(result, list)


def test_reference_latent_injects_reference_latents_key() -> None:
    """reference_latents list is appended to each conditioning entry."""
    conditioning_input = [
        ["token-a", {}],
        ["token-b", {}],
    ]
    fake_samples = object()
    latent: dict[str, Any] = {"samples": fake_samples}

    result = reference_latent(conditioning_input, latent)

    assert len(result) == 2
    for entry in result:
        assert "reference_latents" in entry[1]
        assert entry[1]["reference_latents"] == [fake_samples]


def test_reference_latent_does_not_mutate_input() -> None:
    """Original conditioning list is not modified."""
    conditioning_input = [["token-a", {}]]
    latent: dict[str, Any] = {"samples": object()}

    reference_latent(conditioning_input, latent)

    assert conditioning_input[0][1] == {}


def test_reference_latent_in_all() -> None:
    """US-002-AC02: reference_latent is listed in conditioning.__all__."""
    import comfy_diffusion.conditioning as _cond_mod

    assert "reference_latent" in _cond_mod.__all__
