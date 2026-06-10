"""Tests for video I/O helpers."""

from __future__ import annotations

import builtins
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import comfy_diffusion
import comfy_diffusion.video as video_module
from comfy_diffusion.video import (
    get_video_components,
    get_video_metadata,
    load_video,
    ltxv_img_to_video_inplace,
    save_video,
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


def test_video_module_exports_expected_entrypoints() -> None:
    assert video_module.__all__ == [
        "load_video",
        "save_video",
        "get_video_metadata",
        "get_video_components",
        "ltxv_img_to_video_inplace",
        "ltxv_chunk_feed_forward",
        "ltx2_nag",
        "ltxv_img_to_video_inplace_kj",
        "ltx2_sampling_preview_override",
        "create_video",
        "void_warped_noise",
        "apply_cfg_norm",
    ]


def test_load_video_signature_matches_contract() -> None:
    signature = inspect.signature(load_video)
    assert str(signature) == "(path: 'str | Path') -> 'Any'"


def test_save_video_signature_matches_contract() -> None:
    signature = inspect.signature(save_video)
    assert str(signature) == "(frames: 'Any', path: 'str | Path', fps: 'float') -> 'None'"


def test_get_video_metadata_signature_matches_contract() -> None:
    signature = inspect.signature(get_video_metadata)
    assert str(signature) == "(video_path: 'str | Path') -> 'dict[str, int | float]'"


def test_get_video_components_signature_matches_contract() -> None:
    signature = inspect.signature(get_video_components)
    assert str(signature) == "(video: 'Any') -> 'tuple[Any, Any]'"


def test_video_helpers_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "load_video")
    assert not hasattr(comfy_diffusion, "save_video")
    assert not hasattr(comfy_diffusion, "get_video_metadata")
    assert not hasattr(comfy_diffusion, "get_video_components")


def test_load_video_reads_frames_with_backend_and_returns_bhwc_tensor_or_pil(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"not-a-real-video")

    expected_frames = [object()]
    expected_output = object()
    calls: dict[str, Any] = {}

    def fake_read_frames(_cv2: Any, path: Path) -> list[Any]:
        calls["path"] = path
        return expected_frames

    def fake_frames_to_output(frames: list[Any]) -> Any:
        calls["frames"] = frames
        return expected_output

    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", object()))
    monkeypatch.setattr(video_module, "_read_frames_cv2", fake_read_frames)
    monkeypatch.setattr(video_module, "_frames_to_output", fake_frames_to_output)

    result = load_video(video_path)

    assert result is expected_output
    assert calls["path"] == video_path
    assert calls["frames"] == expected_frames


def test_save_video_writes_frames_with_cv2_backend(monkeypatch: Any, tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"
    frame_a = SimpleNamespace(shape=(8, 12, 3), name="a")
    frame_b = SimpleNamespace(shape=(8, 12, 3), name="b")
    written: list[Any] = []

    class FakeWriter:
        def isOpened(self) -> bool:
            return True

        def write(self, frame: Any) -> None:
            written.append(frame)

        def release(self) -> None:
            written.append("released")

    class FakeCv2:
        COLOR_RGB2BGR = "COLOR_RGB2BGR"

        def __init__(self) -> None:
            self.writer_calls: list[tuple[str, Any, float, tuple[int, int]]] = []

        def VideoWriter_fourcc(self, *_args: Any) -> str:
            return "mp4v"

        def VideoWriter(
            self, path: str, fourcc: Any, fps: float, size: tuple[int, int]
        ) -> FakeWriter:
            self.writer_calls.append((path, fourcc, fps, size))
            return FakeWriter()

        def cvtColor(self, frame: Any, conversion: Any) -> tuple[Any, Any, str]:
            return (frame, conversion, "converted")

    fake_cv2 = FakeCv2()

    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", fake_cv2))
    monkeypatch.setattr(
        video_module,
        "_coerce_frames_to_rgb_uint8",
        lambda _frames: [frame_a, frame_b],
    )

    save_video(frames=object(), path=output_path, fps=24.0)

    assert fake_cv2.writer_calls == [(str(output_path), "mp4v", 24.0, (12, 8))]
    assert written == [
        (frame_a, "COLOR_RGB2BGR", "converted"),
        (frame_b, "COLOR_RGB2BGR", "converted"),
        "released",
    ]


@pytest.mark.parametrize(
    ("filename", "expected_fourcc"),
    [
        ("output.mp4", "mp4v"),
        ("output.webm", "VP80"),
        ("output.MP4", "mp4v"),
        ("output.WEBM", "VP80"),
        ("output.avi", "mp4v"),
    ],
)
def test_save_video_selects_fourcc_based_on_extension(
    monkeypatch: Any, tmp_path: Path, filename: str, expected_fourcc: str
) -> None:
    output_path = tmp_path / filename
    fourcc_args: list[tuple[str, ...]] = []

    class FakeWriter:
        def isOpened(self) -> bool:
            return True

        def write(self, frame: Any) -> None:
            pass

        def release(self) -> None:
            pass

    class FakeCv2:
        COLOR_RGB2BGR = "COLOR_RGB2BGR"

        def VideoWriter_fourcc(self, *args: str) -> str:
            fourcc_args.append(args)
            return "".join(args)

        def VideoWriter(
            self, path: str, fourcc: Any, fps: float, size: tuple[int, int]
        ) -> FakeWriter:
            return FakeWriter()

        def cvtColor(self, frame: Any, conversion: Any) -> Any:
            return frame

    frame = SimpleNamespace(shape=(4, 4, 3))
    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", FakeCv2()))
    monkeypatch.setattr(video_module, "_coerce_frames_to_rgb_uint8", lambda _: [frame])

    save_video(frames=object(), path=output_path, fps=30.0)

    assert len(fourcc_args) == 1
    assert "".join(fourcc_args[0]) == expected_fourcc


def test_get_video_metadata_returns_frame_count_fps_width_height(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "meta.mp4"

    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, prop: float) -> float:
            return {
                1.0: 300.0,
                2.0: 30.0,
                3.0: 640.0,
                4.0: 360.0,
            }[float(prop)]

        def release(self) -> None:
            self.released = True

    fake_capture = FakeCapture()

    class FakeCv2:
        CAP_PROP_FRAME_COUNT = 1.0
        CAP_PROP_FPS = 2.0
        CAP_PROP_FRAME_WIDTH = 3.0
        CAP_PROP_FRAME_HEIGHT = 4.0

        def VideoCapture(self, path: str) -> FakeCapture:
            assert path == str(video_path)
            return fake_capture

    monkeypatch.setattr(video_module, "_get_video_backend", lambda: ("cv2", FakeCv2()))

    metadata = get_video_metadata(video_path)

    assert metadata == {"frame_count": 300, "fps": 30.0, "width": 640, "height": 360}
    assert fake_capture.released is True


def test_get_video_components_is_callable() -> None:
    """AC01 — get_video_components is accessible from comfy_diffusion.video."""
    assert callable(get_video_components)


def test_get_video_components_lazily_imports_comfy_extras_node(
    monkeypatch: Any,
) -> None:
    """AC02 — lazily imports comfy_extras.nodes_video.GetVideoComponents."""
    sentinel_images = object()
    sentinel_audio = object()

    class FakeGetVideoComponents:
        @classmethod
        def execute(cls, video: Any) -> tuple[Any, ...]:
            return (sentinel_images, sentinel_audio, 24.0)

    monkeypatch.setattr(
        video_module, "_get_get_video_components_type", lambda: FakeGetVideoComponents
    )

    result = get_video_components(object())

    assert result[0] is sentinel_images
    assert result[1] is sentinel_audio


def test_get_video_components_returns_images_and_audio_tuple(
    monkeypatch: Any,
) -> None:
    """AC03 — returns a (images_tensor, audio) tuple."""
    img_tensor = object()
    audio_obj = object()

    class FakeGetVideoComponents:
        @classmethod
        def execute(cls, video: Any) -> tuple[Any, ...]:
            return (img_tensor, audio_obj, 30.0)

    monkeypatch.setattr(
        video_module, "_get_get_video_components_type", lambda: FakeGetVideoComponents
    )

    images, audio = get_video_components(object())

    assert images is img_tensor
    assert audio is audio_obj


def test_get_video_backend_raises_clear_error_when_video_extra_is_missing(
    monkeypatch: Any,
) -> None:
    real_import = builtins.__import__

    def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "cv2" or name == "imageio" or name.startswith("imageio."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    with pytest.raises(ModuleNotFoundError, match=r"comfy-diffusion\[video\]"):
        video_module._get_video_backend()


def test_ltxv_img_to_video_inplace_signature_matches_contract() -> None:
    sig = inspect.signature(ltxv_img_to_video_inplace)
    params = sig.parameters
    assert list(params.keys()) == ["vae", "image", "latent", "strength", "bypass"]
    assert params["strength"].default == 1.0
    assert params["bypass"].default is False
    assert str(sig.return_annotation) == "dict[str, Any]"


def test_ltxv_img_to_video_inplace_bypass_returns_input_latent_unchanged() -> None:
    sentinel = {"samples": object(), "noise_mask": object()}
    result = ltxv_img_to_video_inplace(
        vae=object(), image=object(), latent=sentinel, bypass=True
    )
    assert result is sentinel


def test_ltxv_img_to_video_inplace_bypass_does_not_call_vae_or_touch_image(
    monkeypatch: Any,
) -> None:
    class StrictVAE:
        def encode(self, _pixels: Any) -> Any:
            raise AssertionError("vae.encode must not be called when bypass=True")

    latent: dict[str, Any] = {"samples": object()}
    result = ltxv_img_to_video_inplace(
        vae=StrictVAE(), image=object(), latent=latent, bypass=True
    )
    assert result is latent


def test_ltxv_img_to_video_inplace_returns_samples_and_noise_mask(
    monkeypatch: Any,
) -> None:
    """AC03 — returned dict contains 'samples' and 'noise_mask'."""
    torch = pytest.importorskip("torch")

    batch, channels, latent_frames, lh, lw = 1, 16, 8, 4, 4
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}

    # Minimal encoded image: 1 frame
    encoded_t = torch.zeros(batch, channels, 1, lh, lw)
    pixels_out = torch.zeros(batch, lh * 8, lw * 8, 3)

    class FakeVAE:
        downscale_index_formula = (1, 8, 8)

        def encode(self, pixels: Any) -> Any:
            return encoded_t

    # Patch comfy.utils so the test doesn't need ComfyUI vendored
    import types

    fake_comfy = types.ModuleType("comfy")
    fake_comfy_utils = types.ModuleType("comfy.utils")

    def fake_common_upscale(img: Any, w: int, h: int, mode: str, crop: str) -> Any:
        return img

    fake_comfy_utils.common_upscale = fake_common_upscale
    fake_comfy.utils = fake_comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)

    image = torch.zeros(batch, lh * 8, lw * 8, 3)
    result = ltxv_img_to_video_inplace(
        vae=FakeVAE(), image=image, latent=latent, strength=1.0
    )

    assert "samples" in result
    assert "noise_mask" in result
    mask = result["noise_mask"]
    assert mask.shape == (batch, 1, latent_frames, 1, 1)


def test_ltxv_img_to_video_inplace_noise_mask_values_reflect_strength(
    monkeypatch: Any,
) -> None:
    """AC03 — noise_mask first frames set to 1.0 - strength."""
    torch = pytest.importorskip("torch")

    batch, channels, latent_frames, lh, lw = 1, 16, 4, 2, 2
    samples = torch.zeros(batch, channels, latent_frames, lh, lw)
    latent: dict[str, Any] = {"samples": samples}

    encoded_t = torch.zeros(batch, channels, 1, lh, lw)

    class FakeVAE:
        downscale_index_formula = (1, 8, 8)

        def encode(self, pixels: Any) -> Any:
            return encoded_t

    import types

    fake_comfy = types.ModuleType("comfy")
    fake_comfy_utils = types.ModuleType("comfy.utils")
    fake_comfy_utils.common_upscale = lambda img, w, h, mode, crop: img
    fake_comfy.utils = fake_comfy_utils
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", fake_comfy_utils)

    image = torch.zeros(batch, lh * 8, lw * 8, 3)
    strength = 0.6
    result = ltxv_img_to_video_inplace(
        vae=FakeVAE(), image=image, latent=latent, strength=strength
    )

    mask = result["noise_mask"]
    # First encoded frames: 1.0 - strength
    assert abs(float(mask[0, 0, 0, 0, 0]) - (1.0 - strength)) < 1e-5
    # Remaining frames: 1.0
    assert float(mask[0, 0, 1, 0, 0]) == pytest.approx(1.0)


def test_ltxv_img_to_video_inplace_no_top_level_torch_or_comfy_imports() -> None:
    """AC05 — no top-level torch/comfy imports in video module."""
    result = _run_python(
        "import json, sys\n"
        "from comfy_diffusion import video as _v\n"
        "print(json.dumps({\n"
        "  'torch': 'torch' in sys.modules,\n"
        "  'comfy': 'comfy' in sys.modules,\n"
        "}))\n"
    )
    payload = json.loads(result.stdout)
    assert payload["torch"] is False
    assert payload["comfy"] is False


def test_import_comfy_diffusion_video_has_no_cv2_or_imageio_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.video import load_video, save_video, get_video_metadata, get_video_components\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'load_video': load_video.__name__,\n"
        "  'save_video': save_video.__name__,\n"
        "  'get_video_metadata': get_video_metadata.__name__,\n"
        "  'get_video_components': get_video_components.__name__,\n"
        "  'cv2_loaded': 'cv2' in sys.modules,\n"
        "  'imageio_loaded': any(m.startswith('imageio') for m in sys.modules),\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["load_video"] == "load_video"
    assert payload["save_video"] == "save_video"
    assert payload["get_video_metadata"] == "get_video_metadata"
    assert payload["get_video_components"] == "get_video_components"
    assert payload["cv2_loaded"] is False
    assert payload["imageio_loaded"] is False
    heavy = [m for m in payload["new_modules"] if m == "cv2" or m.startswith("imageio")]
    assert heavy == []
