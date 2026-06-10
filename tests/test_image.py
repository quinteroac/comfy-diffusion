"""Tests for image loading helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageOps, features

import comfy_diffusion
import comfy_diffusion.image as image_module
from comfy_diffusion.image import (
    canny,
    dw_preprocessor,
    empty_image,
    image_batch_extend_with_overlap,
    image_composite_masked,
    image_from_batch,
    image_invert,
    image_pad_for_outpaint,
    image_resize_kj,
    image_scale_by,
    image_to_tensor,
    image_upscale_with_model,
    load_image,
    ltxv_preprocess,
    math_expression,
    repeat_image_batch,
    resize_image_mask,
    resize_images_by_longer_edge,
)


class _FakeTensor:
    def __init__(self, data: Any, dtype: object, shape: tuple[int, ...] | None = None) -> None:
        if shape is None:
            shape = self._infer_shape(data)
        self._shape = shape
        self._data = self._flatten(data)
        self.dtype = dtype

    @classmethod
    def full(cls, shape: tuple[int, ...], fill: float, dtype: object) -> _FakeTensor:
        total = 1
        for dim in shape:
            total *= dim
        return cls(data=[float(fill)] * total, dtype=dtype, shape=shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def size(self) -> tuple[int, ...]:
        return self._shape

    def tolist(self) -> Any:
        return self._unflatten(self._data, self._shape)

    def unsqueeze(self, dim: int) -> _FakeTensor:
        if dim < 0:
            dim += len(self._shape) + 1
        if dim < 0 or dim > len(self._shape):
            raise IndexError("dimension out of range")
        new_shape = self._shape[:dim] + (1,) + self._shape[dim:]
        return _FakeTensor(data=self._data.copy(), dtype=self.dtype, shape=new_shape)

    def __mul__(self, scalar: float) -> _FakeTensor:
        return _FakeTensor(
            data=[value * float(scalar) for value in self._data],
            dtype=self.dtype,
            shape=self._shape,
        )

    __rmul__ = __mul__

    def __setitem__(self, key: Any, value: Any) -> None:
        index_lists = self._key_to_indices(key)
        flat_indices = list(self._iter_flat_indices(index_lists))
        if isinstance(value, _FakeTensor):
            source_values = value._data
            assert len(source_values) == len(flat_indices)
            for destination, source in zip(flat_indices, source_values):
                self._data[destination] = source
            return

        scalar = float(value)
        for destination in flat_indices:
            self._data[destination] = scalar

    def _key_to_indices(self, key: Any) -> list[list[int]]:
        if not isinstance(key, tuple):
            key = (key,)
        expanded_key = list(key) + [slice(None)] * (len(self._shape) - len(key))
        index_lists: list[list[int]] = []
        for dim, selector in zip(self._shape, expanded_key):
            if isinstance(selector, int):
                normalized = selector if selector >= 0 else dim + selector
                index_lists.append([normalized])
                continue
            start, stop, step = selector.indices(dim)
            index_lists.append(list(range(start, stop, step)))
        return index_lists

    def _iter_flat_indices(self, index_lists: list[list[int]]) -> Any:
        shape = self._shape

        def walk(dimension: int, coordinates: list[int]) -> Any:
            if dimension == len(index_lists):
                yield self._coords_to_flat_index(coordinates, shape)
                return
            for index in index_lists[dimension]:
                coordinates.append(index)
                yield from walk(dimension + 1, coordinates)
                coordinates.pop()

        yield from walk(0, [])

    @staticmethod
    def _coords_to_flat_index(coords: list[int], shape: tuple[int, ...]) -> int:
        index = 0
        stride = 1
        for dim_size, coord in zip(reversed(shape), reversed(coords)):
            index += coord * stride
            stride *= dim_size
        return index

    @staticmethod
    def _flatten(data: Any) -> list[float]:
        if isinstance(data, list):
            flattened: list[float] = []
            for value in data:
                flattened.extend(_FakeTensor._flatten(value))
            return flattened
        return [float(data)]

    @staticmethod
    def _infer_shape(data: Any) -> tuple[int, ...]:
        shape: list[int] = []
        current = data
        while isinstance(current, list):
            shape.append(len(current))
            current = current[0] if current else []
        return tuple(shape)

    @staticmethod
    def _unflatten(values: list[float], shape: tuple[int, ...]) -> Any:
        if not shape:
            return values[0]
        if len(shape) == 1:
            return values.copy()

        chunk = 1
        for dim in shape[1:]:
            chunk *= dim

        nested: list[Any] = []
        for offset in range(0, len(values), chunk):
            nested.append(_FakeTensor._unflatten(values[offset : offset + chunk], shape[1:]))
        return nested


class _FakeTorch:
    float32 = "torch.float32"

    @staticmethod
    def tensor(data: Any, *, dtype: object) -> _FakeTensor:
        return _FakeTensor(data=data, dtype=dtype)

    @staticmethod
    def ones(shape: tuple[int, ...], *, dtype: object) -> _FakeTensor:
        return _FakeTensor.full(shape=shape, fill=1.0, dtype=dtype)

    @staticmethod
    def zeros(shape: tuple[int, ...], *, dtype: object) -> _FakeTensor:
        return _FakeTensor.full(shape=shape, fill=0.0, dtype=dtype)


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [float(values)]


def _constant_bhwc_image(
    batch: int, height: int, width: int, color: tuple[float, float, float]
) -> _FakeTensor:
    pixel = list(color)
    return _FakeTorch.tensor(
        [[[[pixel[0], pixel[1], pixel[2]] for _ in range(width)] for _ in range(height)]
         for _ in range(batch)],
        dtype=_FakeTorch.float32,
    )


@pytest.fixture(autouse=True)
def _fake_torch_dependency(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        image_module,
        "_get_load_image_dependencies",
        lambda: (Image, ImageOps, _FakeTorch),
    )
    monkeypatch.setattr(image_module, "_get_torch_module", lambda: _FakeTorch)


def test_image_module_exports_expected_entrypoints() -> None:
    assert image_module.__all__ == [
        "load_image",
        "image_to_tensor",
        "images_to_tensor",
        "image_pad_for_outpaint",
        "image_upscale_with_model",
        "image_from_batch",
        "repeat_image_batch",
        "image_composite_masked",
        "ltxv_preprocess",
        "resize_image_mask",
        "resize_images_by_longer_edge",
        "empty_image",
        "math_expression",
        "canny",
        "image_invert",
        "image_scale_by",
        "dw_preprocessor",
        "image_resize_kj",
        "image_batch_extend_with_overlap",
        "image_scale_to_total_pixels",
        "image_scale_to_max_dimension",
        "get_image_size",
        "flux_kontext_image_scale",
    ]


def test_image_to_tensor_converts_pil_image_to_bhwc_float32() -> None:
    # 2×3 image: top row red, bottom row blue
    pil_image = Image.new("RGB", (3, 2))
    pil_image.putpixel((0, 0), (255, 0, 0))
    pil_image.putpixel((1, 0), (255, 0, 0))
    pil_image.putpixel((2, 0), (255, 0, 0))
    pil_image.putpixel((0, 1), (0, 0, 255))
    pil_image.putpixel((1, 1), (0, 0, 255))
    pil_image.putpixel((2, 1), (0, 0, 255))

    tensor = image_to_tensor(pil_image)

    # Shape: (B=1, H=2, W=3, C=3)
    assert tensor.shape == (1, 2, 3, 3)
    assert tensor.dtype == _FakeTorch.float32
    values = tensor.tolist()
    # Batch 0, row 0 → red
    assert values[0][0][0] == pytest.approx([1.0, 0.0, 0.0])
    assert values[0][0][2] == pytest.approx([1.0, 0.0, 0.0])
    # Batch 0, row 1 → blue
    assert values[0][1][0] == pytest.approx([0.0, 0.0, 1.0])


def test_image_to_tensor_converts_non_rgb_mode_to_rgb() -> None:
    # L-mode (grayscale) image with value 128 → RGB (128, 128, 128) → ~0.502
    pil_image = Image.new("L", (2, 2), color=128)

    tensor = image_to_tensor(pil_image)

    assert tensor.shape == (1, 2, 2, 3)
    values = tensor.tolist()
    for row in values[0]:
        for pixel in row:
            assert pixel == pytest.approx([128 / 255.0] * 3, abs=1e-5)


def test_load_image_signature_matches_contract() -> None:
    signature = inspect.signature(load_image)
    assert str(signature) == "(path: 'str | Path') -> 'tuple[Any, Any]'"


def test_load_image_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "load_image")


def test_image_upscale_with_model_signature_matches_contract() -> None:
    signature = inspect.signature(image_upscale_with_model)
    assert str(signature) == "(upscale_model: 'Any', image: 'Any') -> 'Any'"


def test_image_upscale_with_model_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_upscale_with_model")


def test_image_from_batch_signature_matches_contract() -> None:
    signature = inspect.signature(image_from_batch)
    assert str(signature) == "(image: 'Any', batch_index: 'int', length: 'int' = 1) -> 'Any'"


def test_image_from_batch_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_from_batch")


def test_repeat_image_batch_signature_matches_contract() -> None:
    signature = inspect.signature(repeat_image_batch)
    assert str(signature) == "(image: 'Any', amount: 'int') -> 'Any'"


def test_repeat_image_batch_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "repeat_image_batch")


def test_image_composite_masked_signature_matches_contract() -> None:
    signature = inspect.signature(image_composite_masked)
    expected_signature = (
        "(destination: 'Any', source: 'Any', mask: 'Any', x: 'int', y: 'int') -> 'Any'"
    )
    assert str(signature) == expected_signature


def test_image_composite_masked_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_composite_masked")


def test_ltxv_preprocess_signature_matches_contract() -> None:
    signature = inspect.signature(ltxv_preprocess)
    assert str(signature) == "(image: 'Any', width: 'int', height: 'int', img_compression: 'int' = 35) -> 'Any'"


def test_ltxv_preprocess_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "ltxv_preprocess")


def test_ltxv_preprocess_resizes_and_runs_comfy_ltxv_preprocess(monkeypatch: Any) -> None:
    class FakeMovedimTensor:
        def __init__(self, transitions: dict[tuple[int, int], Any]) -> None:
            self.transitions = transitions
            self.calls: list[tuple[int, int]] = []

        def movedim(self, source: int, destination: int) -> Any:
            self.calls.append((source, destination))
            return self.transitions[(source, destination)]

    output_tensor = _FakeTorch.tensor(
        [[[[0.25, 0.5, 0.75], [0.1, 0.2, 0.3]]]],
        dtype=_FakeTorch.float32,
    )
    resized_bchw = FakeMovedimTensor({(1, -1): output_tensor})
    chw_input = object()
    bhwc_input = FakeMovedimTensor({(-1, 1): chw_input})

    calls: dict[str, Any] = {}

    class FakeComfyUtils:
        @staticmethod
        def common_upscale(
            image: Any,
            width: int,
            height: int,
            upscale_method: str,
            crop: str,
        ) -> Any:
            calls["common_upscale"] = {
                "image": image,
                "width": width,
                "height": height,
                "upscale_method": upscale_method,
                "crop": crop,
            }
            return resized_bchw

    class FakeLTXVPreprocess:
        @classmethod
        def execute(cls, image: Any, img_compression: int) -> tuple[Any]:
            calls["ltxv_preprocess"] = {
                "image": image,
                "img_compression": img_compression,
            }
            return (image,)

    monkeypatch.setattr(
        image_module,
        "_get_ltxv_preprocess_dependencies",
        lambda: (FakeComfyUtils, FakeLTXVPreprocess),
    )

    result = ltxv_preprocess(bhwc_input, width=768, height=512)

    assert bhwc_input.calls == [(-1, 1)]
    assert resized_bchw.calls == [(1, -1)]
    assert calls["common_upscale"] == {
        "image": chw_input,
        "width": 768,
        "height": 512,
        "upscale_method": "bilinear",
        "crop": "center",
    }
    assert calls["ltxv_preprocess"] == {
        "image": output_tensor,
        "img_compression": 35,
    }
    assert result is output_tensor
    assert result.shape == (1, 1, 2, 3)


def test_image_upscale_with_model_wraps_comfyui_node_and_returns_bhwc_tensor(
    monkeypatch: Any,
) -> None:
    upscale_model = object()
    image = object()
    expected_upscaled = object()

    class FakeImageUpscaleWithModel:
        @classmethod
        def execute(cls, received_upscale_model: Any, received_image: Any) -> tuple[Any]:
            assert received_upscale_model is upscale_model
            assert received_image is image
            return (expected_upscaled,)

    monkeypatch.setattr(
        image_module,
        "_get_image_upscale_with_model_type",
        lambda: FakeImageUpscaleWithModel,
    )

    upscaled = image_upscale_with_model(upscale_model, image)

    assert upscaled is expected_upscaled


def test_image_from_batch_extracts_contiguous_slice_and_returns_bhwc_tensor(
    monkeypatch: Any,
) -> None:
    image = _FakeTorch.tensor(
        [
            [[[0.1, 0.0, 0.0]]],
            [[[0.2, 0.0, 0.0]]],
            [[[0.3, 0.0, 0.0]]],
            [[[0.4, 0.0, 0.0]]],
        ],
        dtype=_FakeTorch.float32,
    )

    class FakeImageFromBatch:
        @classmethod
        def execute(cls, *, image: Any, batch_index: int, length: int) -> tuple[Any]:
            assert image.shape == (4, 1, 1, 3)
            assert batch_index == 1
            assert length == 2
            sliced_values = image.tolist()[batch_index : batch_index + length]
            return (_FakeTorch.tensor(sliced_values, dtype=_FakeTorch.float32),)

    monkeypatch.setattr(
        image_module,
        "_get_image_from_batch_type",
        lambda: FakeImageFromBatch,
    )

    sliced = image_from_batch(image=image, batch_index=1, length=2)

    assert isinstance(sliced, _FakeTensor)
    assert sliced.shape == (2, 1, 1, 3)
    assert sliced.tolist() == [
        [[[0.2, 0.0, 0.0]]],
        [[[0.3, 0.0, 0.0]]],
    ]


def test_repeat_image_batch_repeats_batch_dimension_and_returns_bhwc_tensor(
    monkeypatch: Any,
) -> None:
    image = _FakeTorch.tensor(
        [
            [[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]],
            [[[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]],
        ],
        dtype=_FakeTorch.float32,
    )

    class FakeRepeatImageBatch:
        @classmethod
        def execute(cls, *, image: Any, amount: int) -> tuple[Any]:
            assert image.shape == (2, 1, 2, 3)
            assert amount == 3
            source = image.tolist()
            repeated: list[Any] = []
            for _ in range(amount):
                repeated.extend(source)
            return (_FakeTorch.tensor(repeated, dtype=_FakeTorch.float32),)

    monkeypatch.setattr(
        image_module,
        "_get_repeat_image_batch_type",
        lambda: FakeRepeatImageBatch,
    )

    repeated = repeat_image_batch(image=image, amount=3)

    assert isinstance(repeated, _FakeTensor)
    assert repeated.shape == (6, 1, 2, 3)
    assert repeated.tolist() == image.tolist() * 3


def test_image_composite_masked_composites_source_using_coordinates_and_mask(
    monkeypatch: Any,
) -> None:
    destination = _FakeTorch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
            ]
        ],
        dtype=_FakeTorch.float32,
    )
    source = _FakeTorch.tensor(
        [[[[0.9, 0.9, 0.9]]]],
        dtype=_FakeTorch.float32,
    )
    mask = _FakeTorch.tensor(
        [[[1.0]]],
        dtype=_FakeTorch.float32,
    )

    class FakeImageCompositeMasked:
        @classmethod
        def execute(
            cls,
            destination: Any,
            source: Any,
            x: int,
            y: int,
            resize_source: bool,
            mask: Any = None,
        ) -> tuple[Any]:
            assert x == 1
            assert y == 0
            assert resize_source is False
            assert mask is not None

            output = _FakeTorch.tensor(destination.tolist(), dtype=_FakeTorch.float32)
            destination_pixel = destination.tolist()[0][y][x]
            source_pixel = source.tolist()[0][0][0]
            alpha = mask.tolist()[0][0][0]
            blended_pixel = [
                alpha * source_channel + (1.0 - alpha) * destination_channel
                for source_channel, destination_channel in zip(source_pixel, destination_pixel)
            ]
            output[0, y, x, :] = _FakeTorch.tensor(blended_pixel, dtype=_FakeTorch.float32)
            return (output,)

    monkeypatch.setattr(
        image_module,
        "_get_image_composite_masked_type",
        lambda: FakeImageCompositeMasked,
    )

    composited = image_composite_masked(
        destination=destination,
        source=source,
        mask=mask,
        x=1,
        y=0,
    )

    assert isinstance(composited, _FakeTensor)
    assert composited.shape == (1, 2, 2, 3)
    assert composited.tolist() == [
        [
            [[0.0, 0.0, 0.0], [0.9, 0.9, 0.9]],
            [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
        ]
    ]


@pytest.mark.parametrize(
    ("mask_value", "expected_pixel"),
    [
        (1.0, [0.8, 0.6, 0.4]),
        (0.0, [0.1, 0.2, 0.3]),
    ],
)
def test_image_composite_masked_mask_controls_blending(
    monkeypatch: Any,
    mask_value: float,
    expected_pixel: list[float],
) -> None:
    destination = _FakeTorch.tensor([[[[0.1, 0.2, 0.3]]]], dtype=_FakeTorch.float32)
    source = _FakeTorch.tensor([[[[0.8, 0.6, 0.4]]]], dtype=_FakeTorch.float32)
    mask = _FakeTorch.tensor([[[mask_value]]], dtype=_FakeTorch.float32)

    class FakeImageCompositeMasked:
        @classmethod
        def execute(
            cls,
            destination: Any,
            source: Any,
            x: int,
            y: int,
            resize_source: bool,
            mask: Any = None,
        ) -> tuple[Any]:
            alpha = mask.tolist()[0][0][0]
            source_pixel = source.tolist()[0][0][0]
            destination_pixel = destination.tolist()[0][0][0]
            blended_pixel = [
                alpha * source_channel + (1.0 - alpha) * destination_channel
                for source_channel, destination_channel in zip(source_pixel, destination_pixel)
            ]
            return (_FakeTorch.tensor([[[blended_pixel]]], dtype=_FakeTorch.float32),)

    monkeypatch.setattr(
        image_module,
        "_get_image_composite_masked_type",
        lambda: FakeImageCompositeMasked,
    )

    composited = image_composite_masked(
        destination=destination,
        source=source,
        mask=mask,
        x=0,
        y=0,
    )

    assert composited.shape == (1, 1, 1, 3)
    assert composited.tolist()[0][0][0] == expected_pixel


def test_image_upscale_with_model_supports_comfyui_v3_result_output(
    monkeypatch: Any,
) -> None:
    upscale_model = object()
    image = object()
    expected_upscaled = object()

    class FakeNodeOutput:
        def __init__(self, result: tuple[Any, ...]) -> None:
            self.result = result

    class FakeImageUpscaleWithModel:
        @classmethod
        def execute(cls, received_upscale_model: Any, received_image: Any) -> Any:
            assert received_upscale_model is upscale_model
            assert received_image is image
            return FakeNodeOutput((expected_upscaled,))

    monkeypatch.setattr(
        image_module,
        "_get_image_upscale_with_model_type",
        lambda: FakeImageUpscaleWithModel,
    )

    upscaled = image_upscale_with_model(upscale_model, image)

    assert upscaled is expected_upscaled


def test_load_image_returns_bhwc_float32_image_and_hw_mask_from_alpha(tmp_path: Path) -> None:
    image_path = tmp_path / "alpha.png"
    source = Image.new("RGBA", (2, 2))
    source.putdata(
        [
            (255, 0, 0, 255),
            (0, 255, 0, 128),
            (0, 0, 255, 0),
            (255, 255, 255, 64),
        ]
    )
    source.save(image_path, format="PNG")

    image_tensor, mask_tensor = load_image(image_path)

    assert image_tensor.dtype == _FakeTorch.float32
    assert mask_tensor.dtype == _FakeTorch.float32
    assert image_tensor.shape == (1, 2, 2, 3)
    assert mask_tensor.shape == (2, 2)
    assert all(0.0 <= value <= 1.0 for value in _flatten(image_tensor.tolist()))
    assert mask_tensor.tolist() == [
        [0.0, 127 / 255.0],
        [1.0, 191 / 255.0],
    ]


@pytest.mark.parametrize(
    ("suffix", "format_name"),
    [
        (".png", "PNG"),
        (".jpg", "JPEG"),
        (".webp", "WEBP"),
    ],
)
def test_load_image_supports_common_formats(tmp_path: Path, suffix: str, format_name: str) -> None:
    if format_name == "WEBP" and not features.check("webp"):
        pytest.skip("Pillow WEBP support is unavailable in this environment")

    image_path = tmp_path / f"sample{suffix}"
    Image.new("RGB", (3, 2), color=(12, 34, 56)).save(image_path, format=format_name)

    image_tensor, mask_tensor = load_image(image_path)

    assert image_tensor.shape == (1, 2, 3, 3)
    assert mask_tensor.shape == (2, 3)


def test_load_image_without_alpha_returns_fully_opaque_zero_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.jpg"
    Image.new("RGB", (2, 3), color=(11, 22, 33)).save(image_path, format="JPEG")

    _, mask_tensor = load_image(image_path)

    assert mask_tensor.shape == (3, 2)
    assert mask_tensor.tolist() == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "oriented.jpg"
    source = Image.new("RGB", (2, 3))
    source.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
    )
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    source.save(image_path, format="JPEG", exif=exif)

    image_tensor, mask_tensor = load_image(image_path)

    assert image_tensor.shape == (1, 2, 3, 3)
    assert mask_tensor.shape == (2, 3)


def test_image_pad_for_outpaint_signature_matches_contract() -> None:
    signature = inspect.signature(image_pad_for_outpaint)
    expected_signature = (
        "(image: 'Any', left: 'int', top: 'int', right: 'int', "
        "bottom: 'int', feathering: 'int') -> 'tuple[Any, Any]'"
    )
    assert str(signature) == expected_signature


def test_image_pad_for_outpaint_returns_padded_image_and_mask_in_bhwc() -> None:
    source = _FakeTorch.tensor(
        [
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]],
            ]
        ],
        dtype=_FakeTorch.float32,
    )

    padded_image, padded_mask = image_pad_for_outpaint(
        source, left=1, top=2, right=3, bottom=1, feathering=0
    )

    assert padded_image.shape == (1, 5, 6, 3)
    assert padded_mask.shape == (1, 5, 6)
    assert padded_image.dtype == _FakeTorch.float32
    assert padded_mask.dtype == _FakeTorch.float32

    padded_image_values = padded_image.tolist()
    assert padded_image_values[0][2][1] == [0.1, 0.2, 0.3]
    assert padded_image_values[0][3][2] == [0.0, 0.1, 0.2]
    assert padded_image_values[0][0][0] == [0.5, 0.5, 0.5]

    expected_mask = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    assert padded_mask.tolist()[0] == expected_mask


def test_image_pad_for_outpaint_applies_feathering_gradient() -> None:
    source = _constant_bhwc_image(batch=1, height=6, width=6, color=(0.1, 0.1, 0.1))

    _, padded_mask = image_pad_for_outpaint(source, left=1, top=1, right=1, bottom=1, feathering=2)
    mask_values = padded_mask.tolist()[0]

    assert mask_values[1][1] == 1.0
    assert mask_values[2][2] == 0.25
    assert mask_values[3][3] == 0.0


def test_resize_image_mask_signature_matches_contract() -> None:
    signature = inspect.signature(resize_image_mask)
    expected = (
        "(image: 'Any', mask: 'Any', width: 'int', height: 'int',"
        " interpolation: 'str' = 'bilinear') -> 'tuple[Any, Any]'"
    )
    assert str(signature) == expected


def test_resize_image_mask_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "resize_image_mask")


def test_resize_image_mask_calls_comfy_node_and_returns_image_mask_tuple(
    monkeypatch: Any,
) -> None:
    image_in = object()
    mask_in = object()
    image_out = object()
    mask_out = object()

    received: dict[str, Any] = {}

    class FakeResizeImageMaskNode:
        @classmethod
        def execute(
            cls,
            *,
            image: Any,
            mask: Any,
            width: int,
            height: int,
            interpolation: str,
        ) -> tuple[Any, Any]:
            received["image"] = image
            received["mask"] = mask
            received["width"] = width
            received["height"] = height
            received["interpolation"] = interpolation
            return (image_out, mask_out)

    monkeypatch.setattr(
        image_module,
        "_get_resize_image_mask_node_type",
        lambda: FakeResizeImageMaskNode,
    )

    result_image, result_mask = resize_image_mask(image_in, mask_in, 512, 768)

    assert result_image is image_out
    assert result_mask is mask_out
    assert received["image"] is image_in
    assert received["mask"] is mask_in
    assert received["width"] == 512
    assert received["height"] == 768
    assert received["interpolation"] == "bilinear"


def test_resize_image_mask_passes_custom_interpolation(monkeypatch: Any) -> None:
    image_in = object()
    mask_in = object()
    image_out = object()
    mask_out = object()

    received: dict[str, Any] = {}

    class FakeResizeImageMaskNode:
        @classmethod
        def execute(
            cls,
            *,
            image: Any,
            mask: Any,
            width: int,
            height: int,
            interpolation: str,
        ) -> tuple[Any, Any]:
            received["interpolation"] = interpolation
            return (image_out, mask_out)

    monkeypatch.setattr(
        image_module,
        "_get_resize_image_mask_node_type",
        lambda: FakeResizeImageMaskNode,
    )

    resize_image_mask(image_in, mask_in, 256, 256, interpolation="nearest")

    assert received["interpolation"] == "nearest"


def test_resize_image_mask_supports_comfyui_v3_result_output(monkeypatch: Any) -> None:
    image_out = object()
    mask_out = object()

    class FakeNodeOutput:
        def __init__(self, result: tuple[Any, ...]) -> None:
            self.result = result

    class FakeResizeImageMaskNode:
        @classmethod
        def execute(cls, *, image: Any, mask: Any, width: int, height: int, interpolation: str) -> Any:
            return FakeNodeOutput((image_out, mask_out))

    monkeypatch.setattr(
        image_module,
        "_get_resize_image_mask_node_type",
        lambda: FakeResizeImageMaskNode,
    )

    result_image, result_mask = resize_image_mask(object(), object(), 64, 64)

    assert result_image is image_out
    assert result_mask is mask_out


def test_resize_images_by_longer_edge_signature_matches_contract() -> None:
    signature = inspect.signature(resize_images_by_longer_edge)
    assert str(signature) == "(images: 'Any', size: 'int') -> 'Any'"


def test_resize_images_by_longer_edge_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "resize_images_by_longer_edge")


def test_resize_images_by_longer_edge_delegates_to_comfyui_node(monkeypatch: Any) -> None:
    input_images = object()
    expected_output = _FakeTorch.tensor(
        [[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]],
        dtype=_FakeTorch.float32,
    )
    captured: dict[str, Any] = {}

    class FakeResizeImagesByLongerEdgeNode:
        @classmethod
        def execute(cls, *, images: Any, size: int) -> tuple[Any]:
            captured["images"] = images
            captured["size"] = size
            return (expected_output,)

    monkeypatch.setattr(
        image_module,
        "_get_resize_images_by_longer_edge_node_type",
        lambda: FakeResizeImagesByLongerEdgeNode,
    )

    result = resize_images_by_longer_edge(input_images, 512)

    assert captured["images"] is input_images
    assert captured["size"] == 512
    assert result is expected_output


def test_resize_images_by_longer_edge_supports_comfyui_v3_result_output(
    monkeypatch: Any,
) -> None:
    expected_output = object()

    class FakeNodeOutput:
        def __init__(self, result: tuple[Any, ...]) -> None:
            self.result = result

    class FakeResizeImagesByLongerEdgeNode:
        @classmethod
        def execute(cls, *, images: Any, size: int) -> Any:
            return FakeNodeOutput((expected_output,))

    monkeypatch.setattr(
        image_module,
        "_get_resize_images_by_longer_edge_node_type",
        lambda: FakeResizeImagesByLongerEdgeNode,
    )

    result = resize_images_by_longer_edge(object(), 768)

    assert result is expected_output


# ---------------------------------------------------------------------------
# empty_image tests (US-003)
# ---------------------------------------------------------------------------


def test_empty_image_is_callable() -> None:
    assert callable(empty_image)


def test_empty_image_signature_matches_contract() -> None:
    signature = inspect.signature(empty_image)
    assert str(signature) == "(width: 'int', height: 'int', batch_size: 'int' = 1, color: 'int' = 0) -> 'Any'"


def test_empty_image_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "empty_image")


def test_empty_image_delegates_to_nodes_empty_image(monkeypatch: Any) -> None:
    expected_output = _FakeTorch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
        dtype=_FakeTorch.float32,
    )
    captured: dict[str, Any] = {}

    class FakeEmptyImage:
        @classmethod
        def execute(cls, *, width: int, height: int, batch_size: int, color: int) -> tuple[Any]:
            captured["width"] = width
            captured["height"] = height
            captured["batch_size"] = batch_size
            captured["color"] = color
            return (expected_output,)

    monkeypatch.setattr(
        image_module,
        "_get_empty_image_type",
        lambda: FakeEmptyImage,
    )

    result = empty_image(width=64, height=32, batch_size=2, color=0xFF0000)

    assert captured["width"] == 64
    assert captured["height"] == 32
    assert captured["batch_size"] == 2
    assert captured["color"] == 0xFF0000
    assert result is expected_output


def test_empty_image_default_args(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeEmptyImage:
        @classmethod
        def execute(cls, *, width: int, height: int, batch_size: int, color: int) -> tuple[Any]:
            captured["batch_size"] = batch_size
            captured["color"] = color
            return (object(),)

    monkeypatch.setattr(image_module, "_get_empty_image_type", lambda: FakeEmptyImage)

    empty_image(width=8, height=8)

    assert captured["batch_size"] == 1
    assert captured["color"] == 0


def test_empty_image_returns_tensor(monkeypatch: Any) -> None:
    expected = _FakeTorch.tensor(
        [[[[1.0, 1.0, 1.0]]]],
        dtype=_FakeTorch.float32,
    )

    class FakeEmptyImage:
        @classmethod
        def execute(cls, *, width: int, height: int, batch_size: int, color: int) -> tuple[Any]:
            return (expected,)

    monkeypatch.setattr(image_module, "_get_empty_image_type", lambda: FakeEmptyImage)

    result = empty_image(width=1, height=1)

    assert result is expected
    assert result.shape == (1, 1, 1, 3)


def test_empty_image_supports_comfyui_v3_result_output(monkeypatch: Any) -> None:
    expected_output = object()

    class FakeNodeOutput:
        def __init__(self, result: tuple[Any, ...]) -> None:
            self.result = result

    class FakeEmptyImage:
        @classmethod
        def execute(cls, *, width: int, height: int, batch_size: int, color: int) -> Any:
            return FakeNodeOutput((expected_output,))

    monkeypatch.setattr(image_module, "_get_empty_image_type", lambda: FakeEmptyImage)

    result = empty_image(width=4, height=4)

    assert result is expected_output


def test_math_expression_signature_matches_contract() -> None:
    signature = inspect.signature(math_expression)
    assert str(signature) == "(expression: 'str', **kwargs: 'float') -> 'int | float'"


def test_math_expression_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "math_expression")


def test_math_expression_evaluates_expression_with_kwargs(monkeypatch: Any) -> None:
    class FakeMathExpressionNode:
        @classmethod
        def execute(cls, *, expression: str, values: dict) -> tuple:
            result = eval(expression, {}, values)  # noqa: S307 — test-only eval
            return (float(result), int(result))

    monkeypatch.setattr(
        image_module,
        "_get_math_expression_node_type",
        lambda: FakeMathExpressionNode,
    )

    result = math_expression("a * fps", a=10.0, fps=24.0)

    assert result == pytest.approx(240.0)


def test_math_expression_returns_numeric_type(monkeypatch: Any) -> None:
    class FakeMathExpressionNode:
        @classmethod
        def execute(cls, *, expression: str, values: dict) -> tuple:
            return (42.0, 42)

    monkeypatch.setattr(
        image_module,
        "_get_math_expression_node_type",
        lambda: FakeMathExpressionNode,
    )

    result = math_expression("42")

    assert isinstance(result, (int, float))


def test_math_expression_supports_comfyui_v3_node_output(monkeypatch: Any) -> None:
    class FakeNodeOutput:
        def __init__(self, result: tuple) -> None:
            self.result = result

    class FakeMathExpressionNode:
        @classmethod
        def execute(cls, *, expression: str, values: dict) -> Any:
            return FakeNodeOutput((7.5, 7))

    monkeypatch.setattr(
        image_module,
        "_get_math_expression_node_type",
        lambda: FakeMathExpressionNode,
    )

    result = math_expression("x + y", x=3.0, y=4.5)

    assert result == pytest.approx(7.5)


def test_math_expression_lazy_import_uses_comfy_extras() -> None:
    # Verify structurally that the getter defers the import to call time.
    import inspect as _inspect

    src = _inspect.getsource(image_module._get_math_expression_node_type)
    assert "comfy_extras.nodes_math" in src
    assert "MathExpressionNode" in src


# ---------------------------------------------------------------------------
# canny tests (US-007)
# ---------------------------------------------------------------------------


def test_canny_is_callable() -> None:
    assert callable(canny)


def test_canny_signature_matches_contract() -> None:
    signature = inspect.signature(canny)
    assert str(signature) == (
        "(image: 'Any', low_threshold: 'int' = 100, high_threshold: 'int' = 200) -> 'Any'"
    )


def test_canny_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "canny")


def test_canny_lazy_import_uses_comfy_extras_nodes_canny() -> None:
    import inspect as _inspect

    src = _inspect.getsource(image_module._get_canny_type)
    assert "comfy_extras.nodes_canny" in src
    assert "Canny" in src


def test_canny_returns_same_spatial_dimensions(monkeypatch: Any) -> None:
    input_image = _FakeTorch.tensor(
        [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [0.0, 0.1, 0.2]]]],
        dtype=_FakeTorch.float32,
    )
    # Output has the same (B, H, W, C) shape as input.
    expected_output = _FakeTorch.tensor(
        [[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]],
        dtype=_FakeTorch.float32,
    )
    captured: dict[str, Any] = {}

    class FakeCanny:
        @classmethod
        def execute(cls, *, image: Any, low_threshold: float, high_threshold: float) -> tuple[Any]:
            captured["image"] = image
            captured["low_threshold"] = low_threshold
            captured["high_threshold"] = high_threshold
            return (expected_output,)

    monkeypatch.setattr(image_module, "_get_canny_type", lambda: FakeCanny)

    result = canny(input_image)

    assert result is expected_output
    assert result.shape == input_image.shape


def test_canny_normalises_thresholds_to_zero_one_range(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeCanny:
        @classmethod
        def execute(cls, *, image: Any, low_threshold: float, high_threshold: float) -> tuple[Any]:
            captured["low_threshold"] = low_threshold
            captured["high_threshold"] = high_threshold
            return (image,)

    monkeypatch.setattr(image_module, "_get_canny_type", lambda: FakeCanny)

    canny(object(), low_threshold=100, high_threshold=200)

    assert captured["low_threshold"] == pytest.approx(100 / 255.0)
    assert captured["high_threshold"] == pytest.approx(200 / 255.0)


def test_canny_default_threshold_values(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    class FakeCanny:
        @classmethod
        def execute(cls, *, image: Any, low_threshold: float, high_threshold: float) -> tuple[Any]:
            captured["low_threshold"] = low_threshold
            captured["high_threshold"] = high_threshold
            return (image,)

    monkeypatch.setattr(image_module, "_get_canny_type", lambda: FakeCanny)

    canny(object())

    assert captured["low_threshold"] == pytest.approx(100 / 255.0)
    assert captured["high_threshold"] == pytest.approx(200 / 255.0)



# ---------------------------------------------------------------------------
# image_invert tests (US-010)
# ---------------------------------------------------------------------------


def test_image_invert_is_callable() -> None:
    assert callable(image_invert)


def test_image_invert_signature_matches_contract() -> None:
    signature = inspect.signature(image_invert)
    assert str(signature) == "(image: 'Any') -> 'Any'"


def test_image_invert_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_invert")


def test_image_invert_lazy_import_uses_nodes_image_invert() -> None:
    import inspect as _inspect

    src = _inspect.getsource(image_module._get_image_invert_type)
    assert "nodes" in src
    assert "ImageInvert" in src
    assert "ensure_comfyui_on_path" in src


def test_image_invert_delegates_to_comfy_node(monkeypatch: Any) -> None:
    input_image = _FakeTorch.tensor(
        [[[[0.2, 0.4, 0.6]]]],
        dtype=_FakeTorch.float32,
    )
    expected_output = _FakeTorch.tensor(
        [[[[0.8, 0.6, 0.4]]]],
        dtype=_FakeTorch.float32,
    )
    captured: dict[str, Any] = {}

    class FakeImageInvert:
        @classmethod
        def execute(cls, *, image: Any) -> tuple[Any]:
            captured["image"] = image
            return (expected_output,)

    monkeypatch.setattr(image_module, "_get_image_invert_type", lambda: FakeImageInvert)

    result = image_invert(input_image)

    assert captured["image"] is input_image
    assert result is expected_output


def test_image_invert_returns_image_tensor(monkeypatch: Any) -> None:
    output = _FakeTorch.tensor(
        [[[[0.5, 0.5, 0.5]]]],
        dtype=_FakeTorch.float32,
    )

    class FakeImageInvert:
        @classmethod
        def execute(cls, *, image: Any) -> tuple[Any]:
            return (output,)

    monkeypatch.setattr(image_module, "_get_image_invert_type", lambda: FakeImageInvert)

    result = image_invert(object())

    assert isinstance(result, _FakeTensor)
    assert result.shape == (1, 1, 1, 3)


def test_image_invert_supports_comfyui_v3_result_output(monkeypatch: Any) -> None:
    expected_output = object()

    class FakeNodeOutput:
        def __init__(self, result: tuple[Any, ...]) -> None:
            self.result = result

    class FakeImageInvert:
        @classmethod
        def execute(cls, *, image: Any) -> Any:
            return FakeNodeOutput((expected_output,))

    monkeypatch.setattr(image_module, "_get_image_invert_type", lambda: FakeImageInvert)

    result = image_invert(object())

    assert result is expected_output



# ---------------------------------------------------------------------------
# image_scale_by tests (US-001)
# ---------------------------------------------------------------------------


def test_image_scale_by_is_callable() -> None:
    assert callable(image_scale_by)


def test_image_scale_by_signature_matches_contract() -> None:
    signature = inspect.signature(image_scale_by)
    assert str(signature) == "(image: 'Any', upscale_method: 'str' = 'lanczos', scale_by: 'float' = 1.0) -> 'Any'"


def test_image_scale_by_in_dunder_all() -> None:
    assert "image_scale_by" in image_module.__all__


def test_image_scale_by_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_scale_by")


def test_image_scale_by_uses_floor_for_dimensions(monkeypatch: Any) -> None:
    """AC02: result dimensions are floor(h * scale_by) × floor(w * scale_by)."""
    import math

    calls: dict[str, Any] = {}

    # Input tensor: batch=1, h=7, w=5, channels=3 → movedim(-1,1) → (1,3,7,5)
    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 3, 7, 5)

        def movedim(self, src: int, dst: int) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    class _FakeMovedTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            # After movedim(-1,1): (1, 3, h, w) = (1, 3, 7, 5)
            return (1, 3, 7, 5)

        def movedim(self, src: int, dst: int) -> object:
            return _FakeOutput()

    class _FakeOutput:
        pass

    class _FakeComfyUtils:
        @staticmethod
        def common_upscale(
            samples: Any, width: int, height: int, upscale_method: str, crop: str
        ) -> _FakeMovedTensor:
            calls["width"] = width
            calls["height"] = height
            calls["upscale_method"] = upscale_method
            calls["crop"] = crop
            return _FakeMovedTensor()

    monkeypatch.setattr(image_module, "_get_comfy_utils", lambda: _FakeComfyUtils)

    scale_by = 0.5
    image_scale_by(_FakeInputTensor(), upscale_method="lanczos", scale_by=scale_by)

    assert calls["width"] == math.floor(5 * scale_by)
    assert calls["height"] == math.floor(7 * scale_by)
    assert calls["upscale_method"] == "lanczos"
    assert calls["crop"] == "disabled"


def test_image_scale_by_returns_image_tensor(monkeypatch: Any) -> None:
    """AC01: function is callable and returns an IMAGE tensor."""
    expected_output = object()

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 3, 4, 4)

        def movedim(self, src: int, dst: int) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    class _FakeMovedTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 3, 4, 4)

        def movedim(self, src: int, dst: int) -> object:
            return expected_output

    class _FakeComfyUtils:
        @staticmethod
        def common_upscale(
            samples: Any, width: int, height: int, upscale_method: str, crop: str
        ) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    monkeypatch.setattr(image_module, "_get_comfy_utils", lambda: _FakeComfyUtils)

    result = image_scale_by(_FakeInputTensor(), upscale_method="lanczos", scale_by=0.5)

    assert result is expected_output


def test_image_scale_by_lazy_import_no_top_level_torch_or_comfy() -> None:
    """AC04: no top-level torch/comfy imports in image.py."""
    import ast
    import pathlib

    source = pathlib.Path(image_module.__file__).read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # These must not appear at module top level (i.e. directly inside Module body)
            pass

    top_level_imports = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    forbidden = {"torch", "comfy"}
    for imp in top_level_imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                assert alias.name.split(".")[0] not in forbidden, (
                    f"Forbidden top-level import: {alias.name}"
                )
        elif isinstance(imp, ast.ImportFrom) and imp.module:
            assert imp.module.split(".")[0] not in forbidden, (
                f"Forbidden top-level import: {imp.module}"
            )


# ---------------------------------------------------------------------------
# dw_preprocessor tests (US-002)
# ---------------------------------------------------------------------------


def test_dw_preprocessor_is_callable() -> None:
    assert callable(dw_preprocessor)


def test_dw_preprocessor_signature_matches_contract() -> None:
    signature = inspect.signature(dw_preprocessor)
    assert str(signature) == (
        "(image: 'Any', detect_hand: 'bool' = True, detect_body: 'bool' = True, "
        "detect_face: 'bool' = True, resolution: 'int' = 512) -> 'Any'"
    )


def test_dw_preprocessor_in_dunder_all() -> None:
    assert "dw_preprocessor" in image_module.__all__


def test_dw_preprocessor_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "dw_preprocessor")


def test_dw_preprocessor_returns_tensor_with_same_batch_dim(monkeypatch: Any) -> None:
    """AC01: returns IMAGE tensor with same batch dimension as input."""
    import numpy as np

    batch_size = 3

    class _FakeTensorItem:
        def cpu(self) -> "_FakeTensorItem":
            return self

        def numpy(self) -> Any:
            return np.zeros((4, 4, 3), dtype=np.float32)

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (batch_size, 4, 4, 3)

        def __getitem__(self, idx: int) -> _FakeTensorItem:
            return _FakeTensorItem()

    pose_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    class _FakeDWposeDetector:
        def __call__(
            self,
            image: Any,
            *,
            detect_resolution: int,
            image_resolution: int,
            include_body: bool,
            include_hand: bool,
            include_face: bool,
            return_pil: bool,
        ) -> Any:
            return pose_frame

    stacked: list[Any] = []

    class _FakeTorchModule:
        float32 = "float32"

        @staticmethod
        def from_numpy(arr: Any) -> "_FakeTorchModule._FrameTensor":
            return _FakeTorchModule._FrameTensor(arr)

        @staticmethod
        def stack(frames: list[Any], dim: int) -> object:
            stacked.extend(frames)
            return object()

        class _FrameTensor:
            def __init__(self, arr: Any) -> None:
                self._arr = arr

            def float(self) -> "_FakeTorchModule._FrameTensor":
                return self

            def __truediv__(self, val: float) -> "_FakeTorchModule._FrameTensor":
                return self

    monkeypatch.setattr(
        image_module,
        "_get_dw_preprocessor_deps",
        lambda: (_FakeDWposeDetector, _FakeTorchModule, np),
    )

    result = dw_preprocessor(_FakeInputTensor())

    assert len(stacked) == batch_size


def test_dw_preprocessor_passes_flags_to_detector(monkeypatch: Any) -> None:
    """Verify detect_hand/body/face and resolution are forwarded to the detector."""
    import numpy as np

    calls: dict[str, Any] = {}

    class _FakeTensorItem:
        def cpu(self) -> "_FakeTensorItem":
            return self

        def numpy(self) -> Any:
            return np.zeros((4, 4, 3), dtype=np.float32)

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 4, 4, 3)

        def __getitem__(self, idx: int) -> _FakeTensorItem:
            return _FakeTensorItem()

    class _FakeDWposeDetector:
        def __call__(
            self,
            image: Any,
            *,
            detect_resolution: int,
            image_resolution: int,
            include_body: bool,
            include_hand: bool,
            include_face: bool,
            return_pil: bool,
        ) -> Any:
            calls["detect_resolution"] = detect_resolution
            calls["image_resolution"] = image_resolution
            calls["include_body"] = include_body
            calls["include_hand"] = include_hand
            calls["include_face"] = include_face
            calls["return_pil"] = return_pil
            return np.zeros((4, 4, 3), dtype=np.uint8)

    class _FakeTorchModule:
        @staticmethod
        def from_numpy(arr: Any) -> Any:
            class _T:
                def float(self) -> "_T":
                    return self

                def __truediv__(self, val: float) -> "_T":
                    return self

            return _T()

        @staticmethod
        def stack(frames: list[Any], dim: int) -> object:
            return object()

    monkeypatch.setattr(
        image_module,
        "_get_dw_preprocessor_deps",
        lambda: (_FakeDWposeDetector, _FakeTorchModule, np),
    )

    dw_preprocessor(
        _FakeInputTensor(),
        detect_hand=False,
        detect_body=True,
        detect_face=True,
        resolution=256,
    )

    assert calls["detect_resolution"] == 256
    assert calls["image_resolution"] == 256
    assert calls["include_body"] is True
    assert calls["include_hand"] is False
    assert calls["include_face"] is True
    assert calls["return_pil"] is False


def test_dw_preprocessor_lazy_import_no_top_level_controlnet_aux() -> None:
    """AC03: controlnet_aux is not imported at module top level."""
    import ast
    import pathlib

    source = pathlib.Path(image_module.__file__).read_text()
    tree = ast.parse(source)
    top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for imp in top_level_imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                assert "controlnet_aux" not in alias.name, (
                    "controlnet_aux must not be imported at module top level"
                )
        elif isinstance(imp, ast.ImportFrom) and imp.module:
            assert "controlnet_aux" not in imp.module, (
                "controlnet_aux must not be imported at module top level"
            )


# ---------------------------------------------------------------------------
# image_resize_kj tests (US-003-AC01)
# ---------------------------------------------------------------------------


def test_image_resize_kj_is_callable() -> None:
    assert callable(image_resize_kj)


def test_image_resize_kj_signature_matches_contract() -> None:
    signature = inspect.signature(image_resize_kj)
    assert str(signature) == (
        "(image: 'Any', width: 'int', height: 'int', upscale_method: 'str' = 'lanczos', "
        "keep_proportion: 'str' = 'crop', pad_color: 'str' = '0, 0, 0', "
        "crop_position: 'str' = 'center', divisible_by: 'int' = 2, "
        "device: 'str' = 'cpu') -> 'tuple[Any, int, int]'"
    )


def test_image_resize_kj_in_dunder_all() -> None:
    assert "image_resize_kj" in image_module.__all__


def test_image_resize_kj_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_resize_kj")


def test_image_resize_kj_stretch_calls_common_upscale_with_target_dims(
    monkeypatch: Any,
) -> None:
    """AC01: stretch mode passes width/height directly to common_upscale."""
    calls: dict[str, Any] = {}

    class _FakeOut:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 256, 512, 3)

    class _FakeMovedTensor:
        def movedim(self, src: int, dst: int) -> _FakeOut:
            return _FakeOut()

    class _FakeInputMovedTensor:
        def movedim(self, src: int, dst: int) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 64, 128, 3)

        def narrow(self, dim: int, start: int, length: int) -> "_FakeInputTensor":
            return self

        def movedim(self, src: int, dst: int) -> _FakeInputMovedTensor:
            return _FakeInputMovedTensor()

    class _FakeComfyUtils:
        @staticmethod
        def common_upscale(
            samples: Any, width: int, height: int, upscale_method: str, crop: str
        ) -> _FakeMovedTensor:
            calls["width"] = width
            calls["height"] = height
            calls["upscale_method"] = upscale_method
            calls["crop"] = crop
            return _FakeMovedTensor()

    monkeypatch.setattr(image_module, "_get_comfy_utils", lambda: _FakeComfyUtils)
    monkeypatch.setattr(image_module, "_get_torch_module", lambda: None)

    image_resize_kj(
        _FakeInputTensor(),
        width=512,
        height=256,
        upscale_method="lanczos",
        keep_proportion="stretch",
        divisible_by=1,
    )

    assert calls["width"] == 512
    assert calls["height"] == 256
    assert calls["upscale_method"] == "lanczos"
    assert calls["crop"] == "disabled"


def test_image_resize_kj_crop_mode_crops_before_upscale(monkeypatch: Any) -> None:
    """AC01: crop mode performs narrow() before calling common_upscale."""
    narrow_calls: list[tuple[int, int, int]] = []

    class _FakeOut:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 100, 100, 3)

    class _FakeMovedTensor:
        def movedim(self, src: int, dst: int) -> _FakeOut:
            return _FakeOut()

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            # 1x100x200x3 — wider than target, so width will be cropped
            return (1, 100, 200, 3)

        def narrow(self, dim: int, start: int, length: int) -> "_FakeInputTensor":
            narrow_calls.append((dim, start, length))
            return self

        def movedim(self, src: int, dst: int) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    class _FakeComfyUtils:
        @staticmethod
        def common_upscale(
            samples: Any, width: int, height: int, upscale_method: str, crop: str
        ) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    monkeypatch.setattr(image_module, "_get_comfy_utils", lambda: _FakeComfyUtils)
    monkeypatch.setattr(image_module, "_get_torch_module", lambda: None)

    image_resize_kj(
        _FakeInputTensor(),
        width=100,
        height=100,
        keep_proportion="crop",
        crop_position="center",
        divisible_by=1,
    )

    # narrow should have been called to crop width and/or height
    assert len(narrow_calls) >= 1


def test_image_resize_kj_returns_tuple_with_int_dims(monkeypatch: Any) -> None:
    """AC01: return value is (IMAGE, int, int)."""

    class _FakeMovedTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 3, 64, 128)

        def movedim(self, src: int, dst: int) -> "_FakeMovedTensor":
            return self

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 64, 128, 3)

        def narrow(self, dim: int, start: int, length: int) -> "_FakeInputTensor":
            return self

        def movedim(self, src: int, dst: int) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    class _FakeComfyUtils:
        @staticmethod
        def common_upscale(
            samples: Any, width: int, height: int, upscale_method: str, crop: str
        ) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    monkeypatch.setattr(image_module, "_get_comfy_utils", lambda: _FakeComfyUtils)
    monkeypatch.setattr(image_module, "_get_torch_module", lambda: None)

    result = image_resize_kj(
        _FakeInputTensor(),
        width=128,
        height=64,
        keep_proportion="stretch",
        divisible_by=1,
    )

    assert isinstance(result, tuple) and len(result) == 3
    _img, w, h = result
    assert isinstance(w, int)
    assert isinstance(h, int)


def test_image_resize_kj_divisible_by_rounds_down(monkeypatch: Any) -> None:
    """AC01: divisible_by truncates the target dimensions."""
    calls: dict[str, Any] = {}

    class _FakeOut:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 256, 512, 3)

    class _FakeMovedTensor:
        def movedim(self, src: int, dst: int) -> _FakeOut:
            return _FakeOut()

    class _FakeInputTensor:
        @property
        def shape(self) -> tuple[int, int, int, int]:
            return (1, 64, 64, 3)

        def narrow(self, dim: int, start: int, length: int) -> "_FakeInputTensor":
            return self

        def movedim(self, src: int, dst: int) -> _FakeMovedTensor:
            return _FakeMovedTensor()

    class _FakeComfyUtils:
        @staticmethod
        def common_upscale(
            samples: Any, width: int, height: int, upscale_method: str, crop: str
        ) -> _FakeMovedTensor:
            calls["width"] = width
            calls["height"] = height
            return _FakeMovedTensor()

    monkeypatch.setattr(image_module, "_get_comfy_utils", lambda: _FakeComfyUtils)
    monkeypatch.setattr(image_module, "_get_torch_module", lambda: None)

    image_resize_kj(
        _FakeInputTensor(),
        width=513,
        height=257,
        keep_proportion="stretch",
        divisible_by=8,
    )

    # 513 - (513 % 8) = 513 - 1 = 512; 257 - (257 % 8) = 257 - 1 = 256
    assert calls["width"] == 512
    assert calls["height"] == 256


def test_image_resize_kj_lazy_import_no_top_level_torch_or_comfy() -> None:
    """AC03: image_resize_kj follows the lazy-import pattern."""
    import ast
    import pathlib

    source = pathlib.Path(image_module.__file__).read_text()
    tree = ast.parse(source)
    top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    forbidden = {"torch", "comfy"}
    for imp in top_level_imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                assert alias.name.split(".")[0] not in forbidden, (
                    f"Forbidden top-level import: {alias.name}"
                )
        elif isinstance(imp, ast.ImportFrom) and imp.module:
            assert imp.module.split(".")[0] not in forbidden, (
                f"Forbidden top-level import: {imp.module}"
            )


# ---------------------------------------------------------------------------
# image_batch_extend_with_overlap tests (US-003-AC02)
# ---------------------------------------------------------------------------


def test_image_batch_extend_with_overlap_is_callable() -> None:
    assert callable(image_batch_extend_with_overlap)


def test_image_batch_extend_with_overlap_signature_matches_contract() -> None:
    signature = inspect.signature(image_batch_extend_with_overlap)
    assert str(signature) == (
        "(source_images: 'Any', new_images: 'Any' = None, overlap: 'int' = 13, "
        "overlap_side: 'str' = 'source', overlap_mode: 'str' = 'filmic_crossfade') -> 'Any'"
    )


def test_image_batch_extend_with_overlap_in_dunder_all() -> None:
    assert "image_batch_extend_with_overlap" in image_module.__all__


def test_image_batch_extend_with_overlap_not_re_exported_from_package_root() -> None:
    assert not hasattr(comfy_diffusion, "image_batch_extend_with_overlap")


def test_image_batch_extend_with_overlap_no_new_images_returns_zeros(
    monkeypatch: Any,
) -> None:
    """AC02: when new_images is None, returns a placeholder zero tensor."""
    import torch

    monkeypatch.setattr(image_module, "_get_torch_module", lambda: torch)
    source = torch.zeros((20, 64, 64, 3))
    result = image_batch_extend_with_overlap(source, new_images=None, overlap=5)
    assert result.shape == (1, 64, 64, 3)
    assert float(result.sum()) == 0.0


def test_image_batch_extend_with_overlap_overlap_exceeds_source_returns_source(
    monkeypatch: Any,
) -> None:
    """AC02: when overlap > len(source_images), source_images is returned unchanged."""
    import torch

    source = torch.ones((3, 8, 8, 3))
    result = image_batch_extend_with_overlap(source, new_images=None, overlap=10)
    assert result is source


def test_image_batch_extend_with_overlap_shape_mismatch_raises(
    monkeypatch: Any,
) -> None:
    """AC02: spatial dim mismatch raises ValueError."""
    import torch

    source = torch.zeros((10, 64, 64, 3))
    new = torch.zeros((5, 32, 32, 3))
    with pytest.raises(ValueError, match="same spatial dimensions"):
        image_batch_extend_with_overlap(source, new_images=new, overlap=3)


def test_image_batch_extend_with_overlap_filmic_crossfade_output_shape(
    monkeypatch: Any,
) -> None:
    """AC02: filmic_crossfade extended output has correct frame count."""
    import torch

    monkeypatch.setattr(image_module, "_get_torch_module", lambda: torch)
    B_src, B_new, overlap = 20, 15, 5
    source = torch.rand((B_src, 8, 8, 3))
    new = torch.rand((B_new, 8, 8, 3))
    result = image_batch_extend_with_overlap(
        source, new_images=new, overlap=overlap, overlap_mode="filmic_crossfade"
    )
    # prefix = source[:-overlap] = 15 frames
    # blended = overlap = 5 frames
    # suffix = new[overlap:] = 10 frames
    # total = 15 + 5 + 10 = 30
    expected_frames = (B_src - overlap) + overlap + (B_new - overlap)
    assert result.shape[0] == expected_frames
    assert result.shape[1:] == (8, 8, 3)


def test_image_batch_extend_with_overlap_linear_blend_midpoint(
    monkeypatch: Any,
) -> None:
    """AC02: linear_blend midpoint frame is 50% mix of src and dst."""
    import torch

    monkeypatch.setattr(image_module, "_get_torch_module", lambda: torch)
    overlap = 2  # odd overlap not needed; linspace[1:-1] gives 2 values for overlap=2
    source = torch.zeros((5, 4, 4, 3))  # source frames are 0
    new_imgs = torch.ones((5, 4, 4, 3))  # new frames are 1

    result = image_batch_extend_with_overlap(
        source, new_images=new_imgs, overlap=overlap,
        overlap_side="source", overlap_mode="linear_blend"
    )
    # alpha = linspace(0,1, overlap+2)[1:-1] → for overlap=2: [0.333..., 0.666...]
    # blended[0] = 1-0.333... * 0 + 0.333... * 1 ≈ 0.333
    prefix_len = len(source) - overlap  # = 3
    blended_frame_0 = result[prefix_len]  # first blended frame
    expected_alpha = torch.linspace(0, 1, overlap + 2)[1:-1][0].item()
    assert abs(float(blended_frame_0.mean()) - expected_alpha) < 1e-5


def test_image_batch_extend_with_overlap_cut_mode_source_side(
    monkeypatch: Any,
) -> None:
    """AC02: cut mode with overlap_side=source drops source tail."""
    import torch

    monkeypatch.setattr(image_module, "_get_torch_module", lambda: torch)
    overlap = 3
    source = torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1).expand(-1, 4, 4, 3)
    new = torch.arange(100, 110, dtype=torch.float32).view(10, 1, 1, 1).expand(-1, 4, 4, 3)
    result = image_batch_extend_with_overlap(
        source, new_images=new, overlap=overlap,
        overlap_side="source", overlap_mode="cut"
    )
    # expect source[:-3] (7 frames) + new (10 frames) = 17 frames
    assert result.shape[0] == 10 - overlap + 10


def test_image_batch_extend_with_overlap_lazy_import_no_top_level_torch() -> None:
    """AC03: image_batch_extend_with_overlap follows the lazy-import pattern."""
    import ast
    import pathlib

    source = pathlib.Path(image_module.__file__).read_text()
    tree = ast.parse(source)
    top_level_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    forbidden = {"torch", "comfy"}
    for imp in top_level_imports:
        if isinstance(imp, ast.Import):
            for alias in imp.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(imp, ast.ImportFrom) and imp.module:
            assert imp.module.split(".")[0] not in forbidden
