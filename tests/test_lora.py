"""Tests for LoRA helpers."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import comfy_diffusion
import comfy_diffusion.lora as lora_module
from comfy_diffusion import apply_lora
from comfy_diffusion.lora import apply_ic_lora_model_only


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


def test_lora_module_exports_expected_helpers() -> None:
    assert lora_module.__all__ == ["apply_lora", "apply_ic_lora_model_only"]


def test_apply_lora_signature_matches_contract() -> None:
    signature = inspect.signature(apply_lora)
    assert str(signature) == (
        "(model: 'Any', clip: 'Any', path: 'str | Path', "
        "strength_model: 'float', strength_clip: 'float') -> 'tuple[Any, Any]'"
    )


def test_apply_ic_lora_model_only_signature_matches_contract() -> None:
    signature = inspect.signature(apply_ic_lora_model_only)
    assert str(signature) == (
        "(model: 'Any', path: 'str | Path', strength_model: 'float' = 1.0) "
        "-> 'tuple[Any, float]'"
    )


def test_apply_lora_returns_patched_model_and_clip_tuple(
    monkeypatch: Any,
) -> None:
    model = object()
    clip = object()
    patched_model = object()
    patched_clip = object()

    loaded_lora = object()
    calls: dict[str, Any] = {}

    def fake_load_torch_file(path: str, *, safe_load: bool) -> object:
        calls["load_torch_file"] = {"path": path, "safe_load": safe_load}
        return loaded_lora

    def fake_load_lora_for_models(
        model_arg: object,
        clip_arg: object,
        lora_arg: object,
        strength_model_arg: float,
        strength_clip_arg: float,
    ) -> tuple[object, object]:
        calls["load_lora_for_models"] = {
            "model": model_arg,
            "clip": clip_arg,
            "lora": lora_arg,
            "strength_model": strength_model_arg,
            "strength_clip": strength_clip_arg,
        }
        return (patched_model, patched_clip)

    import types

    comfy_module = types.ModuleType("comfy")
    comfy_utils_module = types.ModuleType("comfy.utils")
    comfy_sd_module = types.ModuleType("comfy.sd")

    comfy_utils_module.load_torch_file = fake_load_torch_file
    comfy_sd_module.load_lora_for_models = fake_load_lora_for_models

    comfy_module.utils = comfy_utils_module
    comfy_module.sd = comfy_sd_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)
    result = lora_module.apply_lora(model, clip, "/tmp/my_lora.safetensors", 0.8, 0.8)

    assert isinstance(result, tuple)
    assert result == (patched_model, patched_clip)
    assert result[0] is not model
    assert result[1] is not clip
    assert calls["load_torch_file"] == {"path": "/tmp/my_lora.safetensors", "safe_load": True}
    assert calls["load_lora_for_models"] == {
        "model": model,
        "clip": clip,
        "lora": loaded_lora,
        "strength_model": 0.8,
        "strength_clip": 0.8,
    }


def test_apply_ic_lora_model_only_returns_model_and_metadata_downscale(
    monkeypatch: Any,
) -> None:
    model = object()
    patched_model = object()
    loaded_lora = object()
    calls: dict[str, Any] = {}

    def fake_load_torch_file(
        path: str,
        *,
        safe_load: bool,
        return_metadata: bool,
    ) -> tuple[object, dict[str, str]]:
        calls["load_torch_file"] = {
            "path": path,
            "safe_load": safe_load,
            "return_metadata": return_metadata,
        }
        return loaded_lora, {"reference_downscale_factor": "2"}

    def fake_load_lora_for_models(
        model_arg: object,
        clip_arg: object | None,
        lora_arg: object,
        strength_model_arg: float,
        strength_clip_arg: float,
    ) -> tuple[object, None]:
        calls["load_lora_for_models"] = {
            "model": model_arg,
            "clip": clip_arg,
            "lora": lora_arg,
            "strength_model": strength_model_arg,
            "strength_clip": strength_clip_arg,
        }
        return patched_model, None

    import types

    comfy_module = types.ModuleType("comfy")
    comfy_utils_module = types.ModuleType("comfy.utils")
    comfy_sd_module = types.ModuleType("comfy.sd")

    comfy_utils_module.load_torch_file = fake_load_torch_file
    comfy_sd_module.load_lora_for_models = fake_load_lora_for_models
    comfy_module.utils = comfy_utils_module
    comfy_module.sd = comfy_sd_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)

    result = lora_module.apply_ic_lora_model_only(model, "/tmp/ic.safetensors", 0.8)

    assert result == (patched_model, 2.0)
    assert calls["load_torch_file"] == {
        "path": "/tmp/ic.safetensors",
        "safe_load": True,
        "return_metadata": True,
    }
    assert calls["load_lora_for_models"] == {
        "model": model,
        "clip": None,
        "lora": loaded_lora,
        "strength_model": 0.8,
        "strength_clip": 0.0,
    }


def test_apply_ic_lora_model_only_defaults_downscale_to_one(monkeypatch: Any) -> None:
    model = object()
    loaded_lora = object()

    def fake_load_torch_file(
        path: str,
        *,
        safe_load: bool,
        return_metadata: bool,
    ) -> tuple[object, dict[str, str]]:
        return loaded_lora, {}

    import types

    comfy_module = types.ModuleType("comfy")
    comfy_utils_module = types.ModuleType("comfy.utils")
    comfy_sd_module = types.ModuleType("comfy.sd")

    comfy_utils_module.load_torch_file = fake_load_torch_file
    comfy_module.utils = comfy_utils_module
    comfy_module.sd = comfy_sd_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)

    assert lora_module.apply_ic_lora_model_only(model, "/tmp/ic.safetensors", 0.0) == (
        model,
        1.0,
    )


def test_apply_lora_is_re_exported_from_package_root() -> None:
    assert callable(apply_lora)
    assert comfy_diffusion.apply_lora is apply_lora
    assert "apply_lora" in comfy_diffusion.__all__


def test_apply_lora_supports_chaining_without_additional_api(
    monkeypatch: Any,
) -> None:
    base_model = object()
    base_clip = object()
    first_patched_model = object()
    first_patched_clip = object()
    second_patched_model = object()
    second_patched_clip = object()

    loaded_loras: list[dict[str, Any]] = []
    patch_calls: list[dict[str, Any]] = []

    def fake_load_torch_file(path: str, *, safe_load: bool) -> dict[str, Any]:
        loaded = {"path": path, "safe_load": safe_load}
        loaded_loras.append(loaded)
        return loaded

    def fake_load_lora_for_models(
        model_arg: object,
        clip_arg: object,
        lora_arg: dict[str, Any],
        strength_model_arg: float,
        strength_clip_arg: float,
    ) -> tuple[object, object]:
        patch_calls.append(
            {
                "model": model_arg,
                "clip": clip_arg,
                "lora": lora_arg,
                "strength_model": strength_model_arg,
                "strength_clip": strength_clip_arg,
            }
        )
        if len(patch_calls) == 1:
            return (first_patched_model, first_patched_clip)
        return (second_patched_model, second_patched_clip)

    import types

    comfy_module = types.ModuleType("comfy")
    comfy_utils_module = types.ModuleType("comfy.utils")
    comfy_sd_module = types.ModuleType("comfy.sd")

    comfy_utils_module.load_torch_file = fake_load_torch_file
    comfy_sd_module.load_lora_for_models = fake_load_lora_for_models

    comfy_module.utils = comfy_utils_module
    comfy_module.sd = comfy_sd_module

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd_module)

    first_model, first_clip = lora_module.apply_lora(
        base_model,
        base_clip,
        "/tmp/lora_a.safetensors",
        0.7,
        0.5,
    )
    second_model, second_clip = lora_module.apply_lora(
        first_model,
        first_clip,
        "/tmp/lora_b.safetensors",
        0.3,
        0.2,
    )

    # AC01: chaining two calls yields a double-patched model+CLIP pair.
    assert (first_model, first_clip) == (first_patched_model, first_patched_clip)
    assert (second_model, second_clip) == (second_patched_model, second_patched_clip)
    assert patch_calls[0] == {
        "model": base_model,
        "clip": base_clip,
        "lora": loaded_loras[0],
        "strength_model": 0.7,
        "strength_clip": 0.5,
    }
    assert patch_calls[1] == {
        "model": first_model,
        "clip": first_clip,
        "lora": loaded_loras[1],
        "strength_model": 0.3,
        "strength_clip": 0.2,
    }

    # AC02: intermediate result remains independent from later patched output.
    assert first_model is first_patched_model
    assert first_clip is first_patched_clip
    assert second_model is not first_model
    assert second_clip is not first_clip

    # AC03: chaining uses only apply_lora; no new multi-LoRA API was added.
    assert not hasattr(lora_module, "apply_loras")
    assert not hasattr(comfy_diffusion, "apply_loras")


def test_import_comfy_diffusion_lora_has_no_heavy_import_side_effects() -> None:
    result = _run_python(
        "import json\n"
        "import sys\n"
        "import comfy_diffusion\n"
        "baseline_modules = set(sys.modules)\n"
        "from comfy_diffusion.lora import apply_lora\n"
        "post_modules = set(sys.modules)\n"
        "new_modules = sorted(post_modules - baseline_modules)\n"
        "payload = {\n"
        "  'func_name': apply_lora.__name__,\n"
        "  'comfy_sd_loaded': 'comfy.sd' in sys.modules,\n"
        "  'comfy_utils_loaded': 'comfy.utils' in sys.modules,\n"
        "  'new_modules': new_modules,\n"
        "}\n"
        "print(json.dumps(payload))\n"
    )

    payload = json.loads(result.stdout)
    assert payload["func_name"] == "apply_lora"
    assert payload["comfy_sd_loaded"] is False
    assert payload["comfy_utils_loaded"] is False
    heavy = [
        module
        for module in payload["new_modules"]
        if module.startswith(("comfy.sd", "comfy.utils"))
    ]
    assert heavy == [], f"Unexpected heavy modules loaded on import: {heavy}"
