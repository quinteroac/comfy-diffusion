"""Tests for experimental raw ComfyUI node access."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest

import comfy_diffusion.nodes as nodes_module


class _FakeV1Node:
    FUNCTION = "execute"
    CATEGORY = "sampling"
    RETURN_TYPES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {"required": {"seed": ("INT", {"default": 1})}}

    def execute(self, **inputs: object) -> tuple[dict[str, object]]:
        return ({"inputs": inputs},)


class _FakeVAENode(_FakeV1Node):
    CATEGORY = "latent"
    RETURN_TYPES = ("IMAGE",)


class _FakeCLIPNode(_FakeV1Node):
    CATEGORY = "conditioning"
    RETURN_TYPES = ("CONDITIONING",)


class _FakeExtraNode(_FakeV1Node):
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE",)
    RELATIVE_PYTHON_MODULE = "comfy_extras.nodes_canny"


class _FakeSchema:
    node_id = "OpenAIDalle3"
    display_name = "OpenAI DALL-E 3"
    category = "api/image"
    inputs = ["prompt"]
    is_api_node = True


class _FakeAPINode:
    FUNCTION = "EXECUTE_NORMALIZED"
    RETURN_TYPES = ("IMAGE",)
    RELATIVE_PYTHON_MODULE = "comfy_api_nodes.nodes_openai"
    prepared_hidden_inputs: dict[str, object] | None = None

    @classmethod
    def GET_SCHEMA(cls) -> _FakeSchema:
        return _FakeSchema()

    @classmethod
    def PREPARE_CLASS_CLONE(cls, v3_data: dict[str, object]) -> type[_FakeAPINode]:
        hidden_inputs = v3_data["hidden_inputs"]
        assert isinstance(hidden_inputs, dict)
        cls.prepared_hidden_inputs = hidden_inputs
        return cls

    def EXECUTE_NORMALIZED(self, **inputs: object) -> tuple[dict[str, object]]:
        return ({"api_inputs": inputs, "hidden": self.prepared_hidden_inputs},)


class _FakeCustomNode(_FakeV1Node):
    CATEGORY = "custom"
    RETURN_TYPES = ("CUSTOM",)
    RELATIVE_PYTHON_MODULE = "custom_nodes.demo_custom"


def _fake_comfy_nodes_module() -> types.SimpleNamespace:
    async def init_builtin_extra_nodes() -> list[str]:
        return []

    async def init_builtin_api_nodes() -> list[str]:
        return []

    async def init_external_custom_nodes() -> None:
        raise AssertionError("external custom nodes must not be loaded")

    async def load_custom_node(module_path: str, module_parent: str = "custom_nodes") -> bool:
        assert module_parent == "custom_nodes"
        fake_nodes.NODE_CLASS_MAPPINGS["DemoCustom"] = _FakeCustomNode
        fake_nodes.LOADED_MODULE_DIRS["demo_custom"] = module_path
        return True

    fake_nodes = types.SimpleNamespace()
    fake_nodes.NODE_CLASS_MAPPINGS = {
        "CLIPTextEncode": _FakeCLIPNode,
        "Canny": _FakeExtraNode,
        "KSampler": _FakeV1Node,
        "OpenAIDalle3": _FakeAPINode,
        "VAEDecode": _FakeVAENode,
    }
    fake_nodes.LOADED_MODULE_DIRS = {}
    fake_nodes.init_builtin_extra_nodes = init_builtin_extra_nodes
    fake_nodes.init_builtin_api_nodes = init_builtin_api_nodes
    fake_nodes.init_external_custom_nodes = init_external_custom_nodes
    fake_nodes.load_custom_node = load_custom_node
    return fake_nodes

@pytest.fixture(autouse=True)
def clear_node_registry_cache() -> None:
    nodes_module._clear_registry_cache()
    _FakeAPINode.prepared_hidden_inputs = None


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    fake_nodes = _fake_comfy_nodes_module()
    monkeypatch.setattr(nodes_module, "_import_comfy_nodes_module", lambda: fake_nodes)
    return fake_nodes


def test_importing_nodes_module_does_not_import_comfyui_or_torch() -> None:
    sys.modules.pop("nodes", None)
    sys.modules.pop("comfy.model_management", None)

    module = importlib.reload(nodes_module)

    assert module.NodeInfo.__name__ == "NodeInfo"
    assert "nodes" not in sys.modules
    assert "comfy.model_management" not in sys.modules


def test_list_nodes_returns_known_core_and_extra_nodes(
    fake_registry: types.SimpleNamespace,
) -> None:
    nodes = nodes_module.list_nodes()

    assert list(nodes) == sorted(nodes)
    assert {"KSampler", "VAEDecode", "CLIPTextEncode", "Canny"}.issubset(nodes)
    assert nodes["Canny"].module == "comfy_extras.nodes_canny"


def test_default_list_nodes_excludes_api_nodes(fake_registry: types.SimpleNamespace) -> None:
    nodes = nodes_module.list_nodes()

    assert "OpenAIDalle3" not in nodes


def test_include_api_list_nodes_includes_api_nodes(fake_registry: types.SimpleNamespace) -> None:
    nodes = nodes_module.list_nodes(include_api=True)

    assert nodes["OpenAIDalle3"].is_api_node is True
    assert nodes["OpenAIDalle3"].display_name == "OpenAI DALL-E 3"
    assert _FakeAPINode.prepared_hidden_inputs is None


def test_get_node_info_returns_stable_metadata(fake_registry: types.SimpleNamespace) -> None:
    info = nodes_module.get_node_info("VAEDecode")

    assert info.node_id == "VAEDecode"
    assert info.class_name == "_FakeVAENode"
    assert info.category == "latent"
    assert info.return_types == ("IMAGE",)
    assert info.function_name == "execute"
    assert info.is_custom_node is False
    assert info.source_path is None


def test_get_node_returns_underlying_class(fake_registry: types.SimpleNamespace) -> None:
    assert nodes_module.get_node("KSampler") is _FakeV1Node


def test_run_node_calls_node_function(fake_registry: types.SimpleNamespace) -> None:
    result = nodes_module.run_node("KSampler", seed=123)

    assert result == ({"inputs": {"seed": 123}},)


def test_unknown_node_id_raises_key_error(fake_registry: types.SimpleNamespace) -> None:
    with pytest.raises(KeyError, match="Unknown ComfyUI node"):
        nodes_module.get_node("MissingNode")


def test_list_nodes_loads_explicit_custom_node_path(
    fake_registry: types.SimpleNamespace,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    custom_dir = tmp_path / "demo_custom"
    custom_dir.mkdir()
    (custom_dir / "__init__.py").write_text("", encoding="utf-8")

    nodes = nodes_module.list_nodes(custom_node_paths=[custom_dir])

    assert "DemoCustom" in nodes
    assert nodes["DemoCustom"].is_custom_node is True
    assert nodes["DemoCustom"].source_path == str(custom_dir.resolve())


def test_missing_custom_node_path_raises_value_error(
    fake_registry: types.SimpleNamespace,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="does not exist"):
        nodes_module.list_nodes(custom_node_paths=[tmp_path / "missing"])


def test_unsupported_custom_node_path_raises_value_error(
    fake_registry: types.SimpleNamespace,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    bad_file = tmp_path / "README.md"
    bad_file.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a .py file"):
        nodes_module.list_nodes(custom_node_paths=[bad_file])


def test_run_node_can_execute_custom_node(
    fake_registry: types.SimpleNamespace,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    custom_file = tmp_path / "demo_custom.py"
    custom_file.write_text("", encoding="utf-8")

    result = nodes_module.run_node("DemoCustom", custom_node_paths=[custom_file], seed=7)

    assert result == ({"inputs": {"seed": 7}},)


def test_custom_node_cache_key_includes_paths(
    fake_registry: types.SimpleNamespace,
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    nodes_module.list_nodes(custom_node_paths=[first])
    nodes_module.list_nodes(custom_node_paths=[second])

    assert len(nodes_module._REGISTRY_CACHE) == 2


def test_api_node_execution_without_auth_raises(fake_registry: types.SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="COMFY_ORG_API_KEY"):
        nodes_module.run_node("OpenAIDalle3", include_api=True, prompt="hello")


def test_api_node_auth_api_key_injects_hidden_inputs(
    fake_registry: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_urls: list[str] = []
    monkeypatch.setattr(nodes_module, "_set_api_base_url", base_urls.append)

    result = nodes_module.run_node(
        "OpenAIDalle3",
        include_api=True,
        api_auth=nodes_module.ApiNodeAuth(
            api_key="key-123",
            base_url="https://api.example.test",
            unique_id="node-1",
        ),
        prompt="hello",
    )

    assert result == (
        {
            "api_inputs": {"prompt": "hello"},
            "hidden": {
                "API_KEY_COMFY_ORG": "key-123",
                "AUTH_TOKEN_COMFY_ORG": None,
                "UNIQUE_ID": "node-1",
            },
        },
    )
    assert base_urls == ["https://api.example.test"]


def test_api_node_auth_token_injects_hidden_inputs(
    fake_registry: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nodes_module, "_set_api_base_url", lambda base_url: None)

    result = nodes_module.run_node(
        "OpenAIDalle3",
        include_api=True,
        api_auth=nodes_module.ApiNodeAuth(auth_token="token-123"),
        prompt="hello",
    )

    assert result[0]["hidden"]["AUTH_TOKEN_COMFY_ORG"] == "token-123"  # type: ignore[index]
    assert result[0]["hidden"]["API_KEY_COMFY_ORG"] is None  # type: ignore[index]
    assert result[0]["hidden"]["UNIQUE_ID"] == "OpenAIDalle3"  # type: ignore[index]


def test_api_node_auth_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMFY_ORG_API_KEY", " env-key ")
    monkeypatch.setenv("COMFY_ORG_AUTH_TOKEN", " env-token ")
    monkeypatch.setenv("COMFY_API_BASE", "https://api.env.test")

    auth = nodes_module.ApiNodeAuth.from_env()

    assert auth.api_key == "env-key"
    assert auth.auth_token == "env-token"
    assert auth.base_url == "https://api.env.test"


def test_api_node_execution_can_use_env_auth(
    fake_registry: types.SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_urls: list[str] = []
    monkeypatch.setenv("COMFY_ORG_API_KEY", "env-key")
    monkeypatch.setenv("COMFY_API_BASE", "https://api.env.test")
    monkeypatch.setattr(nodes_module, "_set_api_base_url", base_urls.append)

    result = nodes_module.run_node("OpenAIDalle3", include_api=True, prompt="hello")

    assert result[0]["hidden"]["API_KEY_COMFY_ORG"] == "env-key"  # type: ignore[index]
    assert base_urls == ["https://api.env.test"]
