"""Experimental raw access to vendored ComfyUI nodes.

This module is intentionally lazy: importing it must not import torch or ComfyUI.
Use the curated comfy_diffusion modules for stable application code; this module
is an escape hatch for advanced integrations that need direct node access.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_DEFAULT_COMFY_API_BASE = "https://api.comfy.org"
_HIDDEN_API_KEY_COMFY_ORG = "API_KEY_COMFY_ORG"
_HIDDEN_AUTH_TOKEN_COMFY_ORG = "AUTH_TOKEN_COMFY_ORG"
_HIDDEN_UNIQUE_ID = "UNIQUE_ID"


@dataclass(frozen=True)
class ApiNodeAuth:
    """Authentication for ComfyUI API nodes that call the Comfy.org proxy."""

    api_key: str | None = None
    auth_token: str | None = None
    base_url: str = _DEFAULT_COMFY_API_BASE
    unique_id: str | None = None

    @classmethod
    def from_env(cls) -> ApiNodeAuth:
        """Create API node auth from Comfy.org environment variables."""
        return cls(
            api_key=_env_value("COMFY_ORG_API_KEY"),
            auth_token=_env_value("COMFY_ORG_AUTH_TOKEN"),
            base_url=_env_value("COMFY_API_BASE") or _DEFAULT_COMFY_API_BASE,
        )

    def has_credentials(self) -> bool:
        """Return True when an API key or auth token is available."""
        return bool(self.api_key or self.auth_token)


@dataclass(frozen=True)
class NodeInfo:
    """Metadata for a ComfyUI node exposed through the raw node registry."""

    node_id: str
    class_name: str
    display_name: str | None
    module: str
    category: str | None
    input_types: Any
    return_types: tuple[Any, ...]
    function_name: str | None
    is_api_node: bool
    is_custom_node: bool
    source_path: str | None


@dataclass(frozen=True)
class _NodeRecord:
    info: NodeInfo
    node_class: type[Any]


_RegistryKey = tuple[bool, tuple[str, ...]]

_REGISTRY_CACHE: dict[_RegistryKey, dict[str, _NodeRecord]] = {}
_IMPORT_FAILURES: dict[_RegistryKey, tuple[str, ...]] = {}


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _describe_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _run_coro(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive thread bridge
            error["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in error:
        raise error["error"]
    return result.get("value")


def _prepare_cpu_safe_import() -> None:
    """Keep node discovery usable on CPU-only installs when ComfyUI is not loaded yet."""
    import sys

    if "comfy.model_management" in sys.modules:
        return

    try:
        import torch  # noqa: PLC0415
    except Exception:
        return

    if torch.cuda.is_available():
        return

    try:
        cli_args = importlib.import_module("comfy.cli_args")
        cli_args.args.cpu = True
    except Exception:
        return


def _import_comfy_nodes_module() -> Any:
    from comfy_diffusion._runtime import ensure_comfyui_on_path

    ensure_comfyui_on_path()
    _prepare_cpu_safe_import()
    try:
        return importlib.import_module("nodes")
    except Exception as exc:
        raise RuntimeError(
            "Could not import ComfyUI nodes. Install the required comfyui extra and "
            "ensure torch/torchvision match the selected runtime. "
            f"Cause: {_describe_exception(exc)}"
        ) from exc


def _normalize_custom_node_paths(
    custom_node_paths: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    if custom_node_paths is None:
        return ()

    normalized: list[Path] = []
    for raw_path in custom_node_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Custom node path does not exist: {path}")
        if path.is_file() and path.suffix != ".py":
            raise ValueError(f"Custom node file must be a .py file: {path}")
        if path.is_dir() and not (path / "__init__.py").is_file():
            raise ValueError(f"Custom node directory must contain __init__.py: {path}")
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"Custom node path must be a file or directory: {path}")
        normalized.append(path)
    return tuple(normalized)


def _initialize_nodes(
    comfy_nodes: Any,
    *,
    include_api: bool,
    custom_node_paths: tuple[Path, ...],
) -> tuple[str, ...]:
    failures: list[str] = []

    init_extras = getattr(comfy_nodes, "init_builtin_extra_nodes", None)
    if callable(init_extras):
        try:
            failed = _run_coro(init_extras())
            if failed:
                failures.extend(str(item) for item in failed)
        except Exception as exc:
            failures.append(f"comfy_extras: {_describe_exception(exc)}")

    if include_api:
        init_api = getattr(comfy_nodes, "init_builtin_api_nodes", None)
        if callable(init_api):
            try:
                failed = _run_coro(init_api())
                if failed:
                    failures.extend(str(item) for item in failed)
            except Exception as exc:
                failures.append(f"comfy_api_nodes: {_describe_exception(exc)}")

    if custom_node_paths:
        load_custom_node = getattr(comfy_nodes, "load_custom_node", None)
        if not callable(load_custom_node):
            raise RuntimeError("ComfyUI nodes module does not expose load_custom_node().")

        for path in custom_node_paths:
            try:
                loaded = _run_coro(load_custom_node(str(path), module_parent="custom_nodes"))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load custom node from {path}: {_describe_exception(exc)}"
                ) from exc
            if not loaded:
                raise RuntimeError(f"Could not load custom node from {path}.")

    return tuple(failures)


def _safe_input_types(node_class: type[Any], schema: Any | None) -> Any:
    if schema is not None and hasattr(schema, "inputs"):
        return getattr(schema, "inputs")

    input_types = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types):
        return None
    try:
        return input_types()
    except Exception as exc:
        return {"error": _describe_exception(exc)}


def _safe_schema(node_class: type[Any]) -> Any | None:
    get_schema = getattr(node_class, "GET_SCHEMA", None)
    if not callable(get_schema):
        return None
    try:
        return get_schema()
    except Exception:
        return None


def _tuple_attr(node_class: type[Any], name: str) -> tuple[Any, ...]:
    value = getattr(node_class, name, ())
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _node_module(node_class: type[Any]) -> str:
    relative_module = getattr(node_class, "RELATIVE_PYTHON_MODULE", None)
    if isinstance(relative_module, str) and relative_module:
        return relative_module
    return str(getattr(node_class, "__module__", ""))


def _custom_node_source_path(comfy_nodes: Any, module: str) -> str | None:
    if not module.startswith("custom_nodes."):
        return None
    module_name = module.rsplit(".", 1)[-1]
    loaded_dirs = getattr(comfy_nodes, "LOADED_MODULE_DIRS", {})
    if isinstance(loaded_dirs, dict):
        value = loaded_dirs.get(module_name)
        if isinstance(value, str):
            return value
    return None


def _node_info(node_id: str, node_class: type[Any], comfy_nodes: Any) -> NodeInfo:
    schema = _safe_schema(node_class)
    module = _node_module(node_class)
    function_name = getattr(node_class, "FUNCTION", None)

    display_name = None
    category = None
    is_api_node = module.startswith("comfy_api_nodes")
    is_custom_node = module.startswith("custom_nodes.")
    if schema is not None:
        display_name = getattr(schema, "display_name", None)
        category = getattr(schema, "category", None)
        is_api_node = bool(getattr(schema, "is_api_node", is_api_node))

    if display_name is None:
        display_name = getattr(node_class, "DISPLAY_NAME", None)
    if category is None:
        category = getattr(node_class, "CATEGORY", None)

    return NodeInfo(
        node_id=node_id,
        class_name=str(getattr(node_class, "__name__", type(node_class).__name__)),
        display_name=display_name,
        module=module,
        category=category,
        input_types=_safe_input_types(node_class, schema),
        return_types=_tuple_attr(node_class, "RETURN_TYPES"),
        function_name=str(function_name) if function_name is not None else None,
        is_api_node=is_api_node,
        is_custom_node=is_custom_node,
        source_path=_custom_node_source_path(comfy_nodes, module),
    )


def _load_registry(
    include_api: bool,
    custom_node_paths: tuple[Path, ...],
) -> dict[str, _NodeRecord]:
    comfy_nodes = _import_comfy_nodes_module()
    failures = _initialize_nodes(
        comfy_nodes,
        include_api=include_api,
        custom_node_paths=custom_node_paths,
    )
    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS")

    records: dict[str, _NodeRecord] = {}
    for node_id in sorted(mappings):
        node_class = mappings[node_id]
        info = _node_info(str(node_id), node_class, comfy_nodes)
        if info.is_api_node and not include_api:
            continue
        records[str(node_id)] = _NodeRecord(info=info, node_class=node_class)

    _IMPORT_FAILURES[(include_api, tuple(str(path) for path in custom_node_paths))] = failures
    return records


def _registry(
    include_api: bool,
    custom_node_paths: Sequence[str | Path] | None,
) -> dict[str, _NodeRecord]:
    normalized_paths = _normalize_custom_node_paths(custom_node_paths)
    key = (include_api, tuple(str(path) for path in normalized_paths))
    if key not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[key] = _load_registry(include_api, normalized_paths)
    return _REGISTRY_CACHE[key]


def _get_record(
    node_id: str,
    include_api: bool,
    custom_node_paths: Sequence[str | Path] | None,
) -> _NodeRecord:
    try:
        return _registry(include_api, custom_node_paths)[node_id]
    except KeyError as exc:
        raise KeyError(f"Unknown ComfyUI node: {node_id}") from exc


def list_nodes(
    include_api: bool = False,
    custom_node_paths: Sequence[str | Path] | None = None,
) -> dict[str, NodeInfo]:
    """Return deterministic metadata for available ComfyUI nodes."""
    return {
        node_id: record.info
        for node_id, record in _registry(include_api, custom_node_paths).items()
    }


def get_node(
    node_id: str,
    include_api: bool = False,
    custom_node_paths: Sequence[str | Path] | None = None,
) -> type[Any]:
    """Return the raw ComfyUI node class for ``node_id``."""
    return _get_record(node_id, include_api, custom_node_paths).node_class


def get_node_info(
    node_id: str,
    include_api: bool = False,
    custom_node_paths: Sequence[str | Path] | None = None,
) -> NodeInfo:
    """Return metadata for one ComfyUI node."""
    return _get_record(node_id, include_api, custom_node_paths).info


def _resolve_api_auth(node_id: str, api_auth: ApiNodeAuth | None) -> ApiNodeAuth:
    auth = api_auth if api_auth is not None else ApiNodeAuth.from_env()
    if not auth.has_credentials():
        raise RuntimeError(
            f"{node_id} is a Comfy.org API node. Pass ApiNodeAuth(api_key=...) "
            "to run_node() or set COMFY_ORG_API_KEY."
        )
    return auth


def _set_api_base_url(base_url: str) -> None:
    try:
        cli_args = importlib.import_module("comfy.cli_args")
        cli_args.args.comfy_api_base = base_url
    except Exception:
        return


def _prepare_api_node_class(
    node_id: str,
    node_class: type[Any],
    api_auth: ApiNodeAuth,
) -> type[Any]:
    _set_api_base_url(api_auth.base_url)

    hidden_inputs = {
        _HIDDEN_API_KEY_COMFY_ORG: api_auth.api_key,
        _HIDDEN_AUTH_TOKEN_COMFY_ORG: api_auth.auth_token,
        _HIDDEN_UNIQUE_ID: api_auth.unique_id or node_id,
    }
    prepare_class_clone = getattr(node_class, "PREPARE_CLASS_CLONE", None)
    if callable(prepare_class_clone):
        return cast(type[Any], prepare_class_clone({"hidden_inputs": hidden_inputs}))
    return node_class


def run_node(
    node_id: str,
    /,
    include_api: bool = False,
    custom_node_paths: Sequence[str | Path] | None = None,
    api_auth: ApiNodeAuth | None = None,
    **inputs: object,
) -> object:
    """Instantiate and execute a raw ComfyUI node, returning its unmodified result."""
    record = _get_record(
        node_id,
        include_api,
        custom_node_paths,
    )
    node_class = record.node_class
    if record.info.is_api_node:
        node_class = _prepare_api_node_class(
            node_id,
            node_class,
            _resolve_api_auth(node_id, api_auth),
        )

    function_name = getattr(node_class, "FUNCTION", None)
    if not isinstance(function_name, str) or not function_name:
        raise RuntimeError(f"ComfyUI node {node_id} has no executable FUNCTION.")

    instance = node_class()
    function = getattr(instance, function_name, None)
    if not callable(function):
        raise RuntimeError(
            f"ComfyUI node {node_id} does not expose callable FUNCTION {function_name!r}."
        )

    result = function(**inputs)
    if inspect.isawaitable(result):
        return _run_coro(result)
    return result


def _clear_registry_cache() -> None:
    """Clear node registry caches for tests."""
    _REGISTRY_CACHE.clear()
    _IMPORT_FAILURES.clear()
