# comfy-diffusion

[![PyPI version](https://badge.fury.io/py/comfy-diffusion.svg)](https://pypi.org/project/comfy-diffusion/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/comfy-diffusion/)
[![CI](https://github.com/quinteroac/comfy-diffusion/actions/workflows/publish.yml/badge.svg)](https://github.com/quinteroac/comfy-diffusion/actions/workflows/publish.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

`comfy-diffusion` is a standalone Python package that exposes ComfyUI's inference engine as importable modules. It is not a server, node graph runner, web UI, MCP server, daemon, or binary app.

The package vendors ComfyUI at `vendor/ComfyUI` and makes its internal `comfy.*` modules available when runtime APIs need them. Application authors can install this package, import `comfy_diffusion`, and compose inference flows directly in Python.

## Install

Use `uv` for development and dependency resolution:

```bash
uv sync --extra cpu --extra comfyui
```

For CUDA environments:

```bash
uv sync --extra cuda --extra comfyui
```

Useful extras:

| Extra | Includes | Use |
| --- | --- | --- |
| `cpu` | `torch`, `torchvision` | CPU-only development and CI |
| `cuda` | `torch`, `torchvision` via the configured PyTorch CUDA index | NVIDIA GPU inference |
| `comfyui` | ComfyUI runtime dependencies | Importing and running ComfyUI internals |
| `audio` | `torchaudio` | Audio helpers and pipelines |
| `video` | `av`, `imageio`, `opencv-python` | Video I/O helpers |
| `all` | CUDA, audio, video, and ComfyUI runtime dependencies | Full local runtime |

## Python API

The public package root intentionally stays small:

```python
from comfy_diffusion import check_runtime, vae_decode, vae_encode, apply_lora
```

Most APIs are imported from explicit submodules:

```python
from comfy_diffusion.models import ModelManager
from comfy_diffusion.conditioning import encode_prompt
from comfy_diffusion.sampling import sample
```

## Quick Start

Call `check_runtime()` before loading models or sampling. On first runtime use, comfy-diffusion can perform an automatic download of the pinned ComfyUI release when the vendored runtime is missing. Expected failures are returned as an error dict instead of being raised; `check_runtime()` returns an error dict for runtime bootstrap problems.

Example:

```python
from comfy_diffusion import apply_lora, check_runtime, vae_decode
from comfy_diffusion.conditioning import encode_prompt
from comfy_diffusion.models import ModelManager
from comfy_diffusion.sampling import sample

runtime = check_runtime()
if "error" in runtime:
    raise RuntimeError(runtime["error"])

manager = ModelManager(models_dir="/path/to/models")
checkpoint = manager.load_checkpoint("model.safetensors")

model, clip = apply_lora(
    checkpoint.model,
    checkpoint.clip,
    "style.safetensors",
    0.8,
    0.8,
)

positive = encode_prompt(clip, "a portrait, studio lighting")
negative = encode_prompt(clip, "blurry, low quality")

import torch

latent = {"samples": torch.zeros(1, 4, 64, 64)}
denoised = sample(
    model,
    positive,
    negative,
    latent,
    steps=20,
    cfg=7.0,
    sampler_name="euler",
    scheduler="normal",
    seed=42,
)
image = vae_decode(checkpoint.vae, denoised)
image.save("output.png")
```

`comfy_diffusion.pipelines` remains available as an optional helper namespace for explicit ready-made flows, but the main interface is the modular Python API above.

## CLI

The first-party CLI is named `comfy-diffusion` and provides operational package tools only.

```bash
uv run comfy-diffusion runtime check --json
uv run comfy-diffusion runtime paths
uv run comfy-diffusion models list --models-dir /path/to/models
uv run comfy-diffusion models download --manifest models.json --models-dir /path/to/models
```

Model manifest shape:

```json
{
  "models": [
    {
      "type": "hf",
      "repo_id": "org/model",
      "filename": "model.safetensors",
      "dest": "checkpoints",
      "sha256": null
    },
    {
      "type": "url",
      "url": "https://example.com/model.safetensors",
      "dest": "unet/model.safetensors"
    },
    {
      "type": "civitai",
      "model_id": 12345,
      "version_id": 67890,
      "dest": "loras"
    }
  ]
}
```

The CLI does not start servers, manage services, expose MCP tools, run a web UI, queue background jobs, or provide Parallax commands.

## Development

```bash
uv sync --extra cpu --extra comfyui
uv run pytest
uv run ruff check .
```

ComfyUI is pinned as a git submodule at `vendor/ComfyUI`. Do not edit vendored ComfyUI code directly.
