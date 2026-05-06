# comfy-diffusion CLI

The `comfy-diffusion` CLI is a small operational companion to the Python package.

It does not run inference products, start a server, register services, expose MCP tools, or manage a web frontend. Use the Python API for inference composition.

## Runtime

```bash
uv run comfy-diffusion runtime check
uv run comfy-diffusion runtime check --json
uv run comfy-diffusion runtime paths
uv run comfy-diffusion runtime paths --models-dir /path/to/models --json
```

## Models

```bash
uv run comfy-diffusion models list --models-dir /path/to/models
uv run comfy-diffusion models download --manifest models.json --models-dir /path/to/models
```

Manifest format:

```json
{
  "models": [
    {
      "type": "hf",
      "repo_id": "org/model",
      "filename": "model.safetensors",
      "dest": "checkpoints"
    },
    {
      "type": "url",
      "url": "https://example.com/model.safetensors",
      "dest": "unet/model.safetensors"
    },
    {
      "type": "civitai",
      "model_id": 12345,
      "dest": "loras",
      "version_id": 67890
    }
  ]
}
```
