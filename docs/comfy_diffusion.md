# comfy_diffusion

Python library that exposes ComfyUI's inference engine as importable modules — no server, no node graph, no UI.

## Role

Core Python package for direct ComfyUI-backed inference. This repository is centered on the package plus the small `comfy-diffusion` CLI.

## Location

`comfy_diffusion/` at repo root.

## Installation

```bash
# CPU
uv sync --extra cpu --extra comfyui

# CUDA
uv sync --extra cuda --extra comfyui
```

## Public API

| Module | Key symbols |
|--------|-------------|
| `comfy_diffusion` | `check_runtime()`, `vae_decode()`, `vae_encode()`, `apply_lora()` |
| `comfy_diffusion.models` | `ModelManager` — `load_checkpoint`, `load_clip`, `load_vae`, `load_llm` |
| `comfy_diffusion.conditioning` | `encode_prompt()`, advanced/regional/scheduled conditioning |
| `comfy_diffusion.sampling` | `sample()`, `SamplerCustomAdvanced`, schedulers, sigma tools |
| `comfy_diffusion.vae` | tiled, batch, encode/decode variants |
| `comfy_diffusion.lora` | `apply_lora()`, LoRA stacking |
| `comfy_diffusion.controlnet` | `load_controlnet()`, `apply_controlnet()` |
| `comfy_diffusion.latent` | create/resize/crop/compose/batch |
| `comfy_diffusion.image` | load/save/transform |
| `comfy_diffusion.mask` | load/convert/grow/feather |
| `comfy_diffusion.audio` | LTXV Audio VAE, ACE Step text-to-audio |
| `comfy_diffusion.textgen` | `generate_text()`, `generate_ltx2_prompt()` |
| `comfy_diffusion.video` | model sampling patches, video CFG guidance |
| `comfy_diffusion.downloader` | `download_models()`, `HFModelEntry`, `CivitAIModelEntry`, `URLModelEntry` |
| `comfy_diffusion.pipelines.*` | optional ready-made helpers |

## Usage

```python
from comfy_diffusion import check_runtime, vae_decode, apply_lora
from comfy_diffusion.models import ModelManager
from comfy_diffusion.conditioning import encode_prompt
from comfy_diffusion.sampling import sample

check_runtime()

manager = ModelManager(models_dir="/path/to/models")
checkpoint = manager.load_checkpoint("model.safetensors")
model, clip = apply_lora(checkpoint.model, checkpoint.clip, "lora.safetensors", 0.8, 0.8)
positive = encode_prompt(clip, "a portrait")
negative = encode_prompt(clip, "blurry")

import torch
latent = {"samples": torch.zeros(1, 4, 64, 64)}
denoised = sample(model, positive, negative, latent, steps=20, cfg=7.0, sampler_name="euler", scheduler="normal", seed=42)
image = vae_decode(checkpoint.vae, denoised)
image.save("output.png")
```

## Tests

```bash
uv run pytest
```

## Dependencies

- `torch` (optional via `[cpu]` / `[cuda]` extras)
- `ComfyUI` vendored at `vendor/ComfyUI`
- `pillow`, `numpy`, `huggingface_hub`, `tqdm` (core package support)
- `torchaudio`, `opencv-python`, `imageio`, `av` through optional audio/video extras
