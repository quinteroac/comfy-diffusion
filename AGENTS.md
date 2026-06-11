# Agents entry point

- **What this project is:** `comfy-diffusion` is a standalone Python library that exposes ComfyUI's inference engine (`comfy.*` modules) as importable Python modules — no server, no node graph, no UI layer. It is consumed exactly like `diffusers` or `DiffSynth-Studio`: `import comfy_diffusion` and run inference directly in your own code. ComfyUI is vendored as a git submodule at `vendor/ComfyUI` and its internal modules are made importable by runtime APIs when needed. The library is designed to be a single `pip`/`uv` dependency that any Python application (FastAPI backend, script, pipeline) can add without operating a separate ComfyUI server.

- **How to work here:** Use this file as the single entry point. The repo is now a Python package plus the small `comfy-diffusion` CLI; do not add server, MCP, frontend, daemon, service manager, installer, or standalone-binary layers unless explicitly requested. **Python:** use [uv](https://docs.astral.sh/uv/) for all install, run, and dependency commands (`uv sync`, `uv run`, `uv add`) — never use `pip` or `venv` directly.

- **Roadmap and node inventory:** [`ROADMAP.md`](ROADMAP.md) — full iteration plan, node classification (Roadmap / Nice-to-have / Discarded), and optional dependency schema.

- **Key architecture decisions (do not revisit without explicit instruction):**
  - ComfyUI is vendored at `vendor/ComfyUI` as a git submodule pinned to an explicit ref (release tag or full commit SHA) — never floating implicitly. Update the pin deliberately between iterations only.
  - `sys.path` manipulation is encapsulated entirely inside `comfy_diffusion/_runtime.py` — consumers never touch paths manually. Use absolute paths derived from `__file__`.
  - The node system is loaded only through the explicit experimental `comfy_diffusion.nodes` escape hatch. Default node discovery may load ComfyUI core and built-in extra nodes; API nodes require opt-in; external custom nodes are loaded only from explicit trusted paths and are never discovered by scanning ComfyUI's default `custom_nodes` folder.
  - `torch` is an optional dependency declared as extras (`comfy-diffusion[cuda]` / `comfy-diffusion[cpu]`) — never hardcode a torch version or index URL in core dependencies.
  - `check_runtime()` returns an error dict (never raises) when the ComfyUI submodule is not initialized. `python_version` is always populated regardless.
  - All tests must pass on CPU-only environments — CI has no GPU. GPU is validated locally before merging.
  - Test approach: critical paths only (pytest via `uv run pytest`). Test plans are written after prototyping, during the Refactor phase.
  - Git flow: feature branches per iteration (`feature/it-000001-foundation`), merged to `main` via PR.
  - Public API pattern: modules are not auto-imported from `__init__.py` by default. Exceptions are `check_runtime`, `vae_decode`, `vae_encode`, and `apply_lora` which are re-exported for convenience. All other symbols use explicit submodule imports (e.g. `from comfy_diffusion.conditioning import encode_prompt`).
  - Lazy import pattern: no `torch`, `comfy.*`, or `ensure_comfyui_on_path()` at module top level — all deferred to call time inside function bodies. Exception: `vae.py` uses pure duck typing (no comfy import at all) — both patterns are valid.
  - Inference mode ownership: `torch.inference_mode()` is enforced centrally in core execution APIs (`sampling.py`, `vae.py`, and relevant `audio.py` wrappers). Pipeline authors must not duplicate inference-mode wrappers in each `run()` implementation.
  - `path` type annotation: `str | Path` is the established pattern across `ModelManager`, `load_checkpoint`, and `apply_lora`. Do not change to `str | os.PathLike` unless updating all occurrences simultaneously in a dedicated cleanup iteration.
  - No high-level pipeline abstraction: comfy-diffusion is a modular runtime library. There is no `ImagePipeline` or equivalent. Callers compose the building blocks directly. This is intentional — the modularity is the feature.
  - CLI scope: `comfy-diffusion` is operational tooling for runtime diagnostics, path inspection, model listing, manifest-based model downloads, raw node inspection, and trusted custom-node Git installs only.
  - Removed application layers must stay removed: no Parallax CLI, no FastAPI server, no MCP server, no web frontend, no async job queue, no service manager, no PyInstaller binary release flow.
  - External libraries over node ports: prefer `Pillow`, `numpy`, `opencv-python`, `torchaudio` for image transforms, mask ops, video I/O, and audio I/O respectively. Only wrap comfy nodes when they provide non-trivial logic (VAE, samplers, model patches, conditioning). See ROADMAP.md for the full classification.

- **Rule:** All generated resources in this repo must be in English.
- For any file search or grep in the current git indexed directory use fff tools
