"""Model management public API.

This module must stay import-safe in CPU-only environments. It intentionally avoids
importing ComfyUI loaders at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._runtime import ensure_comfyui_on_path


@dataclass
class CheckpointResult:
    """Container for objects produced by a ComfyUI checkpoint load."""

    model: Any
    clip: Any | None
    vae: Any | None


class ModelManager:
    """Entry point for model-loading operations.

    The implementation is intentionally deferred to later user stories; this class
    exists now to provide a stable, side-effect-free import surface.
    """

    def __init__(self, models_dir: str | Path) -> None:
        """Store and validate the models directory used by future load operations."""
        path = Path(models_dir)

        if not path.exists():
            raise ValueError(f"models_dir does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"models_dir is not a directory: {path}")

        self.models_dir = path

        ensure_comfyui_on_path()
        import folder_paths

        folder_paths.add_model_folder_path(
            "checkpoints", str(self.models_dir / "checkpoints"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "embeddings", str(self.models_dir / "embeddings"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "diffusion_models", str(self.models_dir / "unet"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "diffusion_models", str(self.models_dir / "diffusion_models"), is_default=False
        )
        folder_paths.add_model_folder_path(
            "text_encoders", str(self.models_dir / "text_encoders"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "text_encoders", str(self.models_dir / "clip"), is_default=False
        )
        folder_paths.add_model_folder_path(
            "vae", str(self.models_dir / "vae"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "llm", str(self.models_dir / "llm"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "upscale_models", str(self.models_dir / "upscale_models"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "latent_upscale_models", str(self.models_dir / "upscale"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "audio_encoders", str(self.models_dir / "audio_encoders"), is_default=True
        )
        folder_paths.add_model_folder_path(
            "optical_flow", str(self.models_dir / "optical_flow"), is_default=True
        )

    def load_checkpoint(self, filename: str) -> CheckpointResult:
        """Load a checkpoint by filename from the configured checkpoints directory."""
        ensure_comfyui_on_path()

        requested_path = (self.models_dir / "checkpoints" / filename).resolve()
        if not requested_path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {requested_path}")

        import folder_paths
        from comfy import sd as comfy_sd

        checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", filename)
        loaded = comfy_sd.load_checkpoint_guess_config(
            checkpoint_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = loaded[:3]
        return CheckpointResult(model=model, clip=clip, vae=vae)

    def load_checkpoint_from_path(self, path: str | Path) -> CheckpointResult:
        """Load a checkpoint from an absolute path, extracting model, clip, and vae.

        Equivalent to ComfyUI's ``CheckpointLoaderSimple`` — works with bundled
        checkpoints that contain both the diffusion model and the VAE.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        p = Path(path)
        if not p.is_absolute() or not p.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {p}")

        loaded = comfy_sd.load_checkpoint_guess_config(
            str(p),
            output_vae=True,
            output_clip=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        model, clip, vae = loaded[:3]
        return CheckpointResult(model=model, clip=clip, vae=vae)

    def load_vae(self, path: str | Path) -> Any:
        """Load a standalone VAE from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``vae`` folder.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd
        from comfy import utils as comfy_utils

        p = Path(path)
        if p.is_absolute() and p.is_file():
            vae_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"vae file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            vae_path = folder_paths.get_full_path_or_raise("vae", name)

        state_dict, metadata = comfy_utils.load_torch_file(
            vae_path, return_metadata=True
        )
        vae = comfy_sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def load_clip(
        self,
        *paths: str | Path,
        clip_type: str = "stable_diffusion",
    ) -> Any:
        """Load a standalone text encoder (CLIP) from a path or filename.

        If a path is absolute, it must point to an existing file.
        Otherwise each path is treated as a filename under ``text_encoders`` / ``clip``.

        ``clip_type`` selects the encoder architecture (e.g. ``"wan"`` for Wan / UMT5-XXL,
        ``"stable_diffusion"``, ``"sd3"``, ``"flux"``). Must match the model weights.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        if not paths:
            raise ValueError("load_clip requires at least one path")

        valid_clip_types = {
            member.name.lower(): member for member in comfy_sd.CLIPType
        }
        normalized_clip_type = clip_type.lower()
        if normalized_clip_type not in valid_clip_types:
            valid_names = ", ".join(sorted(valid_clip_types))
            raise ValueError(
                f"invalid clip_type '{clip_type}'. valid values: {valid_names}"
            )

        resolved_paths: list[str] = []
        for path in paths:
            p = Path(path)
            if p.is_absolute():
                if not p.is_file():
                    raise FileNotFoundError(f"clip file not found: {p}")
                resolved_paths.append(str(p.resolve()))
                continue

            name = path if isinstance(path, str) else p.name
            try:
                resolved_path = folder_paths.get_full_path_or_raise("text_encoders", name)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"clip file not found: {path}") from exc

            resolved_path_obj = Path(resolved_path)
            if not resolved_path_obj.is_file():
                raise FileNotFoundError(f"clip file not found: {path}")
            resolved_paths.append(str(resolved_path_obj.resolve()))

        return comfy_sd.load_clip(
            ckpt_paths=resolved_paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=valid_clip_types[normalized_clip_type],
        )

    def load_clip_vision(self, path: str | Path) -> Any:
        """Load a CLIP vision model from a path or filename."""
        ensure_comfyui_on_path()

        import comfy.clip_vision

        p = Path(path)
        if p.is_absolute() and p.is_file():
            full_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"clip vision file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            candidate = self.models_dir / "clip_vision" / name
            if not candidate.is_file():
                raise FileNotFoundError(f"clip vision file not found: {candidate.resolve()}")
            full_path = str(candidate.resolve())

        return comfy.clip_vision.load(full_path)

    def load_unet(self, path: str | Path) -> Any:
        """Load a standalone diffusion model (UNet) from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename and resolved under the
        ``diffusion_models`` / ``unet`` folders (see ComfyUI folder layout).
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        p = Path(path)
        if p.is_absolute() and p.is_file():
            full_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"unet file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            full_path = folder_paths.get_full_path_or_raise("diffusion_models", name)

        return comfy_sd.load_diffusion_model(full_path)

    def load_ltxv_audio_vae(self, path: str | Path) -> object:
        """Load an LTXV audio VAE checkpoint from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``vae`` folder.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd
        from comfy import utils as comfy_utils

        p = Path(path)
        if p.is_absolute() and p.is_file():
            checkpoint_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"ltxv audio vae file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", name)

        state_dict, metadata = comfy_utils.load_torch_file(
            checkpoint_path, return_metadata=True
        )
        state_dict = comfy_utils.state_dict_prefix_replace(
            state_dict,
            {"audio_vae.": "autoencoder.", "vocoder.": "vocoder."},
            filter_keys=True,
        )
        vae = comfy_sd.VAE(sd=state_dict, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def load_ltxav_text_encoder(
        self,
        text_encoder_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "default",
    ) -> object:
        """Load an LTXAV text encoder from two separate files.

        ``text_encoder_path`` is the text encoder file (from ``text_encoders/``).
        ``checkpoint_path`` is the companion checkpoint file (from ``checkpoints/``).
        Both can be absolute paths to existing files or relative filenames resolved
        via folder_paths.

        ``device`` controls placement: ``"default"`` uses ComfyUI's device management;
        ``"cpu"`` pins both load and offload devices to CPU.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        te_p = Path(text_encoder_path)
        if te_p.is_absolute() and te_p.is_file():
            resolved_te = str(te_p.resolve())
        elif te_p.is_absolute():
            raise FileNotFoundError(f"ltxav text encoder file not found: {te_p}")
        else:
            name = text_encoder_path if isinstance(text_encoder_path, str) else te_p.name
            resolved_te = folder_paths.get_full_path_or_raise("text_encoders", name)

        ckpt_p = Path(checkpoint_path)
        if ckpt_p.is_absolute() and ckpt_p.is_file():
            resolved_ckpt = str(ckpt_p.resolve())
        elif ckpt_p.is_absolute():
            raise FileNotFoundError(f"ltxav checkpoint file not found: {ckpt_p}")
        else:
            name = checkpoint_path if isinstance(checkpoint_path, str) else ckpt_p.name
            resolved_ckpt = folder_paths.get_full_path_or_raise("checkpoints", name)

        kwargs: dict[str, Any] = {}
        if device == "cpu":
            import torch

            cpu = torch.device("cpu")
            kwargs["model_options"] = {"load_device": cpu, "offload_device": cpu}

        return comfy_sd.load_clip(
            ckpt_paths=[resolved_te, resolved_ckpt],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy_sd.CLIPType.LTXV,
            **kwargs,
        )

    def load_llm(self, path: str | Path) -> Any:
        """Load a standalone LLM/VLM text model from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename searched in this order:
        ``models_dir/llm`` → ``models_dir/text_encoders`` → ``models_dir/clip``.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import sd as comfy_sd

        p = Path(path)
        if p.is_absolute() and p.is_file():
            llm_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"llm file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            candidates = [
                self.models_dir / "llm" / name,
                self.models_dir / "text_encoders" / name,
                self.models_dir / "clip" / name,
            ]
            resolved = next((c.resolve() for c in candidates if c.is_file()), None)
            if resolved is None:
                raise FileNotFoundError(
                    "llm file not found: "
                    f"{name} (tried {candidates[0].resolve()}, {candidates[1].resolve()}, {candidates[2].resolve()})"
                )
            llm_path = str(resolved)

        return comfy_sd.load_clip(
            ckpt_paths=[llm_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

    def load_audio_encoder(self, path: str | Path) -> Any:
        """Load a wav2vec2-compatible audio encoder from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``audio_encoders`` folder.

        Internally calls ``comfy.utils.load_torch_file`` and
        ``comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd``.
        Raises ``FileNotFoundError`` if the file cannot be found.
        Raises ``ValueError`` if the state dict is not a recognised audio encoder.

        Returns an ``AudioEncoderModel`` instance.
        """
        ensure_comfyui_on_path()

        import folder_paths
        from comfy import utils as comfy_utils
        from comfy.audio_encoders import audio_encoders as comfy_audio_encoders

        p = Path(path)
        if p.is_absolute() and p.is_file():
            model_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"audio encoder file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            model_path = folder_paths.get_full_path_or_raise("audio_encoders", name)

        sd = comfy_utils.load_torch_file(model_path, safe_load=True)
        audio_encoder = comfy_audio_encoders.load_audio_encoder_from_sd(sd)
        if audio_encoder is None:
            raise ValueError(
                f"unrecognised audio encoder state dict in: {model_path}"
            )
        return audio_encoder

    def load_optical_flow(self, path: str | Path) -> Any:
        """Load an optical-flow model from a path or filename.

        Mirrors ComfyUI's ``OpticalFlowLoader`` used by VOID.  If ``path`` is an
        absolute path, the file must exist.  Otherwise it is treated as a
        filename under ``models_dir/optical_flow``.
        """
        ensure_comfyui_on_path()

        p = Path(path)
        if p.is_absolute():
            if not p.is_file():
                raise FileNotFoundError(f"optical flow model file not found: {p}")
            model_name = p.name
        else:
            model_name = path if isinstance(path, str) else p.name

        from comfy_extras.nodes_void import OpticalFlowLoader

        output = OpticalFlowLoader.execute(model_name)
        result = getattr(output, "result", output)
        return result[0]

    def load_vae_kj(
        self,
        path: str | Path,
        device: str = "main_device",
        dtype: str = "bf16",
    ) -> Any:
        """Load a standalone VAE with explicit device and dtype placement.

        Mirrors ``VAELoaderKJ.load_vae()`` from KJNodes (comfyui-kjnodes).

        ``path`` may be an absolute path to an existing file, or a filename
        searched under the ``vae`` folder registered with this ``ModelManager``.

        ``device`` must be ``"main_device"`` or ``"cpu"``.
        ``dtype`` must be one of ``"bf16"``, ``"fp16"``, or ``"fp32"``.

        Returns a ``comfy.sd.VAE`` instance loaded with the specified device and
        dtype.  Audio VAEs (containing ``vocoder.conv_post.weight``) are loaded
        as ``AudioVAE`` without applying the device/dtype overrides.
        """
        ensure_comfyui_on_path()

        _dtype_map = {"bf16": None, "fp16": None, "fp32": None}
        if dtype not in _dtype_map:
            raise ValueError(f"invalid dtype '{dtype}'. valid values: {', '.join(_dtype_map)}")
        if device not in ("main_device", "cpu"):
            raise ValueError(f"invalid device '{device}'. valid values: main_device, cpu")

        import torch

        import comfy.model_management
        import folder_paths
        from comfy import sd as comfy_sd
        from comfy import utils as comfy_utils

        torch_dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        torch_dtype = torch_dtype_map[dtype]

        if device == "main_device":
            torch_device = comfy.model_management.get_torch_device()
        else:
            torch_device = torch.device("cpu")

        p = Path(path)
        if p.is_absolute() and p.is_file():
            vae_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"vae file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            vae_path = folder_paths.get_full_path_or_raise("vae", name)

        sd, metadata = comfy_utils.load_torch_file(vae_path, return_metadata=True)

        if "vocoder.conv_post.weight" in sd:
            from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

            return AudioVAE(sd, metadata)

        vae = comfy_sd.VAE(sd=sd, device=torch_device, dtype=torch_dtype, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    def load_latent_upscale_model(self, path: str | Path) -> Any:
        """Load a LATENT_UPSCALE_MODEL from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``upscale`` folder
        (registered as ``latent_upscale_models`` in folder_paths).

        Returns a latent upscale model (nn.Module) compatible with
        ``ltxv_latent_upsample()``.
        """
        ensure_comfyui_on_path()

        import json

        import folder_paths
        from comfy import utils as comfy_utils

        p = Path(path)
        if p.is_absolute() and p.is_file():
            model_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"latent upscale model file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            model_path = folder_paths.get_full_path_or_raise("latent_upscale_models", name)

        sd, metadata = comfy_utils.load_torch_file(
            model_path, safe_load=True, return_metadata=True
        )

        # Dispatch mirrors LatentUpscaleModelLoader.execute() from nodes_hunyuan.py
        if "blocks.0.block.0.conv.weight" in sd:
            from comfy_extras.nodes_hunyuan import HunyuanVideo15SRModel

            config = {
                "in_channels": sd["in_conv.conv.weight"].shape[1],
                "out_channels": sd["out_conv.conv.weight"].shape[0],
                "hidden_channels": sd["in_conv.conv.weight"].shape[0],
                "num_blocks": len(
                    [
                        k
                        for k in sd.keys()
                        if k.startswith("blocks.") and k.endswith(".block.0.conv.weight")
                    ]
                ),
                "global_residual": False,
            }
            model = HunyuanVideo15SRModel("720p", config)
            model.load_sd(sd)
        elif "up.0.block.0.conv1.conv.weight" in sd:
            from comfy_extras.nodes_hunyuan import HunyuanVideo15SRModel

            sd = {
                key.replace("nin_shortcut", "nin_shortcut.conv", 1): value
                for key, value in sd.items()
            }
            config = {
                "z_channels": sd["conv_in.conv.weight"].shape[1],
                "out_channels": sd["conv_out.conv.weight"].shape[0],
                "block_out_channels": tuple(
                    sd[f"up.{i}.block.0.conv1.conv.weight"].shape[0]
                    for i in range(
                        len(
                            [
                                k
                                for k in sd.keys()
                                if k.startswith("up.")
                                and k.endswith(".block.0.conv1.conv.weight")
                            ]
                        )
                    )
                ),
            }
            model = HunyuanVideo15SRModel("1080p", config)
            model.load_sd(sd)
        elif "post_upsample_res_blocks.0.conv2.bias" in sd:
            import torch

            import comfy.model_management
            from comfy.ldm.lightricks.latent_upsampler import LatentUpsampler

            config = json.loads(metadata["config"])
            model = LatentUpsampler.from_config(config).to(
                dtype=comfy.model_management.vae_dtype(
                    allowed_dtypes=[torch.bfloat16, torch.float32]
                )
            )
            model.load_state_dict(sd)
        else:
            raise ValueError(
                "unknown latent upscale model type: unrecognized state dict keys"
            )

        return model

    def load_upscale_model(self, path: str | Path) -> Any:
        """Load an upscale (super-resolution) model from a path or filename.

        If ``path`` is an absolute path to an existing file, that file is loaded.
        Otherwise ``path`` is treated as a filename under the ``upscale_models`` folder.

        Returns a ``spandrel.ImageModelDescriptor`` usable with ``image_upscale_with_model()``.
        Raises ``TypeError`` if the loaded model is not a single-image descriptor.
        """
        ensure_comfyui_on_path()

        import comfy.utils
        from spandrel import ImageModelDescriptor, ModelLoader

        p = Path(path)
        if p.is_absolute() and p.is_file():
            model_path = str(p.resolve())
        elif p.is_absolute():
            raise FileNotFoundError(f"upscale model file not found: {p}")
        else:
            name = path if isinstance(path, str) else p.name
            candidate = self.models_dir / "upscale_models" / name
            if not candidate.is_file():
                raise FileNotFoundError(f"upscale model file not found: {candidate.resolve()}")
            model_path = str(candidate.resolve())

        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise TypeError(
                f"upscale model must be a single-image model (ImageModelDescriptor), "
                f"got {type(out).__name__}"
            )

        return out

def model_sampling_flux(
    model: Any,
    max_shift: float,
    min_shift: float,
    width: int,
    height: int,
) -> Any:
    """Patch a model clone with Flux sampling shift settings.

    The shift value is interpolated from ``min_shift`` and ``max_shift`` based on
    latent token count derived from ``width`` and ``height``.
    """
    ensure_comfyui_on_path()
    import comfy.model_sampling

    patched_model = model.clone()

    x1 = 256
    x2 = 4096
    slope = (max_shift - min_shift) / (x2 - x1)
    intercept = min_shift - slope * x1
    latent_tokens = (width * height) / (8 * 8 * 2 * 2)
    shift = latent_tokens * slope + intercept

    model_sampling_type = type(
        "ModelSamplingAdvanced",
        (comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST),
        {},
    )
    model_sampling = model_sampling_type(model.model.model_config)
    model_sampling.set_parameters(shift=shift)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


def model_sampling_sd3(model: Any, shift: float) -> Any:
    """Patch a model clone with SD3 sampling shift settings."""
    ensure_comfyui_on_path()
    import comfy.model_sampling

    patched_model = model.clone()

    model_sampling_type = type(
        "ModelSamplingAdvanced",
        (comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST),
        {},
    )
    model_sampling = model_sampling_type(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=1000)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


def model_sampling_aura_flow(model: Any, shift: float) -> Any:
    """Patch a model clone with AuraFlow continuous V-prediction shift settings."""
    ensure_comfyui_on_path()
    import comfy.model_sampling

    patched_model = model.clone()

    model_sampling_type = type(
        "ModelSamplingAdvanced",
        (comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST),
        {},
    )
    model_sampling = model_sampling_type(model.model.model_config)
    model_sampling.set_parameters(shift=shift, multiplier=1.0)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


__all__ = [
    "CheckpointResult",
    "ModelManager",
    "load_latent_upscale_model",
    "model_sampling_aura_flow",
    "model_sampling_flux",
    "model_sampling_sd3",
]
