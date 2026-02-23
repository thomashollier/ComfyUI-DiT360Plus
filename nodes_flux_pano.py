"""ComfyUI node for flexible FLUX panorama pipeline loading with optional LoRA."""

import torch
import folder_paths


class FluxPanoramaLoader:
    """Load any FLUX model for 360 panorama generation with optional LoRA.

    Works with any FLUX model (e.g. FLUX.1-dev, FLUX.1-Kontext-dev) and makes
    LoRA optional. Returns DIT360_PIPELINE so all downstream nodes work unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        diffusion_models = folder_paths.get_filename_list("diffusion_models")
        return {
            "required": {
                "model": (sorted(diffusion_models), {
                    "tooltip": "FLUX model from your ComfyUI diffusion_models folder.",
                }),
                "base_pipeline": (["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-Kontext-dev"], {
                    "default": "black-forest-labs/FLUX.1-dev",
                    "tooltip": "HuggingFace repo for pipeline components (text encoders, VAE, scheduler).",
                }),
                "lora_id": ("STRING", {
                    "default": "",
                    "tooltip": "Optional HuggingFace repo ID or local path for LoRA weights. Leave empty for no LoRA.",
                }),
                "dtype": (["float16", "bfloat16"], {
                    "default": "float16",
                    "tooltip": "Model precision. float16 uses less VRAM, bfloat16 may be more stable.",
                }),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VAE tiling to reduce VRAM usage during encode/decode.",
                }),
                "enable_vae_slicing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VAE slicing for memory efficiency.",
                }),
                "cpu_offload": (["off", "model", "balanced", "sequential"], {
                    "default": "model",
                    "tooltip": "CPU offload strategy. 'model' moves whole models to CPU when idle (fast, best for 1024). 'balanced' splits transformer across GPU/CPU (good for 1536-2048). 'sequential' moves one layer at a time (slowest, lowest VRAM). 'off' keeps everything on GPU.",
                }),
                "balanced_offload_gb": ("FLOAT", {
                    "default": 8.0,
                    "min": 1.0,
                    "max": 48.0,
                    "step": 0.5,
                    "tooltip": "GPU memory budget (GB) for transformer layers in 'balanced' mode. Only used in balanced mode.",
                }),
            }
        }

    RETURN_TYPES = ("DIT360_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "DiT360Plus/pipeline"

    def load(self, model, base_pipeline, lora_id, dtype, enable_vae_tiling, enable_vae_slicing, cpu_offload, balanced_offload_gb=8.0):
        from diffusers import FluxPipeline, FluxTransformer2DModel

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        model_path = folder_paths.get_full_path("diffusion_models", model)

        print(f"[FluxPanorama] Loading FLUX transformer from {model_path}...")
        transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
        )

        print(f"[FluxPanorama] Loading pipeline components from {base_pipeline}...")
        pipe = FluxPipeline.from_pretrained(
            base_pipeline,
            transformer=transformer,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        if lora_id.strip():
            print(f"[FluxPanorama] Loading LoRA from {lora_id}...")
            pipe.load_lora_weights(lora_id)
        else:
            print("[FluxPanorama] No LoRA specified, using base model only.")

        # Offload setup
        if cpu_offload == "off":
            pipe = pipe.to("cuda")
            print("[FluxPanorama] Full GPU mode - all components on CUDA.")
        else:
            pipe = pipe.to("cpu")
            torch.cuda.empty_cache()
            print(f"[FluxPanorama] CPU offload mode: '{cpu_offload}'. All components on CPU.")

        # Tag the pipe with offload mode for downstream pipeline functions
        pipe._dit360_offload = cpu_offload

        if cpu_offload == "balanced":
            no_split = []
            if hasattr(pipe.transformer, 'transformer_blocks') and len(pipe.transformer.transformer_blocks) > 0:
                no_split.append(type(pipe.transformer.transformer_blocks[0]).__name__)
            if hasattr(pipe.transformer, 'single_transformer_blocks') and len(pipe.transformer.single_transformer_blocks) > 0:
                no_split.append(type(pipe.transformer.single_transformer_blocks[0]).__name__)
            pipe._dit360_no_split = no_split
            pipe._dit360_balanced_gb = balanced_offload_gb
            print(f"[FluxPanorama] Balanced mode: will dynamically dispatch transformer at generation time ({balanced_offload_gb}GB budget).")

        elif cpu_offload == "sequential":
            from accelerate import cpu_offload as accel_cpu_offload
            accel_cpu_offload(pipe.transformer, execution_device=torch.device("cuda:0"), offload_buffers=True)
            print("[FluxPanorama] Sequential offload: per-layer hooks applied to transformer.")

        if enable_vae_tiling:
            pipe.vae.enable_tiling()
        if enable_vae_slicing:
            pipe.vae.enable_slicing()

        print("[FluxPanorama] Pipeline loaded successfully.")
        return (pipe,)


NODE_CLASS_MAPPINGS = {
    "FluxPanoramaLoader": FluxPanoramaLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxPanoramaLoader": "Flux Panorama Loader",
}
