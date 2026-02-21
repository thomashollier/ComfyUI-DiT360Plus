"""ComfyUI nodes for DiT360 pipeline loading and text-to-panorama generation."""

import torch
import gc
import comfy.utils
from .pipeline.dit360_txt2pano import generate_panorama
import numpy as np
from PIL import Image


class DiT360PipelineLoader:
    """Load FLUX.1-dev with DiT360 LoRA for 360 panorama generation.

    This loads the full diffusers FluxPipeline and applies the DiT360 LoRA adapter.
    The pipeline is used by all other DiT360Plus nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "black-forest-labs/FLUX.1-dev",
                    "tooltip": "HuggingFace model ID or local path for FLUX base model.",
                }),
                "lora_id": ("STRING", {
                    "default": "Insta360-Research/DiT360-Panorama-Image-Generation",
                    "tooltip": "HuggingFace model ID or local path for DiT360 LoRA weights.",
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
            }
        }

    RETURN_TYPES = ("DIT360_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load"
    CATEGORY = "DiT360Plus/pipeline"

    def load(self, model_id, lora_id, dtype, enable_vae_tiling, enable_vae_slicing):
        from diffusers import FluxPipeline

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        print(f"[DiT360Plus] Loading FLUX pipeline from {model_id}...")
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        print(f"[DiT360Plus] Loading DiT360 LoRA from {lora_id}...")
        pipe.load_lora_weights(lora_id)

        pipe = pipe.to("cuda")

        if enable_vae_tiling:
            pipe.vae.enable_tiling()
        if enable_vae_slicing:
            pipe.vae.enable_slicing()

        print("[DiT360Plus] Pipeline loaded successfully.")
        return (pipe,)


class DiT360TextToPanorama:
    """Generate a 360 panoramic image from a text prompt using DiT360.

    Uses the DiT360-augmented FLUX pipeline with circular padding on packed
    latents for seamless horizontal wrapping. Output is always 2:1 aspect ratio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIT360_PIPELINE", {
                    "tooltip": "DiT360 pipeline from the Pipeline Loader node.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "This is a panorama image. A beautiful mountain landscape with a lake",
                    "tooltip": "Text prompt. Prefix with 'This is a panorama image.' for best results.",
                }),
                "width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 16,
                    "tooltip": "Output width. Height auto-calculated as width/2. Recommended: 2048.",
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Number of denoising steps. Recommended: 50.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.8,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Recommended: 2.8 for DiT360.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": "Random seed for reproducibility.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "DiT360Plus/generation"

    def generate(self, pipeline, prompt, width, steps, guidance_scale, seed):
        # Enforce 2:1 aspect ratio with 16-pixel alignment
        width = (width // 16) * 16
        height = width // 2

        pbar = comfy.utils.ProgressBar(steps)

        def progress_callback(step, total):
            pbar.update(1)

        with torch.no_grad():
            result_image = generate_panorama(
                pipe=pipeline,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                callback=progress_callback,
            )

        # Convert PIL to ComfyUI tensor (B, H, W, C) float32 [0,1]
        img_np = np.array(result_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        return (img_tensor,)


class DiT360PipelineUnloader:
    """Unload the DiT360 pipeline to free GPU memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIT360_PIPELINE",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "DiT360Plus/pipeline"

    def unload(self, pipeline):
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[DiT360Plus] Pipeline unloaded, GPU memory freed.")
        return ()


NODE_CLASS_MAPPINGS = {
    "DiT360PipelineLoader": DiT360PipelineLoader,
    "DiT360TextToPanorama": DiT360TextToPanorama,
    "DiT360PipelineUnloader": DiT360PipelineUnloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360PipelineLoader": "DiT360 Pipeline Loader",
    "DiT360TextToPanorama": "DiT360 Text to Panorama",
    "DiT360PipelineUnloader": "DiT360 Pipeline Unloader",
}
