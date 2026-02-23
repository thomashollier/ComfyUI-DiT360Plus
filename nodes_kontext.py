"""ComfyUI node for Kontext-based panorama inpainting."""

import torch
import numpy as np
from PIL import Image

import comfy.utils


def _remove_accelerate_hooks(module):
    """Remove any existing accelerate dispatch hooks from a module tree.

    The FluxPanoramaLoader may have applied accelerate hooks for sequential/balanced
    offload. We need to clear them before applying diffusers' built-in offload,
    which manages device placement through its own mechanism.
    """
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(module)
    except (ImportError, Exception):
        pass


def _tensor_to_pil(tensor):
    """Convert ComfyUI image tensor (B,H,W,C) float32 [0,1] to PIL Image."""
    img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _mask_to_pil(tensor):
    """Convert ComfyUI mask tensor (B,H,W) float32 [0,1] to PIL Image (L mode)."""
    mask_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


def _pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI image tensor (B,H,W,C) float32 [0,1]."""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


class KontextPanoramaEditor:
    """Edit panorama regions using FLUX Kontext inpainting.

    Uses FluxKontextInpaintPipeline from diffusers to edit specific regions
    of a panoramic image. Much simpler than RF-Inversion — no inversion step
    needed. Just provide image + mask + prompt.

    Mask convention: white = area to edit (repaint).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIT360_PIPELINE", {
                    "tooltip": "FLUX pipeline from the Flux Panorama Loader (loaded with Kontext base_pipeline).",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Source panoramic image to edit. (B,H,W,C) float32 [0,1].",
                }),
                "mask": ("MASK", {
                    "tooltip": "Editing mask. White (1.0) = area to repaint, Black (0.0) = preserve.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Describes the desired edit for the masked region.",
                }),
                "steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Number of denoising steps. Recommended: 28.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Recommended: 3.5 for Kontext.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much to modify the masked region. 1.0 = full repaint. Recommended: 1.0.",
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
    FUNCTION = "edit"
    CATEGORY = "DiT360Plus/editing"

    def edit(self, pipeline, image, mask, prompt, steps, guidance_scale, strength, seed):
        from diffusers import FluxKontextInpaintPipeline

        # Wrap existing pipeline components into Kontext inpaint pipeline.
        # This reuses already-loaded model weights — no extra VRAM.
        inpaint_pipe = FluxKontextInpaintPipeline(
            transformer=pipeline.transformer,
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            vae=pipeline.vae,
            scheduler=pipeline.scheduler,
        )

        # Remove any existing accelerate hooks from the loader before applying
        # diffusers' built-in offload (avoids double-hooking conflicts).
        for component in [inpaint_pipe.transformer, inpaint_pipe.text_encoder,
                          inpaint_pipe.text_encoder_2, inpaint_pipe.vae]:
            _remove_accelerate_hooks(component)

        # Configure offload based on the loader's tag
        offload_mode = getattr(pipeline, '_dit360_offload', 'model')
        if offload_mode == "off":
            inpaint_pipe = inpaint_pipe.to("cuda")
        elif offload_mode == "model":
            inpaint_pipe.enable_model_cpu_offload()
        else:  # balanced or sequential
            inpaint_pipe.enable_sequential_cpu_offload()

        # VAE memory optimization
        inpaint_pipe.vae.enable_tiling()
        inpaint_pipe.vae.enable_slicing()

        # Convert ComfyUI tensors to PIL
        image_pil = _tensor_to_pil(image)
        mask_pil = _mask_to_pil(mask)

        # Progress tracking
        pbar = comfy.utils.ProgressBar(steps)

        def step_callback(pipe, step, timestep, callback_kwargs):
            pbar.update(1)
            return callback_kwargs

        print(f"[KontextEditor] Editing panorama ({image_pil.size[0]}x{image_pil.size[1]}), "
              f"steps={steps}, guidance={guidance_scale}, strength={strength}")

        # Run Kontext inpainting
        with torch.no_grad():
            result = inpaint_pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=step_callback,
            ).images[0]

        # Convert back to ComfyUI tensor (B,H,W,C)
        output_tensor = _pil_to_tensor(result)

        print("[KontextEditor] Editing complete.")
        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "KontextPanoramaEditor": KontextPanoramaEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KontextPanoramaEditor": "Kontext Panorama Editor",
}
