"""ComfyUI nodes for DiT360 inpainting and outpainting."""

import torch
import numpy as np
from PIL import Image

import comfy.utils
from .pipeline.dit360_inversion import invert_image, edit_panorama
from .utils.masks import create_mask_for_editing, prepare_mask_for_pipeline


class DiT360ImageInverter:
    """Invert a panoramic image for subsequent editing (inpainting/outpainting).

    Uses RF-Inversion (Controlled Forward ODE) to encode the source image into
    invertible latent representations that can be used for editing while preserving
    the structure of the original image.

    Connect the output to the DiT360 Panorama Editor node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIT360_PIPELINE", {
                    "tooltip": "DiT360 pipeline from the Pipeline Loader node.",
                }),
                "image": ("IMAGE", {
                    "tooltip": "Source panoramic image to invert. Should be 2:1 equirectangular.",
                }),
                "source_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the source image. Can be empty for unconditional inversion.",
                }),
                "num_inversion_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "tooltip": "Number of inversion steps. More steps = more accurate inversion. Recommended: 50.",
                }),
                "gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Inversion fidelity. Higher = more faithful to source. Recommended: 1.0.",
                }),
            }
        }

    RETURN_TYPES = ("DIT360_INVERTED",)
    RETURN_NAMES = ("inverted_data",)
    FUNCTION = "invert"
    CATEGORY = "DiT360Plus/editing"

    def invert(self, pipeline, image, source_prompt, num_inversion_steps, gamma):
        # Convert ComfyUI tensor (B,H,W,C) to PIL
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Determine dimensions (enforce 2:1)
        w, h = pil_image.size
        if abs(w / h - 2.0) > 0.05:
            # Force 2:1 ratio
            h = w // 2
            pil_image = pil_image.resize((w, h), Image.LANCZOS)

        # Round to 16-pixel alignment
        w = (w // 16) * 16
        h = w // 2
        pil_image = pil_image.resize((w, h), Image.LANCZOS)

        pbar = comfy.utils.ProgressBar(num_inversion_steps)

        def progress_callback(step, total):
            pbar.update(1)

        print(f"[DiT360Plus] Inverting image ({w}x{h}) with {num_inversion_steps} steps...")

        with torch.no_grad():
            inverted_data = invert_image(
                pipe=pipeline,
                image=pil_image,
                height=h,
                width=w,
                source_prompt=source_prompt,
                num_inversion_steps=num_inversion_steps,
                gamma=gamma,
                callback=progress_callback,
            )

        print("[DiT360Plus] Image inversion complete.")
        return (inverted_data,)


class DiT360PanoramaEditor:
    """Edit a panoramic image using inpainting or outpainting.

    Uses PersonalizeAnything attention with RF-Inversion to edit specific
    regions of a 360 panorama while preserving the rest.

    For **inpainting**: provide a mask where white = area to edit.
    For **outpainting**: provide a mask where white = existing content to keep.

    Requires inverted data from the DiT360 Image Inverter node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DIT360_PIPELINE", {
                    "tooltip": "DiT360 pipeline from the Pipeline Loader node.",
                }),
                "inverted_data": ("DIT360_INVERTED", {
                    "tooltip": "Inverted latents from the DiT360 Image Inverter node.",
                }),
                "mask": ("MASK", {
                    "tooltip": "Editing mask. For inpainting: white=edit area. For outpainting: white=keep area.",
                }),
                "mode": (["inpaint", "outpaint"], {
                    "default": "outpaint",
                    "tooltip": "Editing mode. Inpaint: replace white mask area. Outpaint: extend beyond white mask area.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "This is a panorama image. A beautiful landscape",
                    "tooltip": "Description of the desired result after editing.",
                }),
                "tau": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Preservation strength (0-100). Higher = stronger source preservation. Recommended: 50.",
                }),
                "eta": ("FLOAT", {
                    "default": 0.75,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Reconstruction vs. edit strength. 1.0 = pure reconstruction (no change). 0.0 = pure transformer (chaotic). Lower values allow stronger edits. Recommended: 0.6-0.8.",
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "tooltip": "Number of denoising steps. Should match inversion steps.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.8,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Recommended: 2.8.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "start_timestep": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Start of eta guidance window. Recommended: 0.0.",
                }),
                "stop_timestep": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "End of eta guidance window. Recommended: 0.99.",
                }),
                "mask_feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Soft mask edge width as percentage of image width. 0 = hard edges. 1-3% gives subtle blending, 5-10% for wider transitions. Resolution-independent.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit"
    CATEGORY = "DiT360Plus/editing"

    def edit(self, pipeline, inverted_data, mask, mode, prompt,
             tau, eta, steps, guidance_scale, seed, start_timestep, stop_timestep,
             mask_feather):

        height = inverted_data["height"]
        width = inverted_data["width"]
        vae_sf = pipeline.vae_scale_factor

        # Compute latent dimensions
        latent_h = height // (vae_sf * 2)
        latent_w = width // (vae_sf * 2)

        # Process mask: resize to latent dims, apply mode inversion, feather edges
        processed_mask = create_mask_for_editing(mask, mode, latent_w, latent_h, feather=mask_feather)

        # Add circular padding and flatten for attention processor
        pipeline_mask = prepare_mask_for_pipeline(processed_mask, latent_w, latent_h)

        pbar = comfy.utils.ProgressBar(steps)

        def progress_callback(step, total):
            pbar.update(1)

        print(f"[DiT360Plus] Editing panorama ({width}x{height}), mode={mode}, tau={tau}, eta={eta}")

        with torch.no_grad():
            result_image = edit_panorama(
                pipe=pipeline,
                inverted_data=inverted_data,
                mask=pipeline_mask,
                source_prompt="",
                edit_prompt=prompt,
                tau=tau / 100.0,
                eta=eta,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                start_timestep=start_timestep,
                stop_timestep=stop_timestep,
                callback=progress_callback,
            )

        # Convert PIL to ComfyUI tensor
        img_np = np.array(result_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        print("[DiT360Plus] Editing complete.")
        return (img_tensor,)


NODE_CLASS_MAPPINGS = {
    "DiT360ImageInverter": DiT360ImageInverter,
    "DiT360PanoramaEditor": DiT360PanoramaEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiT360ImageInverter": "DiT360 Image Inverter",
    "DiT360PanoramaEditor": "DiT360 Panorama Editor",
}
