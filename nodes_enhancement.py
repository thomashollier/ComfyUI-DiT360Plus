"""ComfyUI enhancement nodes for DiT360: edge blending, mask tools, 360 viewer."""

import torch
import numpy as np
import uuid
from pathlib import Path
from PIL import Image

import folder_paths
from .utils.equirect import (
    get_equirect_dimensions,
    validate_aspect_ratio,
    blend_edges,
    check_edge_continuity,
)


class Equirect360EdgeBlender:
    """Post-processing edge blending for seamless panorama wraparound.

    Blends the left and right edges of a panoramic image to ensure
    perfect seamless wraparound when viewed in 360 viewers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Panorama image to blend (2:1 equirectangular).",
                }),
                "blend_width": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 200,
                    "tooltip": "Blend width in pixels. Recommended: 10-20. 0 disables blending.",
                }),
                "blend_mode": (["cosine", "linear", "smooth"], {
                    "default": "cosine",
                    "tooltip": "Blend curve type. Cosine is smoothest.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend"
    CATEGORY = "DiT360Plus/post_process"

    def blend(self, image, blend_width, blend_mode):
        if blend_width == 0:
            return (image,)

        blended = blend_edges(image, blend_width, blend_mode)

        is_seamless = check_edge_continuity(blended, threshold=0.05)
        if is_seamless:
            print(f"[DiT360Plus] Edges blended seamlessly (mode: {blend_mode}, width: {blend_width})")
        else:
            print(f"[DiT360Plus] Warning: edges may still have a visible seam")

        return (blended,)


class Equirect360EmptyLatent:
    """Create an empty latent with enforced 2:1 equirectangular aspect ratio.

    Use this when you want to use standard ComfyUI samplers with DiT360.
    Ensures correct 2:1 dimensions for panoramic generation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 8192,
                    "step": 16,
                    "tooltip": "Output width. Height auto = width/2 for 2:1.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 64,
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "DiT360Plus/latent"

    def generate(self, width, batch_size):
        width, height = get_equirect_dimensions(width, alignment=16)
        latent_width = width // 8
        latent_height = height // 8

        latent = torch.zeros(
            [batch_size, 16, latent_height, latent_width],
            dtype=torch.float32,
        )

        print(f"[DiT360Plus] Created equirectangular latent: {width}x{height} -> {latent_width}x{latent_height}")
        return ({"samples": latent},)


class DiT360MaskProcessor:
    """Create and process masks for DiT360 inpainting/outpainting.

    Converts an input image/mask to the proper format for the editing pipeline.

    For **inpainting**: white areas in the input mask = regions to EDIT (regenerate).
    For **outpainting**: white areas = existing content to KEEP.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {
                    "tooltip": "Input mask. White (1.0) = area of interest.",
                }),
            },
            "optional": {
                "image_mask": ("IMAGE", {
                    "tooltip": "Alternative: use an image as mask (converted to grayscale).",
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "DiT360Plus/masks"

    def process(self, mask, image_mask=None):
        if image_mask is not None:
            # Convert image to grayscale mask
            if image_mask.ndim == 4:
                # (B, H, W, C) -> average channels to get grayscale
                mask = image_mask[0, :, :, :3].mean(dim=-1)
            else:
                mask = image_mask
        elif mask.ndim == 3:
            mask = mask[0]

        # Normalize to 0-1 range
        mask = mask.float()
        if mask.max() > 1.0:
            mask = mask / 255.0

        # Threshold to binary
        mask = (mask > 0.5).float()

        # Return as (1, H, W) for ComfyUI MASK type
        return (mask.unsqueeze(0),)


class Equirect360Viewer:
    """Interactive 360 panorama viewer using Three.js.

    Displays the panorama in an interactive spherical viewer
    within the ComfyUI interface.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Panorama image(s) to preview.",
                }),
                "max_resolution": ("INT", {
                    "default": 4096,
                    "min": 512,
                    "max": 8192,
                    "step": 16,
                    "tooltip": "Max preview width. Only affects the preview, not saved output.",
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "DiT360Plus/preview"

    def preview(self, images, max_resolution):
        results = []
        output_dir = folder_paths.get_temp_directory()

        for idx, image in enumerate(images):
            img_np = np.clip(image.cpu().numpy(), 0.0, 1.0)
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            W, H = img_pil.size
            if W > max_resolution:
                new_W = max_resolution
                new_H = new_W // 2
                img_pil = img_pil.resize((new_W, new_H), Image.LANCZOS)

            W, H = img_pil.size
            if not validate_aspect_ratio(W, H, tolerance=0.05):
                print(f"[DiT360Plus] Warning: Image {idx} is not 2:1 ({W}x{H} = {W/H:.2f}:1)")

            filename = f"dit360plus_viewer_{uuid.uuid4().hex}_{idx:05}.png"
            filepath = Path(output_dir) / filename
            img_pil.save(filepath, format="PNG", compress_level=1)
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "Equirect360EdgeBlender": Equirect360EdgeBlender,
    "Equirect360EmptyLatent": Equirect360EmptyLatent,
    "DiT360MaskProcessor": DiT360MaskProcessor,
    "Equirect360Viewer": Equirect360Viewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Equirect360EdgeBlender": "360 Edge Blender",
    "Equirect360EmptyLatent": "360 Empty Latent",
    "DiT360MaskProcessor": "DiT360 Mask Processor",
    "Equirect360Viewer": "360 Viewer",
}
