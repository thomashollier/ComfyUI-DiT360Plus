"""ComfyUI nodes for equirectangular ↔ perspective projection.

Extract a distortion-free perspective patch from a 360 panorama, edit it with
any standard image tool (Kontext, inpainting, etc.), then composite it back.
"""

import torch

from .utils.perspective import equirect_to_perspective, perspective_to_equirect


class EquirectToPersp:
    """Extract a perspective (rectilinear) view from an equirectangular panorama.

    Point the virtual camera at (yaw, pitch) with the given FOV and render a
    distortion-free perspective image. Use this before editing a local region
    of a panorama with tools that don't understand equirectangular distortion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Equirectangular panorama (2:1 aspect ratio).",
                }),
                "yaw": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Horizontal look direction in degrees. 0 = centre, ±180 = behind.",
                }),
                "pitch": ("FLOAT", {
                    "default": 0.0,
                    "min": -90.0,
                    "max": 90.0,
                    "step": 1.0,
                    "tooltip": "Vertical look direction in degrees. Positive = up, negative = down.",
                }),
                "fov": ("FLOAT", {
                    "default": 90.0,
                    "min": 10.0,
                    "max": 150.0,
                    "step": 1.0,
                    "tooltip": "Horizontal field of view in degrees. 90 is standard, wider = more content.",
                }),
                "output_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Width of the output perspective image in pixels.",
                }),
                "output_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Height of the output perspective image in pixels.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "EQUIRECT_PROJECTION")
    RETURN_NAMES = ("perspective", "equirect_passthrough", "projection_data")
    FUNCTION = "extract"
    CATEGORY = "DiT360Plus/projection"

    def extract(self, image, yaw, pitch, fov, output_width, output_height):
        perspective = equirect_to_perspective(
            image, yaw, pitch, fov, output_width, output_height,
        )

        projection_data = {
            "yaw": yaw,
            "pitch": pitch,
            "fov": fov,
            "persp_w": output_width,
            "persp_h": output_height,
        }

        print(f"[DiT360Plus] Extracted perspective view: yaw={yaw:.1f}° pitch={pitch:.1f}° "
              f"fov={fov:.1f}° → {output_width}x{output_height}")

        return (perspective, image, projection_data)


class PerspToEquirect:
    """Composite an edited perspective patch back into an equirectangular panorama.

    Takes the edited perspective image, the original panorama, and the projection
    data from EquirectToPersp. Reprojects the patch and blends it seamlessly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "perspective_image": ("IMAGE", {
                    "tooltip": "Edited perspective image (from Equirect→Perspective or after editing).",
                }),
                "equirect_image": ("IMAGE", {
                    "tooltip": "Original equirectangular panorama (passthrough from Equirect→Perspective).",
                }),
                "projection_data": ("EQUIRECT_PROJECTION", {
                    "tooltip": "Projection parameters from the Equirect→Perspective node.",
                }),
                "blend_pixels": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Edge feather width in pixels for smooth blending. 0 = hard edge.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composited", "mask")
    FUNCTION = "composite"
    CATEGORY = "DiT360Plus/projection"

    def composite(self, perspective_image, equirect_image, projection_data, blend_pixels):
        composited, mask = perspective_to_equirect(
            equirect_image,
            perspective_image,
            yaw_deg=projection_data["yaw"],
            pitch_deg=projection_data["pitch"],
            fov_deg=projection_data["fov"],
            persp_w=projection_data["persp_w"],
            persp_h=projection_data["persp_h"],
            blend_pixels=blend_pixels,
        )

        print(f"[DiT360Plus] Composited perspective back to equirect: "
              f"yaw={projection_data['yaw']:.1f}° pitch={projection_data['pitch']:.1f}° "
              f"blend={blend_pixels}px")

        return (composited, mask)


NODE_CLASS_MAPPINGS = {
    "EquirectToPersp": EquirectToPersp,
    "PerspToEquirect": PerspToEquirect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EquirectToPersp": "Equirect → Perspective",
    "PerspToEquirect": "Perspective → Equirect",
}
