"""Equirectangular projection utilities for panoramic images."""

import torch
import math
from typing import Tuple, Literal


def validate_aspect_ratio(width: int, height: int, tolerance: float = 0.01) -> bool:
    """Check if dimensions have a 2:1 ratio (equirectangular requirement)."""
    ratio = width / height
    return abs(ratio - 2.0) < tolerance


def get_equirect_dimensions(width: int, alignment: int = 16) -> Tuple[int, int]:
    """Calculate valid equirectangular dimensions with proper alignment."""
    width = (width // alignment) * alignment
    height = width // 2
    return width, height


def blend_edges(
    image: torch.Tensor,
    blend_width: int = 10,
    mode: Literal["linear", "cosine", "smooth"] = "cosine",
) -> torch.Tensor:
    """Blend left and right edges for seamless wraparound.

    Args:
        image: (B, H, W, C) ComfyUI format tensor.
        blend_width: Width of blend region in pixels.
        mode: Blending curve type.
    """
    B, H, W, C = image.shape

    if blend_width <= 0 or blend_width >= W // 2:
        return image

    left_edge = image[:, :, :blend_width, :]
    right_edge = image[:, :, -blend_width:, :]

    if mode == "linear":
        weights = torch.linspace(0, 1, blend_width, device=image.device)
    elif mode == "cosine":
        t = torch.linspace(0, math.pi, blend_width, device=image.device)
        weights = (1 - torch.cos(t)) / 2
    elif mode == "smooth":
        weights = torch.linspace(0, 1, blend_width, device=image.device) ** 2
    else:
        raise ValueError(f"Unknown blend mode: {mode}")

    weights = weights.view(1, 1, -1, 1)

    blended_left = left_edge * (1 - weights) + right_edge * weights
    blended_right = right_edge * (1 - weights) + left_edge * weights

    result = image.clone()
    result[:, :, :blend_width, :] = blended_left
    result[:, :, -blend_width:, :] = blended_right

    return result


def check_edge_continuity(image: torch.Tensor, threshold: float = 0.05) -> bool:
    """Check if left and right edges are continuous."""
    left_edge = image[:, :, 0, :]
    right_edge = image[:, :, -1, :]
    diff = torch.abs(left_edge - right_edge).mean()
    return diff.item() < threshold
