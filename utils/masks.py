"""Mask creation and processing utilities for DiT360 inpainting/outpainting."""

import torch
import numpy as np
from PIL import Image


def create_mask_for_editing(
    mask_tensor: torch.Tensor,
    mode: str,
    latent_w: int,
    latent_h: int,
) -> torch.Tensor:
    """Create a properly formatted mask for DiT360 editing pipeline.

    In DiT360's convention:
      mask=1 -> preserve from source image
      mask=0 -> generate new content

    Args:
        mask_tensor: Input mask (B, H, W) or (B, C, H, W) or (H, W).
                     White (1.0) = area of interest.
        mode: "inpaint" or "outpaint".
              - inpaint: white area = region to EDIT (we invert so white->0 means generate)
              - outpaint: white area = existing content to KEEP (white->1 means preserve)
        latent_w: Target latent width.
        latent_h: Target latent height.

    Returns:
        Mask tensor of shape (latent_h, latent_w) with values 0 or 1.
    """
    # Normalize to 2D
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor[:, 0, :, :]
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]
    # Now (H, W)

    # Resize to latent dimensions using nearest neighbor
    mask_pil = Image.fromarray(
        (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    )
    mask_pil = mask_pil.resize((latent_w, latent_h), Image.Resampling.NEAREST)
    mask_np = np.array(mask_pil)
    mask = torch.tensor(np.where(mask_np > 127, 1, 0), dtype=torch.float32)

    if mode == "inpaint":
        # For inpainting: white input = edit area -> invert so preserved=1
        mask = 1 - mask
    # For outpaint: white input = keep area -> already correct (preserved=1)

    return mask


def prepare_mask_for_pipeline(
    mask: torch.Tensor,
    latent_w: int,
    latent_h: int,
) -> torch.Tensor:
    """Add circular padding to mask and flatten for attention processor.

    Matches the DiT360 editing.py approach:
      mask = torch.cat([mask[:, 0:1], mask, mask[:, -1:]], dim=-1).view(-1, 1)

    Args:
        mask: (latent_h, latent_w) mask tensor.
        latent_w: Latent width (before padding).
        latent_h: Latent height.

    Returns:
        Flattened mask of shape (latent_h * (latent_w + 2), 1).
    """
    # Reshape to (h, w) if needed
    if mask.ndim == 1:
        mask = mask.view(latent_h, latent_w)

    # Add circular padding: prepend last column, append first column
    first_col = mask[:, 0:1]
    last_col = mask[:, -1:]
    mask_padded = torch.cat([last_col, mask, first_col], dim=-1)

    # Flatten to (n_tokens, 1) for attention processor
    return mask_padded.view(-1, 1)
