"""Equirectangular ↔ perspective (rectilinear) projection utilities.

All coordinate math uses NumPy. Pixel remapping uses torch grid_sample for
GPU-accelerated bilinear interpolation.

Conventions:
  - Equirectangular: width = 360°, height = 180°
    pixel (0, 0) = top-left = (lon=-π, lat=+π/2)
    pixel (W-1, H-1) = bottom-right = (lon=+π, lat=-π/2)
  - Yaw: rotation around Y axis (positive = look right), degrees
  - Pitch: rotation around X axis (positive = look up), degrees
  - FOV: horizontal field of view in degrees
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build a 3×3 rotation matrix: pitch (X) then yaw (Y).

    Returns (3, 3) float64 array.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    # R = Ry(yaw) @ Rx(pitch)
    R = np.array([
        [cy,   sy * sp,  sy * cp],
        [0.0,  cp,       -sp],
        [-sy,  cy * sp,  cy * cp],
    ], dtype=np.float64)
    return R


def _build_perspective_rays(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Generate unit-length ray directions for a pinhole camera.

    Args:
        width, height: Output perspective image dimensions.
        fov_deg: Horizontal field of view in degrees.

    Returns:
        (H, W, 3) float64 array of unit vectors. Camera looks along +Z,
        +X = right, +Y = up.
    """
    fov_rad = math.radians(fov_deg)
    focal = width / (2.0 * math.tan(fov_rad / 2.0))

    # Pixel centres
    u = np.arange(width, dtype=np.float64) + 0.5 - width / 2.0
    v = np.arange(height, dtype=np.float64) + 0.5 - height / 2.0

    uu, vv = np.meshgrid(u, v, indexing="xy")

    # Camera coords: +X right, +Y up, +Z forward
    x = uu
    y = -vv  # negate so +Y = up
    z = np.full_like(uu, focal)

    # Normalise to unit vectors
    norm = np.sqrt(x * x + y * y + z * z)
    rays = np.stack([x / norm, y / norm, z / norm], axis=-1)  # (H, W, 3)
    return rays


def _rays_to_equirect_uv(rays: np.ndarray) -> np.ndarray:
    """Convert 3D ray directions to equirectangular UV coords in [-1, 1].

    Args:
        rays: (..., 3) array of unit vectors.

    Returns:
        (..., 2) array with (u, v) in [-1, 1] for grid_sample.
        u=-1 → lon=-π (left edge), u=+1 → lon=+π (right edge)
        v=-1 → lat=+π/2 (top edge), v=+1 → lat=-π/2 (bottom edge)
    """
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]

    lon = np.arctan2(x, z)  # [-π, π]
    lat = np.arcsin(np.clip(y, -1.0, 1.0))  # [-π/2, π/2]

    u = lon / math.pi          # [-1, 1]
    v = -lat / (math.pi / 2)   # [-1, 1], negated: top=-1, bottom=+1

    return np.stack([u, v], axis=-1)


def equirect_to_perspective(
    equirect: torch.Tensor,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> torch.Tensor:
    """Extract a perspective view from an equirectangular panorama.

    Args:
        equirect: (B, H, W, C) float32 [0,1] — ComfyUI image format.
        yaw_deg: Horizontal look direction in degrees (-180 to 180).
        pitch_deg: Vertical look direction in degrees (-90 to 90).
        fov_deg: Horizontal field of view in degrees.
        out_w, out_h: Output perspective image dimensions.

    Returns:
        (B, out_h, out_w, C) float32 tensor — perspective view.
    """
    B, H, W, C = equirect.shape
    device = equirect.device

    # 1. Build rays for the perspective camera and rotate
    rays = _build_perspective_rays(out_w, out_h, fov_deg)  # (out_h, out_w, 3)
    R = _rotation_matrix(yaw_deg, pitch_deg)               # (3, 3)

    # Rotate rays: (out_h, out_w, 3) @ (3, 3)^T
    rotated = rays @ R.T

    # 2. Convert rotated rays to equirect UV
    uv = _rays_to_equirect_uv(rotated)  # (out_h, out_w, 2)

    # 3. Circular-pad the equirect image horizontally to handle seam
    pad_w = max(4, W // 50)
    # Permute to (B, C, H, W) for padding
    eq_bchw = equirect.permute(0, 3, 1, 2)
    # Circular pad: left=pad_w cols from right, right=pad_w cols from left
    eq_padded = torch.cat([
        eq_bchw[:, :, :, -pad_w:],
        eq_bchw,
        eq_bchw[:, :, :, :pad_w],
    ], dim=3)  # (B, C, H, W + 2*pad_w)

    # Adjust UV to account for padding
    padded_W = W + 2 * pad_w
    # Original u in [-1, 1] maps to pixel range [0, W-1] within the padded image
    # at offset pad_w. We need to remap to [-1, 1] for the padded image.
    uv_torch = torch.from_numpy(uv).float().to(device)  # (out_h, out_w, 2)
    # u: [-1,1] → pixel [0, W-1] → offset by pad_w → normalise to [-1,1] in padded
    u_pixel = (uv_torch[..., 0] + 1.0) / 2.0 * (W - 1)  # [0, W-1]
    u_padded = (u_pixel + pad_w) / (padded_W - 1) * 2.0 - 1.0
    uv_padded = torch.stack([u_padded, uv_torch[..., 1]], dim=-1)

    # 4. grid_sample expects (B, C, H, W) input and (B, Hout, Wout, 2) grid
    grid = uv_padded.unsqueeze(0).expand(B, -1, -1, -1)  # (B, out_h, out_w, 2)
    result = F.grid_sample(
        eq_padded,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, out_h, out_w)

    # Back to ComfyUI format (B, H, W, C)
    return result.permute(0, 2, 3, 1)


def perspective_to_equirect(
    equirect: torch.Tensor,
    perspective: torch.Tensor,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    persp_w: int,
    persp_h: int,
    blend_pixels: int = 10,
) -> tuple:
    """Composite a perspective patch back into an equirectangular panorama.

    For each equirect pixel, computes whether it falls within the perspective
    view's frustum. If it does, samples the colour from the perspective image
    and blends it into the panorama.

    Args:
        equirect: (B, H, W, C) float32 — original panorama.
        perspective: (B, persp_h, persp_w, C) float32 — edited perspective patch.
        yaw_deg, pitch_deg, fov_deg: Same values used for extraction.
        persp_w, persp_h: Perspective image dimensions (must match extraction).
        blend_pixels: Edge feather width for smooth blending.

    Returns:
        (composited, mask) where:
          composited: (B, H, W, C) float32 — panorama with patch composited in.
          mask: (B, H, W) float32 — affected region mask (1 = fully replaced).
    """
    B, H, W, C = equirect.shape
    device = equirect.device

    # 1. For each equirect pixel, compute the 3D ray direction
    # Pixel centres
    u_eq = (np.arange(W, dtype=np.float64) + 0.5) / W  # [0, 1]
    v_eq = (np.arange(H, dtype=np.float64) + 0.5) / H  # [0, 1]
    uu, vv = np.meshgrid(u_eq, v_eq, indexing="xy")  # (H, W)

    lon = (uu * 2.0 - 1.0) * math.pi       # [-π, π]
    lat = (0.5 - vv) * math.pi              # [+π/2, -π/2]  (top=+π/2)

    # 3D unit vectors
    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)
    rays = np.stack([x, y, z], axis=-1)  # (H, W, 3)

    # 2. Inverse-rotate rays into camera space
    R = _rotation_matrix(yaw_deg, pitch_deg)
    R_inv = R.T  # Orthogonal matrix: inverse = transpose
    cam_rays = rays @ R_inv.T  # (H, W, 3) — rays in camera coordinate system

    # 3. Pinhole projection: project onto camera image plane
    fov_rad = math.radians(fov_deg)
    focal = persp_w / (2.0 * math.tan(fov_rad / 2.0))

    cam_x, cam_y, cam_z = cam_rays[..., 0], cam_rays[..., 1], cam_rays[..., 2]

    # Only pixels in front of camera (z > 0) can project
    in_front = cam_z > 1e-6

    # Project to pixel coords (with z safety to avoid div by 0)
    safe_z = np.where(in_front, cam_z, 1.0)
    px = cam_x / safe_z * focal + persp_w / 2.0   # pixel x
    py = -cam_y / safe_z * focal + persp_h / 2.0   # pixel y (negate Y)

    # 4. Check which pixels fall within perspective image bounds
    in_bounds = in_front & (px >= 0) & (px <= persp_w - 1) & (py >= 0) & (py <= persp_h - 1)

    # 5. Build blend mask with cosine feathering at edges
    if blend_pixels > 0:
        # Distance to nearest edge
        dist_left = px
        dist_right = (persp_w - 1) - px
        dist_top = py
        dist_bottom = (persp_h - 1) - py
        dist_to_edge = np.minimum(
            np.minimum(dist_left, dist_right),
            np.minimum(dist_top, dist_bottom),
        )
        # Cosine ramp: 0 at edge → 1 at blend_pixels distance
        alpha = np.where(
            in_bounds,
            np.clip(dist_to_edge / max(blend_pixels, 1), 0.0, 1.0),
            0.0,
        )
        # Cosine smoothing
        alpha = (1.0 - np.cos(alpha * math.pi)) / 2.0
    else:
        alpha = np.where(in_bounds, 1.0, 0.0)

    alpha = alpha.astype(np.float32)

    # 6. Normalise perspective pixel coords to [-1, 1] for grid_sample
    grid_u = np.where(in_bounds, px / (persp_w - 1) * 2.0 - 1.0, 0.0).astype(np.float32)
    grid_v = np.where(in_bounds, py / (persp_h - 1) * 2.0 - 1.0, 0.0).astype(np.float32)

    grid = torch.from_numpy(np.stack([grid_u, grid_v], axis=-1)).to(device)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    # 7. Sample from perspective image
    persp_bchw = perspective.permute(0, 3, 1, 2)  # (B, C, persp_h, persp_w)
    sampled = F.grid_sample(
        persp_bchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B, C, H, W)
    sampled = sampled.permute(0, 2, 3, 1)  # (B, H, W, C)

    # 8. Composite with blend mask
    alpha_t = torch.from_numpy(alpha).to(device)  # (H, W)
    alpha_t = alpha_t.unsqueeze(0).unsqueeze(-1).expand(B, H, W, C)  # (B, H, W, C)

    composited = equirect * (1.0 - alpha_t) + sampled * alpha_t

    # 9. Return mask as (B, H, W)
    mask = torch.from_numpy(alpha).to(device).unsqueeze(0).expand(B, -1, -1)

    return composited, mask
