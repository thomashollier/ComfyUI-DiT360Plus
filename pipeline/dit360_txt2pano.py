"""DiT360 text-to-panorama generation.

Wraps a standard FluxPipeline to add circular padding on packed latents,
matching the approach from: https://github.com/Insta360-Research-Team/DiT360/blob/main/src/pipeline.py
"""

import torch
import numpy as np
from typing import Optional, Union, List
from diffusers import FluxPipeline


def generate_panorama(
    pipe: FluxPipeline,
    prompt: str,
    height: int = 1024,
    width: int = 2048,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.8,
    seed: int = 0,
    callback=None,
) -> torch.Tensor:
    """Generate a 360 panorama using DiT360-augmented FluxPipeline.

    This reimplements the circular padding approach from DiT360's src/pipeline.py:
    - Pack latents into FLUX sequence format
    - Add circular padding (wrap first/last columns)
    - Same wrapping on latent_image_ids
    - After denoising, remove padding
    - Decode with VAE

    Args:
        pipe: FluxPipeline with DiT360 LoRA loaded.
        prompt: Text prompt for generation.
        height: Output height (default 1024).
        width: Output width (default 2048).
        num_inference_steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed.
        callback: Optional progress callback(step, total).

    Returns:
        PIL Image of the generated panorama.
    """
    device = pipe.device
    dtype = pipe.transformer.dtype

    # Encode prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
    )

    # Prepare latents
    num_channels_latents = pipe.transformer.config.in_channels // 4
    generator = torch.Generator(device=device).manual_seed(seed)

    latent_h = 2 * (height // (pipe.vae_scale_factor * 2))
    latent_w = 2 * (width // (pipe.vae_scale_factor * 2))

    shape = (1, num_channels_latents, latent_h, latent_w)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # Pack latents: (B, C, H, W) -> (B, H/2 * W/2, C*4)
    latents = latents.view(1, num_channels_latents, latent_h // 2, 2, latent_w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(1, (latent_h // 2) * (latent_w // 2), num_channels_latents * 4)

    # Prepare latent image IDs
    latent_image_ids = torch.zeros(latent_h // 2, latent_w // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(latent_h // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(latent_w // 2)[None, :]
    latent_image_ids = latent_image_ids.to(device=device, dtype=dtype)

    # --- Circular padding on packed latents ---
    n_h = height // 16
    n_w = width // 16
    bsz, _, dim = latents.shape

    latents = latents.reshape(bsz, n_h, n_w, dim)
    first_col = latents[:, :, 0:1, :]
    last_col = latents[:, :, -1:, :]
    latents = torch.cat([last_col, latents, first_col], dim=2)
    latents = latents.reshape(bsz, -1, dim)

    latent_image_ids = latent_image_ids.reshape(n_h, n_w, 3)
    first_col_ids = latent_image_ids[:, 0:1, :]
    last_col_ids = latent_image_ids[:, -1:, :]
    latent_image_ids = torch.cat([last_col_ids, latent_image_ids, first_col_ids], dim=1)
    latent_image_ids = latent_image_ids.reshape(-1, 3)

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = n_h * n_w  # Use unpadded sequence length for shift calculation

    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )

    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps

    # Handle guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # Denoising loop
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if callback is not None:
            callback(i + 1, len(timesteps))

    # --- Remove circular padding ---
    latents = latents.reshape(bsz, n_h, n_w + 2, dim)
    latents = latents[:, :, 1:-1, :]
    latents = latents.reshape(bsz, -1, dim)

    # Unpack latents
    latents = _unpack_latents(latents, height, width, pipe.vae_scale_factor)

    # Decode
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")

    return image[0]


def _unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack FLUX latents from sequence format to spatial format."""
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents
