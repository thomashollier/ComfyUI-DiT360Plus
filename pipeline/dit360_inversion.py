"""DiT360 RF-Inversion pipeline for inpainting and outpainting.

Based on: https://github.com/Insta360-Research-Team/DiT360/blob/main/pa_src/pipeline.py
Uses RF-Inversion (https://arxiv.org/pdf/2410.10792) for controlled image editing.
"""

import torch
import numpy as np
from typing import Optional, List
from PIL import Image

from .attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    """Pack spatial latents into FLUX sequence format."""
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def _unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack FLUX sequence format to spatial latents."""
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents


def _prepare_latent_image_ids(height, width, device, dtype):
    """Create position IDs for FLUX latent tokens."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def _add_circular_padding(latents, latent_image_ids, n_h, n_w):
    """Add circular padding to packed latents and image IDs."""
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

    return latents, latent_image_ids


def _remove_circular_padding(latents, n_h, n_w):
    """Remove circular padding from packed latents."""
    bsz, _, dim = latents.shape
    latents = latents.reshape(bsz, n_h, n_w + 2, dim)
    latents = latents[:, :, 1:-1, :]
    latents = latents.reshape(bsz, -1, dim)
    return latents


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def invert_image(
    pipe,
    image: Image.Image,
    height: int = 1024,
    width: int = 2048,
    source_prompt: str = "",
    num_inversion_steps: int = 50,
    gamma: float = 1.0,
    source_guidance_scale: float = 0.0,
    callback=None,
):
    """Invert an image using RF-Inversion for subsequent editing.

    Performs Algorithm 1 (Controlled Forward ODE) from https://arxiv.org/pdf/2410.10792

    Args:
        pipe: FluxPipeline with DiT360 LoRA.
        image: Input PIL image to invert.
        height: Image height.
        width: Image width.
        source_prompt: Description of source image (can be empty).
        num_inversion_steps: Number of inversion steps.
        gamma: Inversion fidelity (higher = more faithful).
        source_guidance_scale: Guidance for forward ODE (usually 0).
        callback: Optional progress callback(step, total).

    Returns:
        dict with keys: inverted_latents, image_latents, latent_image_ids, height, width
    """
    device = pipe.device
    dtype = pipe.text_encoder.dtype if pipe.text_encoder is not None else pipe.transformer.dtype

    num_channels_latents = pipe.transformer.config.in_channels // 4

    # Encode image
    image_processed = pipe.image_processor.preprocess(
        image=image, height=height, width=width
    )
    image_processed = image_processed.to(dtype).to(device)
    x0 = pipe.vae.encode(image_processed).latent_dist.sample()
    x0 = (x0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    x0 = x0.to(dtype)

    # Pack latents
    lat_h = height // pipe.vae_scale_factor
    lat_w = width // pipe.vae_scale_factor
    image_latents = _pack_latents(x0, 1, num_channels_latents, lat_h, lat_w)
    latent_image_ids = _prepare_latent_image_ids(lat_h // 2, lat_w // 2, device, dtype)

    # Add circular padding
    n_h = height // 16
    n_w = width // 16
    image_latents, latent_image_ids = _add_circular_padding(
        image_latents, latent_image_ids, n_h, n_w
    )

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inversion_steps, num_inversion_steps)
    image_seq_len = (height // pipe.vae_scale_factor // 2) * (width // pipe.vae_scale_factor // 2)
    mu = _calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.16),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps
    scheduler_sigmas = pipe.scheduler.sigmas

    # Encode source prompt
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=source_prompt,
        prompt_2=source_prompt,
        device=device,
    )

    # Guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    # Forward ODE: invert from clean image to noise
    Y_t = image_latents.clone()
    y_1 = torch.randn(Y_t.shape, device=device, dtype=dtype)
    N = len(scheduler_sigmas)

    for i in range(N - 1):
        t_i = torch.tensor(i / N, dtype=dtype, device=device)
        timestep = t_i.repeat(1)

        u_t_i = pipe.transformer(
            hidden_states=Y_t,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        u_t_i_cond = (y_1 - Y_t) / (1 - t_i)
        u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
        Y_t = Y_t + u_hat_t_i * (scheduler_sigmas[i] - scheduler_sigmas[i + 1])

        if callback is not None:
            callback(i + 1, N - 1)

    return {
        "inverted_latents": Y_t,
        "image_latents": image_latents,
        "latent_image_ids": latent_image_ids,
        "height": height,
        "width": width,
    }


def edit_panorama(
    pipe,
    inverted_data: dict,
    mask: torch.Tensor,
    source_prompt: str,
    edit_prompt: str,
    tau: float = 0.5,
    eta: float = 1.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 2.8,
    seed: int = 0,
    start_timestep: float = 0.0,
    stop_timestep: float = 0.99,
    mask_blend_threshold: float = 0.5,
    callback=None,
):
    """Edit a panorama using inverted latents and mask.

    Performs controlled reverse ODE with PersonalizeAnything attention.

    Args:
        pipe: FluxPipeline with DiT360 LoRA.
        inverted_data: Output from invert_image().
        mask: Flattened mask from prepare_mask_for_pipeline().
              Shape (n_tokens, 1). Value 1=preserve, 0=generate.
        source_prompt: Description of source image.
        edit_prompt: Description of desired edit result.
        tau: Attention replacement threshold (0-1). Higher = stronger preservation.
        eta: RF-Inversion guidance strength.
        num_inference_steps: Denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed.
        start_timestep: Start of eta guidance window (fraction).
        stop_timestep: End of eta guidance window (fraction).
        mask_blend_threshold: Timestep threshold for mask-based latent blending.
        callback: Optional progress callback(step, total).

    Returns:
        PIL Image of the edited panorama.
    """
    device = pipe.device
    dtype = pipe.text_encoder.dtype if pipe.text_encoder is not None else pipe.transformer.dtype

    inverted_latents = inverted_data["inverted_latents"]
    image_latents = inverted_data["image_latents"]
    latent_image_ids = inverted_data["latent_image_ids"]
    height = inverted_data["height"]
    width = inverted_data["width"]

    n_h = height // 16
    n_w = width // 16

    num_channels_latents = pipe.transformer.config.in_channels // 4
    img_dims = n_h * (n_w + 2)  # with circular padding

    # Set up PersonalizeAnything attention processors
    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=tau, mask=mask, device=device, img_dims=img_dims
        ),
    )

    # Encode prompts: [source_prompt, edit_prompt]
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[source_prompt, edit_prompt],
        prompt_2=[source_prompt, edit_prompt],
        device=device,
    )

    # Prepare latents: source inverted + new random noise
    generator = torch.Generator(device=device).manual_seed(seed)

    # Generate new random latents for the edit branch
    lat_h = 2 * (height // (pipe.vae_scale_factor * 2))
    lat_w = 2 * (width // (pipe.vae_scale_factor * 2))
    shape = (1, num_channels_latents, lat_h, lat_w)
    new_latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    new_latents = _pack_latents(new_latents, 1, num_channels_latents, lat_h, lat_w)

    # Add circular padding to new latents
    new_latents_padded, _ = _add_circular_padding(
        new_latents,
        _prepare_latent_image_ids(lat_h // 2, lat_w // 2, device, dtype),
        n_h, n_w,
    )

    # Stack: [source_inverted, edit_new]
    latents = torch.cat([inverted_latents, new_latents_padded], dim=0)
    bsz, _, dim = latents.shape

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = (height // pipe.vae_scale_factor // 2) * (width // pipe.vae_scale_factor // 2)
    mu = _calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.16),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps
    scheduler_sigmas = pipe.scheduler.sigmas

    # Apply strength to get correct timestep range
    start_step = int(start_timestep * num_inference_steps)
    stop_step = min(int(stop_timestep * num_inference_steps), num_inference_steps)

    # Guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    y_0 = image_latents.clone()

    # Denoising loop with RF-Inversion control
    for i, t in enumerate(timesteps):
        t_i = 1 - t / 1000

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # Pass timestep to attention processors for tau-based switching
        joint_attention_kwargs = {"timestep": timestep[0].item() / 1000}

        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]

        # RF-Inversion controlled denoising (applied to ALL branches)
        # y_0 broadcasts from (1, N, D) to (2, N, D) across source + edit batches
        v_t = -noise_pred
        v_t_cond = (y_0 - latents) / (1 - t_i + 1e-8)
        eta_t = eta if start_step <= i < stop_step else 0.0
        v_hat_t = v_t + eta_t * (v_t_cond - v_t)
        latents = latents + v_hat_t * (scheduler_sigmas[i] - scheduler_sigmas[i + 1])

        # Mask-based latent blending: enforce consistency in preserved regions
        if mask is not None and timestep[0].item() / 1000 >= mask_blend_threshold:
            flat_mask = mask.to(device).float().view(-1, 1)
            latents[1] = latents[1] * (1.0 - flat_mask) + latents[0] * flat_mask

        if callback is not None:
            callback(i + 1, len(timesteps))

    # Remove circular padding
    latents = _remove_circular_padding(latents, n_h, n_w)

    # Take the edit branch result (index 1)
    edit_latents = latents[1:2]

    # Unpack and decode
    edit_latents = _unpack_latents(edit_latents, height, width, pipe.vae_scale_factor)
    edit_latents = (edit_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(edit_latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")

    # Restore default attention processors
    from diffusers.models.attention_processor import FluxAttnProcessor2_0
    pipe.transformer.set_attn_processor(FluxAttnProcessor2_0())

    return image[0]


def _unpack_latents(latents, height, width, vae_scale_factor):
    """Unpack FLUX latents from sequence format to spatial format."""
    batch_size, num_patches, channels = latents.shape
    height = height // vae_scale_factor
    width = width // vae_scale_factor
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
    return latents
