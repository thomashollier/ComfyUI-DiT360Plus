"""DiT360 RF-Inversion pipeline for inpainting and outpainting.

Based on: https://github.com/Insta360-Research-Team/DiT360/blob/main/pa_src/pipeline.py
Uses RF-Inversion (https://arxiv.org/pdf/2410.10792) for controlled image editing.
"""

import torch
import numpy as np
from typing import Optional, List
from PIL import Image

from .attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor


def _should_offload(pipe):
    """Check if we need manual component offloading."""
    return getattr(pipe, '_dit360_offload', 'off') != 'off'


def _move_to(pipe, component_name, device):
    """Move a pipeline component to a device."""
    comp = getattr(pipe, component_name, None)
    if comp is not None:
        comp.to(device)


def _free_vram():
    """Empty CUDA cache."""
    torch.cuda.empty_cache()


def _dispatch_balanced(pipe, batch_size=1, height=1024, width=2048):
    """Dynamically dispatch transformer across GPU/CPU for balanced mode.

    Queries actual free VRAM and packs as many transformer layers as possible,
    reserving space for activations based on batch size and resolution.
    The user's balanced_offload_gb acts as a hard cap.
    """
    from accelerate import infer_auto_device_map, dispatch_model

    user_cap_gb = getattr(pipe, '_dit360_balanced_gb', 8.0)
    no_split = getattr(pipe, '_dit360_no_split', [])

    # Query actual free VRAM right now (text encoders already off GPU)
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)

    # Estimate activation memory: batch_size × sequence_length × hidden_dim × overhead
    seq_len = (height // 16) * (width // 16 + 2)  # with circular padding
    # ~50 bytes per token per layer active at once (conservative estimate for intermediates)
    activation_gb = (batch_size * seq_len * 3072 * 50) / (1024 ** 3)
    activation_gb = max(activation_gb, 2.0)  # at least 2GB reserve

    # Budget = free VRAM minus activation reserve, capped by user setting
    budget_gb = min(free_gb - activation_gb, user_cap_gb)
    budget_gb = max(budget_gb, 2.0)  # always at least 2GB for transformer

    device_map = infer_auto_device_map(
        pipe.transformer,
        max_memory={0: f"{budget_gb}GiB", "cpu": "128GiB"},
        no_split_module_classes=no_split,
    )
    dispatch_model(pipe.transformer, device_map)

    n_gpu = sum(1 for v in device_map.values() if str(v) in ("0", "cuda:0"))
    n_cpu = sum(1 for v in device_map.values() if str(v) == "cpu")
    print(f"[DiT360Plus] Balanced dispatch: {n_gpu} groups on GPU, {n_cpu} on CPU "
          f"({budget_gb:.1f}GB used, {activation_gb:.1f}GB reserved for activations, "
          f"batch={batch_size}, {width}x{height})")


def _teardown_balanced(pipe):
    """Remove dispatch hooks and move transformer back to CPU."""
    from accelerate.hooks import remove_hook_from_submodules
    remove_hook_from_submodules(pipe.transformer)
    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()


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
    device = torch.device("cuda")
    dtype = pipe.transformer.dtype
    offload = _should_offload(pipe)

    num_channels_latents = pipe.transformer.config.in_channels // 4

    # --- Step 1: Encode image (VAE on GPU) ---
    if offload:
        _move_to(pipe, 'vae', device)

    image_processed = pipe.image_processor.preprocess(
        image=image, height=height, width=width
    )
    image_processed = image_processed.to(dtype).to(device)
    x0 = pipe.vae.encode(image_processed).latent_dist.sample()
    x0 = (x0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    x0 = x0.to(dtype)

    if offload:
        _move_to(pipe, 'vae', "cpu")
        _free_vram()

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

    # --- Step 2: Encode prompt (text encoders on GPU) ---
    if offload:
        _move_to(pipe, 'text_encoder', device)
        _move_to(pipe, 'text_encoder_2', device)

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=source_prompt,
        prompt_2=source_prompt,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    text_ids = text_ids.to(device)

    if offload:
        _move_to(pipe, 'text_encoder', "cpu")
        _move_to(pipe, 'text_encoder_2', "cpu")
        _free_vram()

    # Guidance
    if pipe.transformer.config.guidance_embeds:
        guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    # --- Step 3: Forward ODE (transformer on GPU) ---
    offload_mode = getattr(pipe, '_dit360_offload', 'off')
    if offload_mode == 'balanced':
        _dispatch_balanced(pipe, batch_size=1, height=height, width=width)
    elif offload_mode == 'model':
        _move_to(pipe, 'transformer', device)
    # 'sequential': accelerate hooks already in place from loader

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

    if offload_mode == 'balanced':
        _teardown_balanced(pipe)
    elif offload_mode == 'model':
        _move_to(pipe, 'transformer', "cpu")
        _free_vram()
    elif offload_mode == 'sequential':
        _free_vram()

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
        eta: Edit strength at mask=0 positions (0=pure transformer, 1=reconstruct original).
             Preserve positions (mask=1) always use eta=1.0 regardless.
        num_inference_steps: Denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed.
        start_timestep: Start of eta guidance window (fraction).
        stop_timestep: End of eta guidance window (fraction).
        callback: Optional progress callback(step, total).

    Returns:
        PIL Image of the edited panorama.
    """
    device = torch.device("cuda")
    dtype = pipe.transformer.dtype
    offload = _should_offload(pipe)

    inverted_latents = inverted_data["inverted_latents"].to(device)
    image_latents = inverted_data["image_latents"].to(device)
    latent_image_ids = inverted_data["latent_image_ids"].to(device)
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

    # --- Step 1: Encode prompts (text encoders on GPU) ---
    if offload:
        _move_to(pipe, 'text_encoder', device)
        _move_to(pipe, 'text_encoder_2', device)

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[source_prompt, edit_prompt],
        prompt_2=[source_prompt, edit_prompt],
        device=device,
    )
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    text_ids = text_ids.to(device)

    if offload:
        _move_to(pipe, 'text_encoder', "cpu")
        _move_to(pipe, 'text_encoder_2', "cpu")
        _free_vram()

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

    # --- Step 2: Denoise (transformer on GPU) ---
    offload_mode = getattr(pipe, '_dit360_offload', 'off')
    if offload_mode == 'balanced':
        _dispatch_balanced(pipe, batch_size=2, height=height, width=width)
    elif offload_mode == 'model':
        _move_to(pipe, 'transformer', device)
    # 'sequential': accelerate hooks already in place from loader

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

        # RF-Inversion controlled denoising with per-token eta.
        # Source branch (latents[0]) always reconstructs (eta=1.0 everywhere).
        # Edit branch (latents[1]) uses eta=1.0 at preserve positions (mask=1)
        # and user eta at edit positions (mask=0), so edits only happen where masked.
        v_t = -noise_pred
        v_t_cond = (y_0 - latents) / (1 - t_i + 1e-8)
        eta_t = eta if start_step <= i < stop_step else 0.0
        if mask is not None:
            mask_flat = mask.to(device=device, dtype=dtype).view(1, -1, 1)  # (1, n_tokens, 1)
            # preserve positions (mask=1) → always eta=1.0, edit positions (mask=0) → user eta
            eta_edit = mask_flat + (1 - mask_flat) * eta_t
            eta_map = torch.cat([torch.ones_like(eta_edit), eta_edit], dim=0)  # (2, n_tokens, 1)
            v_hat_t = v_t + eta_map * (v_t_cond - v_t)
        else:
            v_hat_t = v_t + eta_t * (v_t_cond - v_t)
        latents = latents + v_hat_t * (scheduler_sigmas[i] - scheduler_sigmas[i + 1])

        # Hard latent blending: copy source latents into edit branch at preserve positions.
        # Applied every step to prevent any drift in preserved regions.
        if mask is not None:
            flat_mask = mask.to(device=device, dtype=dtype).view(-1, 1)
            latents[1] = latents[1] * (1 - flat_mask) + latents[0] * flat_mask

        if callback is not None:
            callback(i + 1, len(timesteps))

    if offload_mode == 'balanced':
        _teardown_balanced(pipe)
    elif offload_mode == 'model':
        _move_to(pipe, 'transformer', "cpu")
        _free_vram()
    elif offload_mode == 'sequential':
        _free_vram()

    # Restore default attention processors
    from diffusers.models.attention_processor import FluxAttnProcessor2_0
    pipe.transformer.set_attn_processor(FluxAttnProcessor2_0())

    # Remove circular padding
    latents = _remove_circular_padding(latents, n_h, n_w)

    # Take the edit branch result (index 1)
    edit_latents = latents[1:2]

    # --- Step 3: Decode (VAE on GPU) ---
    if offload:
        _move_to(pipe, 'vae', device)

    edit_latents = _unpack_latents(edit_latents, height, width, pipe.vae_scale_factor)
    edit_latents = (edit_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(edit_latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")

    if offload:
        _move_to(pipe, 'vae', "cpu")
        _free_vram()

    return image[0]
