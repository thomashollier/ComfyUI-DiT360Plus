"""DiT360 text-to-panorama generation.

Wraps a standard FluxPipeline to add circular padding on packed latents,
matching the approach from: https://github.com/Insta360-Research-Team/DiT360/blob/main/src/pipeline.py
"""

import torch
import numpy as np
from typing import Optional, Union, List
from diffusers import FluxPipeline


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
          f"batch={batch_size}, {width}x{height//1})")


def _teardown_balanced(pipe):
    """Remove dispatch hooks and move transformer back to CPU."""
    from accelerate.hooks import remove_hook_from_submodules
    remove_hook_from_submodules(pipe.transformer)
    pipe.transformer.to("cpu")
    torch.cuda.empty_cache()


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
    device = torch.device("cuda")
    dtype = pipe.transformer.dtype
    offload = _should_offload(pipe)

    # --- Step 1: Encode prompt (text encoders on GPU) ---
    if offload:
        _move_to(pipe, 'text_encoder', device)
        _move_to(pipe, 'text_encoder_2', device)

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    text_ids = text_ids.to(device)

    if offload:
        _move_to(pipe, 'text_encoder', "cpu")
        _move_to(pipe, 'text_encoder_2', "cpu")
        _free_vram()

    # --- Step 2: Prepare latents ---
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

    # --- Step 3: Denoise (transformer on GPU) ---
    offload_mode = getattr(pipe, '_dit360_offload', 'off')
    if offload_mode == 'balanced':
        _dispatch_balanced(pipe, batch_size=1, height=height, width=width)
    elif offload_mode == 'model':
        _move_to(pipe, 'transformer', device)
    # 'sequential': accelerate hooks already in place from loader

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

    if offload_mode == 'balanced':
        _teardown_balanced(pipe)
    elif offload_mode == 'model':
        _move_to(pipe, 'transformer', "cpu")
        _free_vram()
    elif offload_mode == 'sequential':
        _free_vram()

    # --- Remove circular padding ---
    latents = latents.reshape(bsz, n_h, n_w + 2, dim)
    latents = latents[:, :, 1:-1, :]
    latents = latents.reshape(bsz, -1, dim)

    # Unpack latents
    latents = _unpack_latents(latents, height, width, pipe.vae_scale_factor)

    # --- Step 4: Decode (VAE on GPU) ---
    if offload:
        _move_to(pipe, 'vae', device)

    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")

    if offload:
        _move_to(pipe, 'vae', "cpu")
        _free_vram()

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
