# ComfyUI-DiT360Plus

## Project Overview

ComfyUI custom nodes for 360-degree panoramic image generation using FLUX.1-dev with the DiT360 LoRA adapter. Supports text-to-panorama, inpainting, and outpainting.

**Owner**: Thomas Hollier (thomashollier)
**License**: Apache-2.0

## Architecture

### Node Modules

| File | Nodes | Purpose |
|------|-------|---------|
| `nodes_pipeline.py` | PipelineLoader, TextToPanorama, PipelineUnloader | Model loading and text-to-pano generation |
| `nodes_editing.py` | ImageInverter, PanoramaEditor | RF-Inversion based inpainting/outpainting |
| `nodes_enhancement.py` | EdgeBlender, EmptyLatent, MaskProcessor, Viewer | Post-processing and utilities |

### Pipeline Code (`pipeline/`)

| File | Purpose |
|------|---------|
| `dit360_txt2pano.py` | Text-to-panorama with circular padding on packed latents |
| `dit360_inversion.py` | RF-Inversion (forward/reverse ODE) for image editing |
| `attn_processor.py` | PersonalizeAnything attention processor for masked editing |

### Utilities (`utils/`)

| File | Purpose |
|------|---------|
| `equirect.py` | Equirectangular projection helpers (aspect ratio, edge blending) |
| `masks.py` | Mask creation and preparation for the editing pipeline |

### Frontend (`web/js/`)

| File | Purpose |
|------|---------|
| `equirect360_viewer.js` | 360 panorama preview widget for ComfyUI |

## Key Technical Concepts

### Circular Padding
After FLUX packs latents `(B,C,H,W)` → `(B, H/2*W/2, C*4)`, we reshape to a grid `(B, n_h, n_w, D)` and wrap the first/last columns: `[last_col | original | first_col]`. This lets the transformer "see" across the horizontal seam. Padding is removed after denoising, before VAE decode.

### VRAM Management — Four Offload Modes
We bypass diffusers' built-in `enable_model_cpu_offload()` because it conflicts with our direct `pipe.transformer()` calls. Instead we manage component placement manually:

- **off**: All on GPU. No movement.
- **model**: Whole components (text encoders, transformer, VAE) moved between CPU/GPU as needed. Fast, good for 1024 width.
- **balanced**: Dynamic dispatch via `accelerate.infer_auto_device_map` + `dispatch_model`. Fires AFTER text encoding frees GPU, so maximum VRAM is available for transformer layers. Queries actual free VRAM, reserves space for activations scaled by batch size and resolution. User's `balanced_offload_gb` acts as a cap. Torn down after denoising.
- **sequential**: `accelerate.cpu_offload` hooks on the transformer, set at load time. One layer at a time. Slowest but lowest VRAM.

### RF-Inversion (Editing)
Based on https://arxiv.org/abs/2410.10792. Two phases:
1. **Inversion** (`invert_image`): Forward ODE encodes source image to noise space. Batch size 1.
2. **Editing** (`edit_panorama`): Reverse ODE denoises with PersonalizeAnything attention that swaps tokens between source/edit branches at masked positions. Batch size 2.

### Mask Convention
Internal: `mask=1` = preserve from source, `mask=0` = generate new content.
- Inpainting: user paints white on areas to EDIT → mask is inverted internally
- Outpainting: user paints white on areas to KEEP → used directly

## Custom Types

- `DIT360_PIPELINE` — FluxPipeline with DiT360 LoRA, tagged with `_dit360_offload`, `_dit360_balanced_gb`, `_dit360_no_split`
- `DIT360_INVERTED` — Dict with `inverted_latents`, `image_latents`, `latent_image_ids`, `height`, `width`

## Development Notes

### Adding New Nodes
1. Create node class with `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`
2. Add to `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in the module
3. Import and merge in `__init__.py`

### ComfyUI Conventions
- Image tensors: `(B, H, W, C)` float32 [0,1]
- Mask tensors: `(B, H, W)` float32 [0,1]
- Latent tensors: `(B, C, H, W)` wrapped in `{"samples": tensor}`
- Use `comfy.utils.ProgressBar` for progress tracking
- Use `folder_paths` for model/output directory access

### FLUX-Specific
- Transformer: `FluxTransformer2DModel` — 19 double-stream + 38 single-stream blocks, ~12GB fp16
- Latent packing: `(B,C,H,W)` → `(B, H/2*W/2, C*4)` sequence format
- VAE scale factor: 8 (but packing makes effective 16 for token grid)
- Latent image IDs: `(n_h, n_w, 3)` positional encoding for RoPE

### Testing
- Requires CUDA GPU and FLUX.1-dev model
- Test text-to-pano at 1024 width with "model" offload first (fastest)
- Test at 2048 with "balanced" offload
- Editing pipeline needs an input image + mask

## Dependencies

- `diffusers>=0.25.0` — FluxPipeline, FluxTransformer2DModel
- `transformers>=4.28.1` — Text encoders (CLIP, T5-XXL)
- `accelerate>=0.26.0` — Device map dispatch, CPU offload hooks
- `huggingface-hub>=0.20.0` — Model downloads
- HuggingFace auth required for gated FLUX.1-dev repo
