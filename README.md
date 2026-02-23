# ComfyUI-DiT360Plus

ComfyUI custom nodes for **360-degree panoramic image generation** using [DiT360](https://github.com/Insta360-Research-Team/DiT360) with FLUX.1-dev. Supports text-to-panorama generation, inpainting, and outpainting.

Based on the research paper: [DiT360: Panoramic Image Generation](https://fenghora.github.io/DiT360-Page/)

## Features

- **Text-to-Panorama**: Generate seamless 360 equirectangular panoramas from text prompts
- **Inpainting**: Edit specific regions of existing panoramic images
- **Outpainting**: Extend panoramic image boundaries with new content
- **Smart VRAM Management**: Four offload modes (off/model/balanced/sequential) with dynamic GPU allocation
- **Edge Blending**: Post-process for seamless horizontal wraparound
- **360 Preview**: In-node panorama preview widget

## Nodes

### Pipeline

| Node | Description |
|------|-------------|
| **Flux Panorama Loader** | Loads any FLUX model (FLUX.1-dev, Kontext) with optional LoRA and configurable VRAM management |
| **DiT360 Text to Panorama** | Generates 360 panorama from a text prompt |
| **DiT360 Pipeline Unloader** | Frees GPU memory when done |

### Editing

| Node | Description |
|------|-------------|
| **Kontext Panorama Editor** | Inpaints panorama regions using FLUX Kontext (image + mask + prompt) |
| **DiT360 Image Inverter** | Inverts an image via RF-Inversion for editing |
| **DiT360 Panorama Editor** | Inpaints or outpaints using inverted latents + mask |

### Enhancement

| Node | Description |
|------|-------------|
| **360 Edge Blender** | Blends left/right edges for seamless wrap |
| **360 Empty Latent** | Creates 2:1 aspect ratio latent |
| **DiT360 Mask Processor** | Converts image/mask to binary editing mask |
| **360 Viewer** | Displays panorama preview in the node |

## Installation

### ComfyUI Manager (Recommended)

Search for `ComfyUI-DiT360Plus` in ComfyUI Manager and click Install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/thomashollier/ComfyUI-DiT360Plus.git
cd ComfyUI-DiT360Plus
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (for `scaled_dot_product_attention`)
- CUDA GPU with 16GB+ VRAM (24GB recommended for 2048 width)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) access on HuggingFace (requires `huggingface-cli login`)

The DiT360 LoRA weights are downloaded automatically from [Insta360-Research/DiT360-Panorama-Image-Generation](https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation).

## Usage

### Text to Panorama

The simplest workflow: generate a 360 panorama from text.

```
Flux Panorama Loader -> DiT360 Text to Panorama -> 360 Edge Blender -> 360 Viewer
```

1. Add **Flux Panorama Loader** — select your FLUX model, set base_pipeline to FLUX.1-dev, add the DiT360 LoRA ID, choose an offload mode
2. Add **DiT360 Text to Panorama** — enter your prompt (prefix with "This is a panorama image." for best results)
3. Add **360 Edge Blender** — ensures seamless horizontal wrapping
4. Add **360 Viewer** — preview the result

**Recommended settings:**
- Width: 2048 (height auto-calculated as 1024)
- Steps: 50
- Guidance scale: 2.8
- Seed: any

### Kontext Inpainting

Edit specific regions of a panorama using FLUX Kontext — no inversion step needed.

```
Flux Panorama Loader (Kontext) -> Kontext Panorama Editor -> Save Image / 360 Viewer
Load Image (panorama + painted mask) --^
```

1. Add **Flux Panorama Loader** — set base_pipeline to FLUX.1-Kontext-dev, leave LoRA empty
2. Load your panorama and paint a mask on it (white = area to edit)
3. **Kontext Panorama Editor** — enter your edit prompt, the masked region is repainted

### RF-Inversion Inpainting

Edit specific regions using the full RF-Inversion pipeline (more control, slower).

```
Flux Panorama Loader -> Image Inverter -----> Panorama Editor -> Edge Blender -> Viewer
Load Image (panorama) --^                       ^
Load Image (mask) -> Mask Processor ------------'
```

1. Load your panorama and a mask image (white = area to edit)
2. **DiT360 Image Inverter** encodes the source image into editable latent space
3. **DiT360 Mask Processor** converts the mask to the correct format
4. **DiT360 Panorama Editor** (mode: `inpaint`) regenerates the white mask region

### Outpainting

Extend a panoramic image with new content.

Same workflow as RF-Inversion inpainting, but:
- The mask should have white = existing content to keep, black = area to generate
- Set mode to `outpaint` in the Panorama Editor

### VRAM Management

The Flux Panorama Loader offers four CPU offload strategies:

| Mode | Peak VRAM | Speed | Best For |
|------|-----------|-------|----------|
| **off** | ~24GB+ | Fastest | Large GPUs (48GB+) |
| **model** | ~15GB | Fast | 1024 width on 24GB GPUs |
| **balanced** | Configurable | Medium | 1536-2048 width on 24GB GPUs |
| **sequential** | ~3GB | Slowest | Low VRAM GPUs (8-16GB) |

**Balanced mode** dynamically dispatches transformer layers to GPU after text encoding, maximizing GPU utilization based on available VRAM. The `balanced_offload_gb` parameter sets the maximum GPU budget — the system automatically reserves space for activations based on batch size and resolution.

Note: actual VRAM usage will be ~1GB above the set budget due to CUDA overhead, latent tensors, and prompt embeddings.

### Key Parameters

| Parameter | Node | Description | Recommended |
|-----------|------|-------------|-------------|
| `cpu_offload` | FluxPanoramaLoader | VRAM management strategy | `model` for 1024, `balanced` for 2048 |
| `balanced_offload_gb` | FluxPanoramaLoader | GPU budget for balanced mode (GB) | 8-16 |
| `guidance_scale` | TextToPanorama, Editor | Classifier-free guidance | 2.8 |
| `tau` | Editor | Source preservation strength (0-100) | 50 |
| `eta` | Editor | RF-Inversion guidance strength | 1.0 |
| `gamma` | Inverter | Inversion fidelity | 1.0 |
| `blend_width` | EdgeBlender | Edge blend width in pixels | 10-20 |
| `mask_blend_threshold` | Editor | Timestep threshold for mask blending | 0.5 |

## Example Workflows

Example workflows are included in the `examples/` folder:

- `text_to_panorama.json` — Basic text-to-panorama generation
- `inpainting_outpainting.json` — Inpainting/outpainting with mask

Load these via ComfyUI's **Load Workflow** button.

## How It Works

### Circular Padding

DiT360 achieves seamless 360 wrapping by adding **circular padding** to the packed latent sequence. After FLUX packs spatial latents into a token sequence `(B, N, D)`, the tokens are reshaped to a grid `(B, n_h, n_w, D)`, and the first/last columns are wrapped:

```
[last_col | original_cols | first_col]
```

This lets the transformer "see" across the horizontal seam during denoising. The padding is removed after denoising before VAE decoding.

### RF-Inversion (Editing)

Inpainting/outpainting uses [RF-Inversion](https://arxiv.org/abs/2410.10792) — a controlled forward/reverse ODE approach:

1. **Inversion**: The source image is encoded through a forward ODE to get invertible noise representations (batch size 1, ~50 transformer passes)
2. **Editing**: A controlled reverse ODE denoises with [PersonalizeAnything](https://github.com/Insta360-Research-Team/DiT360) attention processors that replace tokens at masked positions, preserving source content where specified (batch size 2, ~100 transformer passes)

### Dynamic VRAM Management (Balanced Mode)

Balanced mode uses a phased approach to maximize GPU utilization:

1. Text encoders load to GPU, encode prompt, offload to CPU (~10GB peak, brief)
2. System queries actual free VRAM, estimates activation memory for the current batch size and resolution
3. `accelerate.dispatch_model` packs as many transformer layers onto GPU as the budget allows
4. Denoising runs with optimal GPU utilization
5. Hooks removed, transformer back to CPU, VAE decodes

This avoids the VRAM spike that occurs when text encoders and transformer compete for GPU space simultaneously.

## Credits

- [DiT360](https://github.com/Insta360-Research-Team/DiT360) by Insta360 Research Team
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) by Black Forest Labs
- [RF-Inversion](https://arxiv.org/abs/2410.10792) by Rout et al.
- [ComfyUI-DiT360](https://github.com/cedarconnor/ComfyUI-DiT360) by cedarconnor (reference implementation)

## License

Apache-2.0
