# ComfyUI-DiT360Plus

ComfyUI custom nodes for **360-degree panoramic image generation** using [DiT360](https://github.com/Insta360-Research-Team/DiT360) with FLUX.1-dev. Supports text-to-panorama generation, inpainting, and outpainting.

Based on the research paper: [DiT360: Panoramic Image Generation](https://fenghora.github.io/DiT360-Page/)

## Features

- **Text-to-Panorama**: Generate seamless 360 equirectangular panoramas from text prompts
- **Inpainting**: Edit specific regions of existing panoramic images
- **Outpainting**: Extend panoramic image boundaries with new content
- **Edge Blending**: Post-process for seamless horizontal wraparound
- **360 Preview**: In-node panorama preview widget

## Nodes

### Pipeline

| Node | Description |
|------|-------------|
| **DiT360 Pipeline Loader** | Loads FLUX.1-dev + DiT360 LoRA adapter |
| **DiT360 Text to Panorama** | Generates 360 panorama from a text prompt |
| **DiT360 Pipeline Unloader** | Frees GPU memory when done |

### Editing

| Node | Description |
|------|-------------|
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
git clone https://github.com/your-repo/ComfyUI-DiT360Plus.git
cd ComfyUI-DiT360Plus
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ (for `scaled_dot_product_attention`)
- CUDA GPU with 24GB+ VRAM recommended (16GB possible with float16 + VAE tiling)
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) access on HuggingFace

The DiT360 LoRA weights are downloaded automatically from [Insta360-Research/DiT360-Panorama-Image-Generation](https://huggingface.co/Insta360-Research/DiT360-Panorama-Image-Generation).

## Usage

### Text to Panorama

The simplest workflow: generate a 360 panorama from text.

```
DiT360 Pipeline Loader -> DiT360 Text to Panorama -> 360 Edge Blender -> 360 Viewer
```

1. Add **DiT360 Pipeline Loader** - loads FLUX.1-dev + DiT360 LoRA
2. Add **DiT360 Text to Panorama** - enter your prompt (prefix with "This is a panorama image." for best results)
3. Add **360 Edge Blender** - ensures seamless horizontal wrapping
4. Add **360 Viewer** - preview the result

**Recommended settings:**
- Width: 2048 (height auto-calculated as 1024)
- Steps: 50
- Guidance scale: 2.8
- Seed: any

### Inpainting

Edit specific regions within an existing panorama.

```
Pipeline Loader -----> Image Inverter -----> Panorama Editor -> Edge Blender -> Viewer
Load Image (panorama) --^                       ^
Load Image (mask) -> Mask Processor ------------'
```

1. Load your panorama and a mask image (white = area to edit)
2. **DiT360 Image Inverter** encodes the source image into editable latent space
3. **DiT360 Mask Processor** converts the mask to the correct format
4. **DiT360 Panorama Editor** (mode: `inpaint`) regenerates the white mask region

### Outpainting

Extend a panoramic image with new content.

Same workflow as inpainting, but:
- The mask should have white = existing content to keep, black = area to generate
- Set mode to `outpaint` in the Panorama Editor

### Key Parameters

| Parameter | Node | Description | Recommended |
|-----------|------|-------------|-------------|
| `guidance_scale` | TextToPanorama, Editor | Classifier-free guidance | 2.8 |
| `tau` | Editor | Source preservation strength (0-100) | 50 |
| `eta` | Editor | RF-Inversion guidance strength | 1.0 |
| `gamma` | Inverter | Inversion fidelity | 1.0 |
| `blend_width` | EdgeBlender | Edge blend width in pixels | 10-20 |
| `mask_blend_threshold` | Editor | Timestep threshold for mask blending | 0.5 |

## Example Workflows

Example workflows are included in the `examples/` folder:

- `text_to_panorama.json` - Basic text-to-panorama generation
- `inpainting_outpainting.json` - Inpainting/outpainting with mask

Load these via ComfyUI's **Load Workflow** button.

## How It Works

### Circular Padding

DiT360 achieves seamless 360 wrapping by adding **circular padding** to the packed latent sequence. After FLUX packs spatial latents into a token sequence `(B, N, D)`, the tokens are reshaped to a grid `(B, n_h, n_w, D)`, and the first/last columns are wrapped:

```
[last_col | original_cols | first_col]
```

This lets the transformer "see" across the horizontal seam during denoising. The padding is removed after denoising before VAE decoding.

### RF-Inversion (Editing)

Inpainting/outpainting uses [RF-Inversion](https://arxiv.org/abs/2410.10792) - a controlled forward/reverse ODE approach:

1. **Inversion**: The source image is encoded through a forward ODE to get invertible noise representations
2. **Editing**: A controlled reverse ODE denoises with [PersonalizeAnything](https://github.com/Insta360-Research-Team/DiT360) attention processors that replace tokens at masked positions, preserving source content where specified

## Credits

- [DiT360](https://github.com/Insta360-Research-Team/DiT360) by Insta360 Research Team
- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) by Black Forest Labs
- [RF-Inversion](https://arxiv.org/abs/2410.10792) by Rout et al.
- [ComfyUI-DiT360](https://github.com/cedarconnor/ComfyUI-DiT360) by cedarconnor (reference implementation)

## License

Apache-2.0
