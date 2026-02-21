"""
ComfyUI-DiT360Plus: 360 Panorama Generation with Inpainting & Outpainting

Nodes for generating and editing seamless 360-degree equirectangular panoramic
images using FLUX.1-dev with the DiT360 LoRA adapter.

PIPELINE NODES:
  - DiT360PipelineLoader     : Load FLUX + DiT360 LoRA
  - DiT360TextToPanorama     : Generate panorama from text
  - DiT360PipelineUnloader   : Free GPU memory

EDITING NODES:
  - DiT360ImageInverter      : Invert image for editing
  - DiT360PanoramaEditor     : Inpaint/outpaint panoramas

ENHANCEMENT NODES:
  - Equirect360EmptyLatent   : 2:1 aspect ratio latent helper
  - Equirect360EdgeBlender   : Edge blending for seamless wrap
  - DiT360MaskProcessor      : Mask preparation
  - Equirect360Viewer        : Interactive 360 preview

Based on: https://github.com/Insta360-Research-Team/DiT360
"""

import os

# Pipeline nodes
from .nodes_pipeline import (
    NODE_CLASS_MAPPINGS as PIPELINE_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as PIPELINE_NAMES,
)

# Editing nodes
from .nodes_editing import (
    NODE_CLASS_MAPPINGS as EDITING_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as EDITING_NAMES,
)

# Enhancement nodes
from .nodes_enhancement import (
    NODE_CLASS_MAPPINGS as ENHANCEMENT_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as ENHANCEMENT_NAMES,
)

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    **PIPELINE_NODES,
    **EDITING_NODES,
    **ENHANCEMENT_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **PIPELINE_NAMES,
    **EDITING_NAMES,
    **ENHANCEMENT_NAMES,
}

# Register web directory for Three.js viewer
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"[DiT360Plus] v{__version__} loaded - {len(NODE_CLASS_MAPPINGS)} nodes registered")
