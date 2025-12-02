"""
ComfyUI-BespokeAI-3D
Custom nodes for generating 3D models using BespokeAI API.

Installation:
1. Clone/copy this folder to ComfyUI/custom_nodes/
2. pip install -r requirements.txt
3. Restart ComfyUI
4. Get your API key from https://bespokeai.build
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
