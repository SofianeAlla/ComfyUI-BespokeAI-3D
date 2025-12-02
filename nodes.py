"""
ComfyUI Custom Node for BespokeAI 3D Generation API
Converts images to 3D models using BespokeAI's AI-powered pipeline.
"""

import os
import time
import json
import base64
import struct
import requests
import numpy as np
import torch
from io import BytesIO
from PIL import Image
import folder_paths
from comfy_api.latest import IO, Types


def parse_glb(glb_data):
    """Parse GLB binary data and extract vertices and faces as torch tensors."""
    # GLB Header: magic (4) + version (4) + length (4)
    magic, version, length = struct.unpack('<4sII', glb_data[:12])

    if magic != b'glTF':
        raise ValueError("Invalid GLB file: wrong magic number")

    offset = 12
    json_chunk = None
    bin_chunk = None

    while offset < len(glb_data):
        chunk_length, chunk_type = struct.unpack('<II', glb_data[offset:offset+8])
        offset += 8
        chunk_data = glb_data[offset:offset+chunk_length]
        offset += chunk_length

        if chunk_type == 0x4E4F534A:  # JSON
            json_chunk = json.loads(chunk_data.decode('utf-8'))
        elif chunk_type == 0x004E4942:  # BIN
            bin_chunk = chunk_data

    if json_chunk is None or bin_chunk is None:
        raise ValueError("Invalid GLB file: missing JSON or BIN chunk")

    # Extract mesh data
    mesh = json_chunk['meshes'][0]
    primitive = mesh['primitives'][0]

    # Get position accessor
    position_accessor_idx = primitive['attributes']['POSITION']
    position_accessor = json_chunk['accessors'][position_accessor_idx]
    position_buffer_view = json_chunk['bufferViews'][position_accessor['bufferView']]

    # Get indices accessor
    indices_accessor_idx = primitive['indices']
    indices_accessor = json_chunk['accessors'][indices_accessor_idx]
    indices_buffer_view = json_chunk['bufferViews'][indices_accessor['bufferView']]

    # Extract vertices
    v_offset = position_buffer_view.get('byteOffset', 0) + position_accessor.get('byteOffset', 0)
    v_count = position_accessor['count']
    vertices_np = np.frombuffer(bin_chunk[v_offset:v_offset + v_count * 12], dtype=np.float32).reshape(-1, 3)

    # Extract indices/faces
    i_offset = indices_buffer_view.get('byteOffset', 0) + indices_accessor.get('byteOffset', 0)
    i_count = indices_accessor['count']

    # Handle different component types
    component_type = indices_accessor['componentType']
    if component_type == 5125:  # UNSIGNED_INT
        indices_np = np.frombuffer(bin_chunk[i_offset:i_offset + i_count * 4], dtype=np.uint32)
    elif component_type == 5123:  # UNSIGNED_SHORT
        indices_np = np.frombuffer(bin_chunk[i_offset:i_offset + i_count * 2], dtype=np.uint16)
    elif component_type == 5121:  # UNSIGNED_BYTE
        indices_np = np.frombuffer(bin_chunk[i_offset:i_offset + i_count], dtype=np.uint8)
    else:
        indices_np = np.frombuffer(bin_chunk[i_offset:i_offset + i_count * 4], dtype=np.uint32)

    faces_np = indices_np.reshape(-1, 3).astype(np.int64)

    vertices = torch.from_numpy(vertices_np.copy()).float()
    faces = torch.from_numpy(faces_np.copy()).long()

    return vertices, faces


class BespokeAI3DGeneration(IO.ComfyNode):
    """
    Generate 3D models from images using BespokeAI API.
    Supports AI enhancement, PBR textures, low-poly mode, and part segmentation.
    """

    API_URL = "https://heovujhdxkvbkaaguzwl.supabase.co/functions/v1/public-3d-api"
    GENERATION_TIME = 300  # 5 minutes in seconds

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BespokeAI3DGeneration",
            display_name="BespokeAI 3D Generation",
            category="BespokeAI/3D",
            inputs=[
                IO.Image.Input("image"),
                IO.String.Input("api_key", default="", multiline=False),
                IO.Combo.Input("resolution", options=["500k", "1m", "1.5m"], default="1m"),
                IO.Boolean.Input("with_texture", default=True),
                IO.Boolean.Input("ai_enhancement", default=True),
                IO.Boolean.Input("low_poly", default=False, optional=True),
                IO.Boolean.Input("segmentation", default=False, optional=True),
                IO.String.Input("prompt", default="", multiline=True, optional=True),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                IO.String.Output("glb_url"),
            ]
        )

    @classmethod
    def execute(cls, image, api_key, resolution, with_texture, ai_enhancement,
                low_poly=False, segmentation=False, prompt="") -> IO.NodeOutput:
        """Main generation function."""

        if not api_key or not api_key.strip():
            raise ValueError("API key is required. Get yours at https://bespokeai.build")

        api_key = api_key.strip()

        if segmentation and resolution != "500k":
            print("[BespokeAI] Warning: Segmentation only works with 500k resolution. Forcing 500k.")
            resolution = "500k"

        print("[BespokeAI] Preparing image...")
        image_data = cls.image_to_base64(image)

        print("[BespokeAI] Submitting 3D generation request...")
        submit_response = cls.submit_generation(
            api_key=api_key,
            image_data=image_data,
            resolution=resolution,
            with_texture=with_texture,
            ai_enhancement=ai_enhancement,
            low_poly=low_poly,
            segmentation=segmentation,
            prompt=prompt
        )

        task_id = submit_response.get("taskId")

        print(f"[BespokeAI] Task submitted: {task_id}")
        print("[BespokeAI] Generating 3D model (this may take up to 5 minutes)...")

        result = cls.poll_task_with_progress(
            api_key=api_key,
            task_id=task_id,
            segmentation=segmentation
        )

        # Extract GLB URL
        model_url = result.get("modelUrl", "")
        result_files = result.get("resultFiles", [])

        glb_url = ""
        for file_info in result_files:
            file_type = file_info.get("Type", "").lower()
            if file_type == "glb":
                glb_url = file_info.get("Url", "")
                break

        if not glb_url and model_url:
            glb_url = model_url

        if not glb_url:
            raise RuntimeError("No GLB URL returned from API")

        # Download and parse GLB
        print("[BespokeAI] Downloading GLB file...")
        response = requests.get(glb_url, timeout=120)
        response.raise_for_status()

        print("[BespokeAI] Parsing 3D mesh...")
        vertices, faces = parse_glb(response.content)

        # Add batch dimension
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)

        mesh = Types.MESH(vertices, faces)

        print("[BespokeAI] 3D generation complete!")

        return IO.NodeOutput(mesh, glb_url)

    @staticmethod
    def image_to_base64(image_tensor):
        """Convert ComfyUI IMAGE tensor to base64 PNG string."""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{base64_str}"

    @staticmethod
    def submit_generation(api_key, image_data, resolution, with_texture,
                          ai_enhancement, low_poly, segmentation, prompt):
        """Submit 3D generation request to BespokeAI API."""
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }

        payload = {
            "imageData": image_data,
            "resolution": resolution,
            "withTexture": with_texture,
            "aiEnhancement": ai_enhancement,
            "lowPoly": low_poly,
            "segmentation": segmentation,
        }

        if prompt and prompt.strip():
            payload["prompt"] = prompt.strip()

        response = requests.post(BespokeAI3DGeneration.API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 401:
            raise ValueError("Invalid API key. Please check your BespokeAI API key.")
        elif response.status_code == 402:
            raise ValueError("Insufficient credits. Please add more credits to your BespokeAI account.")
        elif response.status_code == 429:
            raise ValueError("Rate limit exceeded. Please wait before making more requests.")
        elif response.status_code == 400:
            error_data = response.json()
            raise ValueError(f"Invalid request: {error_data.get('error', 'Unknown error')}")
        elif not response.ok:
            raise RuntimeError(f"API request failed with status {response.status_code}: {response.text}")

        return response.json()

    @staticmethod
    def poll_task_with_progress(api_key, task_id, segmentation):
        """Poll for task completion with smooth progress bar."""
        headers = {"X-API-Key": api_key}

        params = {"taskId": task_id}
        if segmentation:
            params["segmentation"] = "true"

        start_time = time.time()
        poll_interval = 3.0

        while True:
            elapsed = time.time() - start_time

            # Calculate smooth progress (0-95% over 5 minutes, last 5% on completion)
            progress = min(95, (elapsed / BespokeAI3DGeneration.GENERATION_TIME) * 95)

            # Create progress bar
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "=" * filled + ">" + " " * (bar_length - filled - 1)
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)

            print(f"\r[BespokeAI] [{bar}] {progress:.1f}% - {elapsed_min:02d}:{elapsed_sec:02d} elapsed", end="", flush=True)

            # Check API status
            response = requests.get(BespokeAI3DGeneration.API_URL, headers=headers, params=params, timeout=30)

            if not response.ok:
                print()  # New line after progress bar
                error_data = response.json() if response.text else {}
                raise RuntimeError(f"Generation failed: {error_data.get('error', response.text)}")

            data = response.json()
            status = data.get("status", "unknown")

            if status == "complete":
                # Show 100% completion
                bar = "=" * bar_length
                print(f"\r[BespokeAI] [{bar}] 100.0% - Complete!                    ")
                return data
            elif status == "failed" or "error" in data:
                print()  # New line after progress bar
                raise RuntimeError(f"Generation failed: {data.get('error', 'Unknown error')}")

            time.sleep(poll_interval)


class BespokeAI3DGenerationFromURL(IO.ComfyNode):
    """
    Generate 3D models from image URLs using BespokeAI API.
    Use this node if you already have an image URL instead of a ComfyUI image.
    """

    API_URL = "https://heovujhdxkvbkaaguzwl.supabase.co/functions/v1/public-3d-api"
    GENERATION_TIME = 300

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BespokeAI3DGenerationFromURL",
            display_name="BespokeAI 3D Generation (URL)",
            category="BespokeAI/3D",
            inputs=[
                IO.String.Input("image_url", default="", multiline=False),
                IO.String.Input("api_key", default="", multiline=False),
                IO.Combo.Input("resolution", options=["500k", "1m", "1.5m"], default="1m"),
                IO.Boolean.Input("with_texture", default=True),
                IO.Boolean.Input("ai_enhancement", default=True),
                IO.Boolean.Input("low_poly", default=False, optional=True),
                IO.Boolean.Input("segmentation", default=False, optional=True),
                IO.String.Input("prompt", default="", multiline=True, optional=True),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                IO.String.Output("glb_url"),
            ]
        )

    @classmethod
    def execute(cls, image_url, api_key, resolution, with_texture, ai_enhancement,
                low_poly=False, segmentation=False, prompt="") -> IO.NodeOutput:
        """Generate 3D from image URL."""

        if not api_key or not api_key.strip():
            raise ValueError("API key is required. Get yours at https://bespokeai.build")

        if not image_url or not image_url.strip():
            raise ValueError("Image URL is required.")

        api_key = api_key.strip()
        image_url = image_url.strip()

        if segmentation and resolution != "500k":
            print("[BespokeAI] Warning: Segmentation only works with 500k resolution. Forcing 500k.")
            resolution = "500k"

        print("[BespokeAI] Submitting 3D generation request...")
        submit_response = BespokeAI3DGeneration.submit_generation(
            api_key=api_key,
            image_data=image_url,
            resolution=resolution,
            with_texture=with_texture,
            ai_enhancement=ai_enhancement,
            low_poly=low_poly,
            segmentation=segmentation,
            prompt=prompt
        )

        task_id = submit_response.get("taskId")

        print(f"[BespokeAI] Task submitted: {task_id}")
        print("[BespokeAI] Generating 3D model (this may take up to 5 minutes)...")

        result = BespokeAI3DGeneration.poll_task_with_progress(
            api_key=api_key,
            task_id=task_id,
            segmentation=segmentation
        )

        model_url = result.get("modelUrl", "")
        result_files = result.get("resultFiles", [])

        glb_url = ""
        for file_info in result_files:
            file_type = file_info.get("Type", "").lower()
            if file_type == "glb":
                glb_url = file_info.get("Url", "")
                break

        if not glb_url and model_url:
            glb_url = model_url

        if not glb_url:
            raise RuntimeError("No GLB URL returned from API")

        # Download and parse GLB
        print("[BespokeAI] Downloading GLB file...")
        response = requests.get(glb_url, timeout=120)
        response.raise_for_status()

        print("[BespokeAI] Parsing 3D mesh...")
        vertices, faces = parse_glb(response.content)

        # Add batch dimension
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)

        mesh = Types.MESH(vertices, faces)

        print("[BespokeAI] 3D generation complete!")

        return IO.NodeOutput(mesh, glb_url)


# Export for ComfyUI extension system
from comfy_api.latest import ComfyExtension
from typing_extensions import override


class BespokeAI3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            BespokeAI3DGeneration,
            BespokeAI3DGenerationFromURL,
        ]


async def comfy_entrypoint() -> BespokeAI3DExtension:
    return BespokeAI3DExtension()
