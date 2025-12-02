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
import threading
from io import BytesIO
from PIL import Image
import folder_paths
import comfy.utils


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


# Try to import MESH type from ComfyUI
try:
    from comfy_api.latest._util import MESH
except ImportError:
    try:
        from comfy_api.latest import Types
        MESH = Types.MESH
    except ImportError:
        # Fallback: define our own MESH class
        class MESH:
            def __init__(self, vertices, faces):
                self.vertices = vertices
                self.faces = faces


class BespokeAI3DGeneration:
    """
    Generate 3D models from images using BespokeAI API.
    Supports AI enhancement, PBR textures, low-poly mode, and part segmentation.
    """

    API_URL = "https://heovujhdxkvbkaaguzwl.supabase.co/functions/v1/public-3d-api"
    GENERATION_TIME = 300  # 5 minutes in seconds

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "resolution": (["500k", "1m", "1.5m"], {"default": "1m"}),
                "with_texture": ("BOOLEAN", {"default": True}),
                "ai_enhancement": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "low_poly": ("BOOLEAN", {"default": False}),
                "segmentation": ("BOOLEAN", {"default": False}),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "glb_url")
    FUNCTION = "generate_3d"
    CATEGORY = "BespokeAI/3D"

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.model_dir = os.path.join(self.output_dir, "bespokeai_3d")
        os.makedirs(self.model_dir, exist_ok=True)

    def image_to_base64(self, image_tensor):
        """Convert ComfyUI IMAGE tensor to base64 PNG string."""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{base64_str}"

    def submit_generation(self, api_key, image_data, resolution, with_texture,
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

        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=60)

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

    def poll_task_with_progress(self, api_key, task_id, segmentation):
        """Poll for task completion with smooth progress bar."""
        headers = {"X-API-Key": api_key}

        params = {"taskId": task_id}
        if segmentation:
            params["segmentation"] = "true"

        start_time = time.time()
        poll_interval = 5.0  # Poll API every 5 seconds

        # Total steps for 5 minutes = 300 seconds
        total_steps = 100
        pbar = comfy.utils.ProgressBar(total_steps)

        # Track completion status
        completed = False
        result_data = None
        error_msg = None

        def update_progress():
            """Update progress bar smoothly every second."""
            nonlocal completed
            while not completed:
                elapsed = time.time() - start_time
                # Progress goes from 0 to 95 over 5 minutes (300 seconds)
                current_step = min(95, int((elapsed / self.GENERATION_TIME) * 95))
                pbar.update_absolute(current_step)
                time.sleep(1)

        # Start progress updater thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

        try:
            while True:
                # Check API status
                try:
                    response = requests.get(self.API_URL, headers=headers, params=params, timeout=30)
                except requests.exceptions.RequestException as e:
                    # Network error, keep trying
                    time.sleep(poll_interval)
                    continue

                if not response.ok:
                    error_data = response.json() if response.text else {}
                    error_msg = f"Generation failed: {error_data.get('error', response.text)}"
                    break

                data = response.json()
                status = data.get("status", "unknown")

                if status == "complete":
                    result_data = data
                    break
                elif status == "failed" or "error" in data:
                    error_msg = f"Generation failed: {data.get('error', 'Unknown error')}"
                    break

                time.sleep(poll_interval)
        finally:
            completed = True
            progress_thread.join(timeout=2)

        if error_msg:
            raise RuntimeError(error_msg)

        # Show 100% completion
        pbar.update_absolute(100)
        return result_data

    def generate_3d(self, image, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt=""):
        """Main generation function."""

        if not api_key or not api_key.strip():
            raise ValueError("API key is required. Get yours at https://bespokeai.build")

        api_key = api_key.strip()

        if segmentation and resolution != "500k":
            print("[BespokeAI] Warning: Segmentation only works with 500k resolution. Forcing 500k.")
            resolution = "500k"

        print("[BespokeAI] Preparing image...")
        image_data = self.image_to_base64(image)

        print("[BespokeAI] Submitting 3D generation request...")
        submit_response = self.submit_generation(
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

        result = self.poll_task_with_progress(
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

        mesh = MESH(vertices, faces)

        print("[BespokeAI] 3D generation complete!")

        return (mesh, glb_url)


class BespokeAI3DGenerationFromURL:
    """
    Generate 3D models from image URLs using BespokeAI API.
    Use this node if you already have an image URL instead of a ComfyUI image.
    """

    API_URL = "https://heovujhdxkvbkaaguzwl.supabase.co/functions/v1/public-3d-api"
    GENERATION_TIME = 300

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "resolution": (["500k", "1m", "1.5m"], {"default": "1m"}),
                "with_texture": ("BOOLEAN", {"default": True}),
                "ai_enhancement": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "low_poly": ("BOOLEAN", {"default": False}),
                "segmentation": ("BOOLEAN", {"default": False}),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "glb_url")
    FUNCTION = "generate_3d"
    CATEGORY = "BespokeAI/3D"

    def __init__(self):
        self._main = BespokeAI3DGeneration()

    def generate_3d(self, image_url, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt=""):
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
        submit_response = self._main.submit_generation(
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

        result = self._main.poll_task_with_progress(
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

        mesh = MESH(vertices, faces)

        print("[BespokeAI] 3D generation complete!")

        return (mesh, glb_url)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BespokeAI3DGeneration": BespokeAI3DGeneration,
    "BespokeAI3DGenerationFromURL": BespokeAI3DGenerationFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BespokeAI3DGeneration": "BespokeAI 3D Generation",
    "BespokeAI3DGenerationFromURL": "BespokeAI 3D Generation (URL)",
}
