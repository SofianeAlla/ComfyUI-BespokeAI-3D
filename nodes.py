"""
ComfyUI Custom Node for BespokeAI 3D Generation API
Converts images to 3D models using BespokeAI's AI-powered pipeline.
"""

import os
import time
import json
import base64
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import folder_paths
import comfy.utils


class BespokeAI3DGeneration:
    """
    Generate 3D models from images using BespokeAI API.
    Supports AI enhancement, PBR textures, low-poly mode, and part segmentation.
    Outputs the full GLB file with textures preserved.
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
                "filename_prefix": ("STRING", {"default": "bespokeai_3d"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "glb_url")
    FUNCTION = "generate_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

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
        """Poll for task completion with smooth 5-minute progress bar."""
        headers = {"X-API-Key": api_key}

        params = {"taskId": task_id}
        if segmentation:
            params["segmentation"] = "true"

        start_time = time.time()

        # 5 minutes = 300 seconds, 100 steps
        total_steps = 100
        pbar = comfy.utils.ProgressBar(total_steps)

        current_step = 0
        last_check_time = 0
        check_interval = 5.0  # Check API every 5 seconds

        while current_step < total_steps:
            elapsed = time.time() - start_time

            # Update progress bar based on time (smooth increment)
            expected_step = min(99, int((elapsed / self.GENERATION_TIME) * 100))
            if expected_step > current_step:
                current_step = expected_step
                pbar.update_absolute(current_step)

            # Check API status periodically
            if elapsed - last_check_time >= check_interval:
                last_check_time = elapsed
                try:
                    response = requests.get(self.API_URL, headers=headers, params=params, timeout=30)

                    if response.ok:
                        data = response.json()
                        status = data.get("status", "unknown")

                        if status == "complete":
                            # Success! Jump to 100%
                            pbar.update_absolute(total_steps)
                            return data
                        elif status == "failed" or "error" in data:
                            raise RuntimeError(f"Generation failed: {data.get('error', 'Unknown error')}")
                except requests.exceptions.RequestException:
                    # Network error, continue waiting
                    pass

            # Small sleep to prevent busy loop
            time.sleep(0.5)

        # If we reach here, 5 minutes passed - do one final check
        try:
            response = requests.get(self.API_URL, headers=headers, params=params, timeout=30)
            if response.ok:
                data = response.json()
                if data.get("status") == "complete":
                    pbar.update_absolute(total_steps)
                    return data
        except:
            pass

        raise RuntimeError("Generation timed out after 5 minutes. Please try again.")

    def generate_3d(self, image, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt="", filename_prefix="bespokeai_3d"):
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

        # Download and save the full GLB file (with textures!)
        print("[BespokeAI] Downloading GLB file (with textures)...")
        response = requests.get(glb_url, timeout=120)
        response.raise_for_status()

        # Save to file
        timestamp = int(time.time())
        filename = f"{filename_prefix}_{timestamp}.glb"
        glb_path = os.path.join(self.model_dir, filename)

        with open(glb_path, "wb") as f:
            f.write(response.content)

        print(f"[BespokeAI] 3D model saved: {glb_path}")
        print("[BespokeAI] 3D generation complete!")

        # Return results for UI - use "3d" key for built-in 3D viewer
        return {"ui": {"3d": [{"filename": filename, "subfolder": "bespokeai_3d", "type": "output"}]},
                "result": (glb_path, glb_url)}


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
                "filename_prefix": ("STRING", {"default": "bespokeai_3d"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_path", "glb_url")
    FUNCTION = "generate_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def __init__(self):
        self._main = BespokeAI3DGeneration()

    def generate_3d(self, image_url, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt="", filename_prefix="bespokeai_3d"):
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

        # Download and save the full GLB file (with textures!)
        print("[BespokeAI] Downloading GLB file (with textures)...")
        response = requests.get(glb_url, timeout=120)
        response.raise_for_status()

        # Save to file
        timestamp = int(time.time())
        filename = f"{filename_prefix}_{timestamp}.glb"
        glb_path = os.path.join(self._main.model_dir, filename)

        with open(glb_path, "wb") as f:
            f.write(response.content)

        print(f"[BespokeAI] 3D model saved: {glb_path}")
        print("[BespokeAI] 3D generation complete!")

        # Return results for UI - use "3d" key for built-in 3D viewer
        return {"ui": {"3d": [{"filename": filename, "subfolder": "bespokeai_3d", "type": "output"}]},
                "result": (glb_path, glb_url)}


class BespokeAI3DPreview:
    """
    Preview a 3D GLB file in ComfyUI using the built-in 3D viewer.
    Takes a file path to a GLB file and displays it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def preview_3d(self, glb_path):
        if not glb_path or not os.path.exists(glb_path):
            print("[BespokeAI] No valid GLB file path provided")
            return {"ui": {"3d": []}}

        # Get relative path info for UI - must use forward slashes
        output_dir = folder_paths.get_output_directory()

        # Normalize paths for comparison
        glb_path_norm = os.path.normpath(glb_path)
        output_dir_norm = os.path.normpath(output_dir)

        if glb_path_norm.startswith(output_dir_norm):
            rel_path = os.path.relpath(glb_path_norm, output_dir_norm)
            # Convert to forward slashes for web
            rel_path = rel_path.replace("\\", "/")
            parts = rel_path.split("/")
            if len(parts) > 1:
                subfolder = "/".join(parts[:-1])
                filename = parts[-1]
            else:
                subfolder = ""
                filename = parts[0]
        else:
            subfolder = ""
            filename = os.path.basename(glb_path)

        print(f"[BespokeAI] 3D Preview: {glb_path}")

        # Use "3d" key for ComfyUI's built-in 3D viewer
        return {"ui": {"3d": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BespokeAI3DGeneration": BespokeAI3DGeneration,
    "BespokeAI3DGenerationFromURL": BespokeAI3DGenerationFromURL,
    "BespokeAI3DPreview": BespokeAI3DPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BespokeAI3DGeneration": "BespokeAI 3D Generation",
    "BespokeAI3DGenerationFromURL": "BespokeAI 3D Generation (URL)",
    "BespokeAI3DPreview": "BespokeAI 3D Preview",
}
