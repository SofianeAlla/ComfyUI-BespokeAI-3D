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
    GENERATION_TIME = 600  # 10 minutes in seconds (some generations take longer)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
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

    def poll_task_with_progress(self, api_key, task_id, segmentation, pbar, start_time):
        """Poll for task completion with progress bar that started at task execution."""
        headers = {"X-API-Key": api_key}

        params = {"taskId": task_id}
        if segmentation:
            params["segmentation"] = "true"

        total_steps = 100
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
                        elif status == "failed":
                            error_msg = data.get('error', 'Unknown error')
                            # Don't fail on timeout - backend might still be processing
                            if "timed out" not in error_msg.lower():
                                raise RuntimeError(f"Generation failed: {error_msg}")
                            else:
                                print(f"\n[BespokeAI] Backend timeout, continuing to wait...")
                        elif "error" in data and "timed out" not in str(data.get('error', '')).lower():
                            raise RuntimeError(f"Generation failed: {data.get('error', 'Unknown error')}")
                except requests.exceptions.RequestException:
                    # Network error, continue waiting
                    pass

            # Small sleep to prevent busy loop
            time.sleep(0.5)

        # If we reach here, time passed - do one final check
        try:
            response = requests.get(self.API_URL, headers=headers, params=params, timeout=30)
            if response.ok:
                data = response.json()
                if data.get("status") == "complete":
                    pbar.update_absolute(total_steps)
                    return data
        except:
            pass

        raise RuntimeError("Generation timed out. Please try again.")

    def generate_3d(self, image, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt="", filename_prefix="bespokeai_3d"):
        """Main generation function."""

        if not api_key or not api_key.strip():
            raise ValueError("API key is required. Get yours at https://bespokeai.build")

        api_key = api_key.strip()

        # Start progress bar and timer IMMEDIATELY when task executes
        total_steps = 100
        pbar = comfy.utils.ProgressBar(total_steps)
        start_time = time.time()
        pbar.update_absolute(0)

        if segmentation and resolution != "500k":
            print("[BespokeAI] Warning: Segmentation only works with 500k resolution. Forcing 500k.")
            resolution = "500k"

        print("[BespokeAI] Preparing image...")
        image_data = self.image_to_base64(image)

        # Update progress after image preparation
        elapsed = time.time() - start_time
        pbar.update_absolute(min(5, int((elapsed / self.GENERATION_TIME) * 100)))

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
        print("[BespokeAI] Generating 3D model (this may take up to 10 minutes)...")

        result = self.poll_task_with_progress(
            api_key=api_key,
            task_id=task_id,
            segmentation=segmentation,
            pbar=pbar,
            start_time=start_time
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

        return {"result": (glb_path, glb_url)}


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
                    "password": True,
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

        # Start progress bar and timer IMMEDIATELY when task executes
        total_steps = 100
        pbar = comfy.utils.ProgressBar(total_steps)
        start_time = time.time()
        pbar.update_absolute(0)

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
        print("[BespokeAI] Generating 3D model (this may take up to 10 minutes)...")

        result = self._main.poll_task_with_progress(
            api_key=api_key,
            task_id=task_id,
            segmentation=segmentation,
            pbar=pbar,
            start_time=start_time
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

        return {"result": (glb_path, glb_url)}


class BespokeAI3DPreview:
    """
    Preview a 3D GLB file in ComfyUI using the built-in 3D viewer.
    Connect the glb_path output from BespokeAI 3D Generation to this node.
    The 3D viewer displays natively and loads the generated model after execution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of 3D files from output directory for the viewer
        output_dir = folder_paths.get_output_directory()
        model_dir = os.path.join(output_dir, "bespokeai_3d")
        os.makedirs(model_dir, exist_ok=True)

        files = [""]
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.lower().endswith(('.glb', '.gltf', '.obj', '.fbx', '.stl')):
                    files.append(f"bespokeai_3d/{f}")

        return {
            "required": {
                "model_file": (sorted(files), {"default": ""}),
                "image": ("LOAD_3D", {}),
            },
            "optional": {
                "glb_path": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True
    EXPERIMENTAL = True

    def preview_3d(self, model_file, image, glb_path=""):
        # If a glb_path is connected, use it to determine the viewer path
        if glb_path and glb_path.strip():
            output_dir = folder_paths.get_output_directory()
            model_file_norm = os.path.normpath(glb_path)
            output_dir_norm = os.path.normpath(output_dir)

            if model_file_norm.startswith(output_dir_norm):
                rel_path = os.path.relpath(model_file_norm, output_dir_norm)
                viewer_path = rel_path.replace("\\", "/")
            else:
                viewer_path = glb_path.replace("\\", "/")

            print(f"[BespokeAI] 3D Preview (from generation): {viewer_path}")

            # Return UI result to update the viewer with the new model
            return {"ui": {"result": [f"output/{viewer_path}", None, None]}}

        # Otherwise use the dropdown selection
        if model_file:
            print(f"[BespokeAI] 3D Preview (from dropdown): {model_file}")
            return {"ui": {"result": [f"output/{model_file}", None, None]}}

        print("[BespokeAI] No model file provided")
        return {"ui": {"result": ["", None, None]}}


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
