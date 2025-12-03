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


class BespokeAI3DGeneration:
    """
    Generate 3D models from images using BespokeAI API.
    Supports AI enhancement, PBR textures, low-poly mode, and part segmentation.
    Output mesh_path can be connected to ComfyUI's built-in Preview3D node (model_file input).
    """

    API_URL = "https://heovujhdxkvbkaaguzwl.supabase.co/functions/v1/public-3d-api"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "bspk_your_api_key_here"
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
                    "placeholder": "Optional: Custom prompt for AI enhancement"
                }),
                "poll_interval": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 30.0, "step": 1.0}),
                "max_poll_attempts": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "model_url", "enhanced_image_url")
    FUNCTION = "generate_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.model_dir = os.path.join(self.output_dir, "bespokeai_3d")
        os.makedirs(self.model_dir, exist_ok=True)

    def image_to_base64(self, image_tensor):
        """Convert ComfyUI IMAGE tensor to base64 PNG string."""
        # ComfyUI images are [B, H, W, C] float tensors in range [0, 1]
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]  # Take first image if batched

        # Convert to numpy and scale to 0-255
        img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

        # Create PIL Image
        pil_image = Image.fromarray(img_np)

        # Convert to base64
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

    def poll_task(self, api_key, task_id, segmentation, poll_interval, max_attempts):
        """Poll for task completion."""
        headers = {"X-API-Key": api_key}

        params = {"taskId": task_id}
        if segmentation:
            params["segmentation"] = "true"

        for attempt in range(max_attempts):
            response = requests.get(self.API_URL, headers=headers, params=params, timeout=30)

            if not response.ok:
                error_data = response.json() if response.text else {}
                raise RuntimeError(f"Polling failed: {error_data.get('error', response.text)}")

            data = response.json()
            status = data.get("status", "unknown")

            if status == "complete":
                return data
            elif status == "processing":
                progress = data.get("progress", 0)
                print(f"[BespokeAI] Processing... {progress}% (attempt {attempt + 1}/{max_attempts})")
                time.sleep(poll_interval)
            elif status == "failed" or "error" in data:
                raise RuntimeError(f"Generation failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"[BespokeAI] Unknown status: {status}")
                time.sleep(poll_interval)

        raise TimeoutError(f"Task did not complete within {max_attempts * poll_interval} seconds")

    def download_file(self, url, filename):
        """Download a file from URL to the output directory."""
        filepath = os.path.join(self.model_dir, filename)

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        return filepath

    def generate_3d(self, image, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt="",
                    poll_interval=5.0, max_poll_attempts=120):
        """Main generation function."""

        if not api_key or not api_key.strip():
            raise ValueError("API key is required. Get yours at https://bespokeai.build")

        api_key = api_key.strip()

        # Validate segmentation + resolution combo
        if segmentation and resolution != "500k":
            print("[BespokeAI] Warning: Segmentation only works with 500k resolution. Forcing 500k.")
            resolution = "500k"

        # Convert image to base64
        print("[BespokeAI] Preparing image...")
        image_data = self.image_to_base64(image)

        # Submit generation request
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
        credits_used = submit_response.get("creditsUsed", 0)
        enhanced_image_url = submit_response.get("enhancedImageUrl", "")

        print(f"[BespokeAI] Task submitted: {task_id} (Credits used: {credits_used})")

        # Poll for completion
        print("[BespokeAI] Waiting for generation to complete...")
        result = self.poll_task(
            api_key=api_key,
            task_id=task_id,
            segmentation=segmentation,
            poll_interval=poll_interval,
            max_attempts=max_poll_attempts
        )

        # Extract file URLs
        model_url = result.get("modelUrl", "")
        result_files = result.get("resultFiles", [])

        glb_url = ""

        for file_info in result_files:
            file_type = file_info.get("Type", "").lower()
            if file_type == "glb":
                glb_url = file_info.get("Url", "")
                break

        # Fallback to modelUrl if specific files not found
        if not glb_url and model_url:
            glb_url = model_url

        # Download GLB file
        timestamp = int(time.time())
        mesh_path = ""

        if glb_url:
            print("[BespokeAI] Downloading GLB file...")
            mesh_path = self.download_file(glb_url, f"model_{timestamp}.glb")
            print(f"[BespokeAI] GLB saved: {mesh_path}")

        print("[BespokeAI] 3D generation complete!")

        return (mesh_path, model_url, enhanced_image_url)


class BespokeAI3DGenerationFromURL:
    """
    Generate 3D models from image URLs using BespokeAI API.
    Use this node if you already have an image URL instead of a ComfyUI image.
    Output mesh_path can be connected to ComfyUI's built-in Preview3D node (model_file input).
    """

    API_URL = "https://heovujhdxkvbkaaguzwl.supabase.co/functions/v1/public-3d-api"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://example.com/image.jpg"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "bspk_your_api_key_here"
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
                    "placeholder": "Optional: Custom prompt for AI enhancement"
                }),
                "poll_interval": ("FLOAT", {"default": 5.0, "min": 2.0, "max": 30.0, "step": 1.0}),
                "max_poll_attempts": ("INT", {"default": 120, "min": 10, "max": 600}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("mesh_path", "model_url", "enhanced_image_url")
    FUNCTION = "generate_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.model_dir = os.path.join(self.output_dir, "bespokeai_3d")
        os.makedirs(self.model_dir, exist_ok=True)

        # Reuse methods from main class
        self._main = BespokeAI3DGeneration()

    def generate_3d(self, image_url, api_key, resolution, with_texture, ai_enhancement,
                    low_poly=False, segmentation=False, prompt="",
                    poll_interval=5.0, max_poll_attempts=120):
        """Generate 3D from image URL."""

        if not api_key or not api_key.strip():
            raise ValueError("API key is required. Get yours at https://bespokeai.build")

        if not image_url or not image_url.strip():
            raise ValueError("Image URL is required.")

        api_key = api_key.strip()
        image_url = image_url.strip()

        # Validate segmentation + resolution combo
        if segmentation and resolution != "500k":
            print("[BespokeAI] Warning: Segmentation only works with 500k resolution. Forcing 500k.")
            resolution = "500k"

        # Submit generation request with URL directly
        print("[BespokeAI] Submitting 3D generation request...")
        submit_response = self._main.submit_generation(
            api_key=api_key,
            image_data=image_url,  # API accepts URLs directly
            resolution=resolution,
            with_texture=with_texture,
            ai_enhancement=ai_enhancement,
            low_poly=low_poly,
            segmentation=segmentation,
            prompt=prompt
        )

        task_id = submit_response.get("taskId")
        credits_used = submit_response.get("creditsUsed", 0)
        enhanced_image_url = submit_response.get("enhancedImageUrl", "")

        print(f"[BespokeAI] Task submitted: {task_id} (Credits used: {credits_used})")

        # Poll for completion
        print("[BespokeAI] Waiting for generation to complete...")
        result = self._main.poll_task(
            api_key=api_key,
            task_id=task_id,
            segmentation=segmentation,
            poll_interval=poll_interval,
            max_attempts=max_poll_attempts
        )

        # Extract file URLs
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

        # Download GLB file
        timestamp = int(time.time())
        mesh_path = ""

        if glb_url:
            print("[BespokeAI] Downloading GLB file...")
            mesh_path = self._main.download_file(glb_url, f"model_{timestamp}.glb")
            print(f"[BespokeAI] GLB saved: {mesh_path}")

        print("[BespokeAI] 3D generation complete!")

        return (mesh_path, model_url, enhanced_image_url)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BespokeAI3DGeneration": BespokeAI3DGeneration,
    "BespokeAI3DGenerationFromURL": BespokeAI3DGenerationFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BespokeAI3DGeneration": "BespokeAI 3D Generation",
    "BespokeAI3DGenerationFromURL": "BespokeAI 3D Generation (URL)",
}
