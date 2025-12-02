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
from server import PromptServer
from aiohttp import web


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
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_url", "enhanced_image_url")
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
        poll_interval = 3.0

        while True:
            elapsed = time.time() - start_time

            # Calculate smooth progress (0-95% over 5 minutes, last 5% on completion)
            progress = min(95, (elapsed / self.GENERATION_TIME) * 95)

            # Create progress bar
            bar_length = 40
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)

            print(f"\r[BespokeAI] [{bar}] {progress:.1f}% - {elapsed_min:02d}:{elapsed_sec:02d} elapsed", end="", flush=True)

            # Check API status
            response = requests.get(self.API_URL, headers=headers, params=params, timeout=30)

            if not response.ok:
                print()  # New line after progress bar
                error_data = response.json() if response.text else {}
                raise RuntimeError(f"Generation failed: {error_data.get('error', response.text)}")

            data = response.json()
            status = data.get("status", "unknown")

            if status == "complete":
                # Show 100% completion
                bar = "█" * bar_length
                print(f"\r[BespokeAI] [{bar}] 100.0% - Complete!                    ")
                return data
            elif status == "failed" or "error" in data:
                print()  # New line after progress bar
                raise RuntimeError(f"Generation failed: {data.get('error', 'Unknown error')}")

            time.sleep(poll_interval)

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
        enhanced_image_url = submit_response.get("enhancedImageUrl", "")

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

        print("[BespokeAI] 3D generation complete!")

        return (glb_url, enhanced_image_url)


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
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("glb_url", "enhanced_image_url")
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
        enhanced_image_url = submit_response.get("enhancedImageUrl", "")

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

        print("[BespokeAI] 3D generation complete!")

        return (glb_url, enhanced_image_url)


class BespokeAIURLOutput:
    """
    Output node to display URLs from BespokeAI 3D generation.
    Shows the GLB model URL and enhanced image URL.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_url": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "enhanced_image_url": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "output_urls"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def output_urls(self, glb_url, enhanced_image_url=""):
        print(f"\n{'='*60}")
        print("[BespokeAI] OUTPUT URLs")
        print(f"{'='*60}")
        print(f"GLB Model URL: {glb_url}")
        if enhanced_image_url:
            print(f"Enhanced Image URL: {enhanced_image_url}")
        print(f"{'='*60}\n")

        return {"ui": {"glb_url": [glb_url], "enhanced_image_url": [enhanced_image_url]}}


class BespokeAI3DPreview:
    """
    Preview node for 3D GLB models.
    Downloads and displays a preview of the 3D model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_url": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def preview_3d(self, glb_url):
        if not glb_url:
            print("[BespokeAI] No GLB URL provided for preview")
            return {"ui": {"glb_url": [""]}}

        print(f"[BespokeAI] 3D Preview available at: {glb_url}")

        return {"ui": {"glb_url": [glb_url]}}


class BespokeAISave3D:
    """
    Save node for 3D GLB models.
    Downloads the GLB file from URL and saves it locally.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glb_url": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "bespokeai_3d"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_3d"
    CATEGORY = "BespokeAI/3D"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.model_dir = os.path.join(self.output_dir, "bespokeai_3d")
        os.makedirs(self.model_dir, exist_ok=True)

    def save_3d(self, glb_url, filename_prefix="bespokeai_3d"):
        if not glb_url:
            raise ValueError("No GLB URL provided to save")

        print("[BespokeAI] Downloading GLB file...")

        response = requests.get(glb_url, timeout=120)
        response.raise_for_status()

        timestamp = int(time.time())
        filename = f"{filename_prefix}_{timestamp}.glb"
        filepath = os.path.join(self.model_dir, filename)

        with open(filepath, "wb") as f:
            f.write(response.content)

        print(f"[BespokeAI] 3D model saved: {filepath}")

        return {"ui": {"saved_path": [filepath]}, "result": (filepath,)}


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "BespokeAI3DGeneration": BespokeAI3DGeneration,
    "BespokeAI3DGenerationFromURL": BespokeAI3DGenerationFromURL,
    "BespokeAIURLOutput": BespokeAIURLOutput,
    "BespokeAI3DPreview": BespokeAI3DPreview,
    "BespokeAISave3D": BespokeAISave3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BespokeAI3DGeneration": "BespokeAI 3D Generation",
    "BespokeAI3DGenerationFromURL": "BespokeAI 3D Generation (URL)",
    "BespokeAIURLOutput": "BespokeAI URL Output",
    "BespokeAI3DPreview": "BespokeAI 3D Preview",
    "BespokeAISave3D": "BespokeAI Save 3D",
}
