# Example Workflows

Import these workflows directly into ComfyUI by dragging the JSON file onto the canvas or using **Load** in the menu.

## Available Workflows

### `basic_image_to_3d.json`
Basic workflow that loads an image and converts it to a 3D model.

**Nodes used:**
- Load Image
- BespokeAI 3D Generation

**Instructions:**
1. Import the workflow
2. Select your image in the Load Image node
3. Enter your BespokeAI API key
4. Configure options (resolution, AI enhancement, etc.)
5. Queue the workflow

## Creating Your Own Workflows

The BespokeAI nodes can be combined with any other ComfyUI nodes. Some ideas:

- **With ControlNet**: Generate an image first, then convert to 3D
- **With upscalers**: Upscale your image before 3D conversion
- **Batch processing**: Use loops to process multiple images
- **With inpainting**: Clean up images before conversion

## Contributing Workflows

Have a useful workflow? Submit a PR to add it to this collection!
