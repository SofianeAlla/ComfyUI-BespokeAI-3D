# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- `BespokeAI 3D Generation` node for image-to-3D conversion
- `BespokeAI 3D Generation (URL)` node for URL-based input
- Support for multiple resolutions (500k, 1m, 1.5m)
- AI enhancement with custom prompt support
- PBR texture generation
- Low poly mode for optimized models
- Part segmentation (500k resolution)
- Automatic polling with configurable intervals
- GLB and OBJ output format support
- Comprehensive error handling for all API error codes
- Progress logging during generation

### Technical
- Asynchronous task polling
- Automatic file download to ComfyUI output directory
- Base64 image encoding for ComfyUI IMAGE inputs
- URL passthrough for direct image URLs

## [Unreleased]

### Planned
- 3D model preview node
- Texture map extraction node
- Batch processing optimizations
- Webhook support for external notifications
