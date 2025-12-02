# Contributing to ComfyUI-BespokeAI-3D

First off, thank you for considering contributing to ComfyUI-BespokeAI-3D! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### Suggesting Features

Feature suggestions are tracked as GitHub issues. When creating a feature request, use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- A clear and descriptive title
- Detailed description of the proposed feature
- Explain why this feature would be useful
- List any alternatives you've considered

### Pull Requests

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Make your changes** and commit them:
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push** to your fork:
   ```bash
   git push origin feature/amazing-feature
   ```
6. Open a **Pull Request**

### Code Style

- Follow PEP 8 guidelines for Python code
- Use descriptive variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and single-purpose
- Comment complex logic

### Testing

Before submitting:

1. Test your changes with ComfyUI
2. Verify all existing functionality still works
3. Test with different image types and sizes
4. Check error handling works correctly

### Commit Messages

- Use clear and meaningful commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issues when applicable: `Fix #123`

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ComfyUI-BespokeAI-3D.git

# Navigate to project
cd ComfyUI-BespokeAI-3D

# Install dependencies
pip install -r requirements.txt

# Create a symlink in ComfyUI custom_nodes (optional)
ln -s $(pwd) /path/to/ComfyUI/custom_nodes/ComfyUI-BespokeAI-3D
```

## Questions?

Feel free to open an issue with the `question` label or reach out via:

- **GitHub Issues**: For bugs and feature requests
- **BespokeAI Support**: [bespokeai.build/support](https://bespokeai.build/support)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
