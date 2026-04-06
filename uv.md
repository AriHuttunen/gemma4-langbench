# Installing uv on macOS

## Quick Start

1. **Install via Homebrew:**

```bash
brew install uv
```

Or via the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Verify installation:**

```bash
uv --version
```

## Common Usage

```bash
# Install a package
uv pip install <package>

# Create a virtual environment
uv venv

# Install from requirements file
uv pip install -r requirements.txt
```

## Resources

- [Official documentation](https://docs.astral.sh/uv/)
- [uv GitHub repository](https://github.com/astral-sh/uv)
