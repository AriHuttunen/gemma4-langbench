# Installing Unsloth on macOS

## System Requirements

- macOS with CPU support (Apple MLX coming soon)

## Unsloth Studio (Web UI)

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

### Launch

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

### Update

```bash
unsloth studio update
```

## Unsloth Core (Code-Based)

> For Python and uv installation, see [uv.md](uv.md).

```bash
uv venv unsloth_env --python 3.12
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

## Developer Install

```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
./install.sh --local
unsloth studio -H 0.0.0.0 -p 8888
```

## Uninstallation

```bash
rm -rf ~/.unsloth/studio
```

## Resources

- [Official website](https://unsloth.ai/)
- [Unsloth GitHub repository](https://github.com/unslothai/unsloth)
- [Installation guide](https://docs.unsloth.ai/get-started/installation)
