# Installing Unsloth

## System Requirements

**GPU Support:**

- NVIDIA: RTX 30/40/50, Blackwell, DGX Spark, Station
- macOS: CPU and Apple MLX (coming soon)
- AMD: Chat + Data works; training via Unsloth Core
- CPU: Supported for Chat and Data Recipes

## Unsloth Studio (Web UI)

### macOS / Linux / WSL

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

### Windows (PowerShell)

```powershell
irm https://unsloth.ai/install.ps1 | iex
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

> For Python installation, see [uv.md](uv.md).

### Linux / WSL

```bash
uv venv unsloth_env --python 3.12
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

### Windows

```powershell
uv venv unsloth_env --python 3.12
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```

## Docker

```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 8000:8000 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

## Developer Install (macOS / Linux / WSL)

```bash
git clone https://github.com/unslothai/unsloth
cd unsloth
./install.sh --local
unsloth studio -H 0.0.0.0 -p 8888
```

## Uninstallation

- **macOS / Linux / WSL:** `rm -rf ~/.unsloth/studio`
- **Windows:** `Remove-Item -Recurse -Force "$HOME\.unsloth\studio"`

## Resources

- [Official website](https://unsloth.ai/)
- [Unsloth GitHub repository](https://github.com/unslothai/unsloth)
- [Installation guide](https://docs.unsloth.ai/get-started/installation)
