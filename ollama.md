# Installing Ollama

## Quick Start

### macOS / Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or download the macOS installer directly: [Ollama.dmg](https://ollama.com/download/Ollama.dmg)

### Windows

**PowerShell:**

```powershell
irm https://ollama.com/install.ps1 | iex
```

Or download the installer directly: [OllamaSetup.exe](https://ollama.com/download/OllamaSetup.exe)

### Docker

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Verify Installation

```bash
ollama --version
```

## Pull a Model

```bash
ollama pull gemma3
```

## Resources

- [Official download page](https://ollama.com/download)
- [Ollama GitHub repository](https://github.com/ollama/ollama)
- [Available models](https://ollama.com/library)
- [Linux manual install instructions](https://docs.ollama.com/linux#manual-install)
