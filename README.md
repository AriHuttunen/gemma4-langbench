# gemma4-langbench
Evaluate how well gemma-4 supports various languages

## Setup Guides

- [Installing Ollama on macOS](ollama.md)
- [Installing LM Studio on macOS](lmstudio.md)
- [Installing uv on macOS](uv.md)
- [Installing Unsloth on macOS](unsloth.md)

## Dataset

Download the [Belebele](https://huggingface.co/datasets/facebook/belebele) benchmark (English, Finnish, Swedish, Estonian):

```bash
uv run download_belebele.py
```

This saves JSONL files to `data/belebele/`.
