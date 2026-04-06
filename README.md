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

## Evaluation

Run the [Belebele](belebele.md) reading comprehension benchmark against a model served by LM Studio:

```bash
uv run eval_belebele.py                       # 100 questions, English
uv run eval_belebele.py -n 50                 # 50 questions
uv run eval_belebele.py -l fin_Latn           # Finnish
uv run eval_belebele.py -l swe_Latn -n 200   # 200 questions, Swedish
```

Requires LM Studio's local server running on `localhost:1234` with a model loaded.
