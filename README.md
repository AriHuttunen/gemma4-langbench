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

Evaluate all languages at once (round-robin, resumable):

```bash
uv run eval_all_langs.py              # all 900 per language
uv run eval_all_langs.py -n 100       # first 100 per language
uv run eval_all_langs.py --reset      # start fresh
```

## Results

**Model:** `lmstudio-community/gemma-4-E4B-it-GGUF` (`gemma-4-E4B-it-Q8_0.gguf`, 9.02 GB)
**Method:** Generative, 0-shot, temperature 0
**Inference:** LM Studio 0.4.9 on macOS
**System:** Apple M2 Pro, 16 GB RAM, Metal GPU
**Date:** 2026-04-06

```
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      810     90       0   90.0%
est_Latn        900    900      664    236       0   73.8%
fin_Latn        900    900      766    134       0   85.1%
swe_Latn        900    900      785    115       0   87.2%
------------------------------------------------------------
TOTAL          3600   3600     3025    575           84.0%
```

Prompt template and evaluation details: [belebele.md](belebele.md)
