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
uv run eval_all_langs.py                                        # LM Studio, all 900 per language
uv run eval_all_langs.py -n 100                                 # first 100 per language
uv run eval_all_langs.py --reset                                # start fresh
OPENROUTER_API_KEY=sk-... uv run eval_all_langs.py --model anthropic/claude-haiku-4.5  # OpenRouter
```

## Results

All results: generative, 0-shot, temperature 0. Prompt template and evaluation details: [belebele.md](belebele.md)

### Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6` via OpenRouter)

```
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      855     45       0   95.0%
est_Latn        900    900      815     85       0   90.6%
fin_Latn        900    900      830     70       0   92.2%
swe_Latn        900    900      800    100       0   88.9%
------------------------------------------------------------
TOTAL          3600   3600     3300    300           91.7%
```

### Gemma-4 E4B Q8_0 (`gemma-4-E4B-it-Q8_0.gguf`, 9.02 GB, LM Studio 0.4.9, Apple M2 Pro)

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

### Claude Haiku 4.5 (`anthropic/claude-haiku-4.5` via OpenRouter)

```
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      766    134       0   85.1%
est_Latn        900    900      718    182       0   79.8%
fin_Latn        900    900      761    139       0   84.6%
swe_Latn        900    900      773    127       0   85.9%
------------------------------------------------------------
TOTAL          3600   3600     3018    582           83.8%
```
