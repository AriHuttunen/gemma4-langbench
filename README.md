# gemma4-langbench
Evaluate how well gemma-4 supports various languages

## Setup Guides

- [Installing Ollama on macOS](ollama.md)
- [Installing LM Studio on macOS](lmstudio.md)
- [Enabling Gemma 4 thinking mode in LM Studio](lmstudio_SETUP_THINKING.md)
- [Installing uv on macOS](uv.md)
- [Installing Unsloth on macOS](unsloth.md)

## Tools

### `test_claude_on_openrouter.py`

Smoke test for Claude models via OpenRouter. Runs a question with and without extended thinking and prints elapsed time, whether reasoning fired, a reasoning excerpt, and the answer. Useful for verifying your API key and that thinking is actually active before a full eval run.

```fish
set -x OPENROUTER_API_KEY sk-or-...
uv run test_claude_on_openrouter.py
```

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
OPENROUTER_API_KEY=sk-... uv run eval_all_langs.py --model anthropic/claude-haiku-4.5  # OpenRouter (bash/zsh)
```

On **fish shell**, inline env vars don't work — set the key first:

```fish
set -x OPENROUTER_API_KEY sk-...
uv run eval_all_langs.py --model anthropic/claude-haiku-4.5
```

## Wrong-answer log

`eval_all_langs.py` appends a JSONL record to `./wrong_answers_<model>.jsonl` for every question the model gets wrong (including unparseable responses and API errors). Each record contains enough information to reproduce the failure without re-running the model:

| Field | Description |
|---|---|
| `timestamp` | UTC time of the call |
| `model` / `base_url` | Exact model identifier and API endpoint |
| `language` / `dialect` | Language file stem (e.g. `fin_Latn`) and Belebele dialect tag |
| `link` | Source article URL — together with `language` uniquely locates the question |
| `error_type` | `wrong_answer`, `unparseable`, or `api_error` |
| `correct_label` / `correct_text` | The right answer letter and its full text |
| `predicted_label` / `raw_response` | What the model returned |
| `passage` / `question` / `choices` | Full question content in the tested language |
| `english` | The same passage, question, and choices in English (omitted when language is `eng_Latn`) |
| `elapsed_seconds` | Latency for this call |

The file is append-only and safe to interrupt — progress is tracked separately in `.eval_state_<model>.json`. Running with `--reset` deletes both files before starting fresh.

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

### Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6` via OpenRouter) with thinking turned on

```
Model: anthropic/claude-sonnet-4.6
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      856     44       0   95.1%
est_Latn        900    900      811     89       0   90.1%
fin_Latn        900    900      828     72       0   92.0%
swe_Latn        900    900      805     95       0   89.4%
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
