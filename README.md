# gemma4-langbench
Evaluate how well gemma-4 supports various languages

## Setup Guides

- [Installing Ollama on macOS](ollama.md)
- [Installing LM Studio on macOS](lmstudio.md)
- [Enabling Gemma 4 thinking mode in LM Studio](lmstudio_SETUP_THINKING.md)
- [Installing uv on macOS](uv.md)
- [Installing Unsloth on macOS](unsloth.md)

## Tools

### `test_openrouter.py`

Smoke test for models via OpenRouter (Claude and Gemma). Runs a question with and without extended thinking (Claude only) and prints elapsed time, whether reasoning fired, a reasoning excerpt, and the answer. Useful for verifying your API key and that thinking is actually active before a full eval run.

```fish
set -x OPENROUTER_API_KEY sk-or-...
uv run test_openrouter.py
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

Evaluate all languages at once (round-robin, resumable) — **`eval_all_langs_v2.py`** (recommended):

```bash
# LM Studio (local)
uv run eval_all_langs_v2.py --local                             # all 900 per language
uv run eval_all_langs_v2.py --local -n 100                      # first 100 per language
uv run eval_all_langs_v2.py --local --thinking                  # enable extended thinking
uv run eval_all_langs_v2.py --local --reset                     # start fresh

# OpenRouter (bash/zsh)
OPENROUTER_API_KEY=sk-... uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6
OPENROUTER_API_KEY=sk-... uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6 --thinking
OPENROUTER_API_KEY=sk-... uv run eval_all_langs_v2.py --model google/gemma-4-26b-a4b-it --thinking -n 100
```

On **fish shell**, inline env vars don't work — set the key first:

```fish
set -x OPENROUTER_API_KEY sk-...
uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6
uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6 --thinking
```

`--local` and `--model` are mutually exclusive; exactly one is required. `--thinking` enables extended reasoning (temperature=1, 2000-token budget) and appends `_thinking` to all output file names so thinking/non-thinking runs don't overwrite each other.

### Dry-run tests

Verify the script works without writing any files (`--dry-run` skips all state/log writes):

```bash
# local, no thinking
uv run eval_all_langs_v2.py --local --dry-run -n 3

# local, thinking
uv run eval_all_langs_v2.py --local --thinking --dry-run -n 3

# openrouter claude, no thinking
uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6 --dry-run -n 3

# openrouter claude, thinking
uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6 --thinking --dry-run -n 3

# openrouter gemma, no thinking
uv run eval_all_langs_v2.py --model google/gemma-4-26b-a4b-it --dry-run -n 3

# openrouter gemma, thinking
uv run eval_all_langs_v2.py --model google/gemma-4-26b-a4b-it --thinking --dry-run -n 3
```

## Run outputs

Each run writes all its files to `runs/<run_id>/`:

```
runs/
  <run_id>/
    state.json          — per-question results + run metadata (source of truth)
    wrong_answers.jsonl — detailed record for every non-correct answer
    run.log             — lifecycle timestamps
```

`<run_id>` is the model name with `/` replaced by `_`, with `_thinking` appended when thinking is on (e.g. `anthropic_claude-sonnet-4.6_thinking`). `--reset` wipes the entire `runs/<run_id>/` directory before starting fresh.

### state.json

```json
{
  "meta": {
    "model": "anthropic/claude-sonnet-4.6",
    "base_url": "https://openrouter.ai/api/v1",
    "thinking": true,
    "api_kwargs": {"temperature": 1, "max_tokens": 2500, "reasoning_max_tokens": 2000},
    "n_per_language": 900,
    "prompt_sha256": "…",
    "git_sha": "3df7690",
    "started_at": "2026-04-19T09:00:00+00:00",
    "updated_at": "2026-04-19T09:45:12+00:00",
    "completed_at": null
  },
  "languages": {
    "eng_Latn": {
      "https://…/article|1": {"outcome": "correct",      "predicted": "A", "correct": "A", "elapsed": 0.42},
      "https://…/article|2": {"outcome": "wrong_answer", "predicted": "C", "correct": "A", "elapsed": 0.51}
    }
  }
}
```

Every question is recorded — correct ones included. Key format is `"{link}|{question_number}"`. `outcome` is `correct`, `wrong_answer`, `unparseable`, or `api_error`. State is written atomically after each question so Ctrl-C is safe; resume picks up from the dict lookup.

### wrong_answers.jsonl

Appended for every non-correct outcome. Each record is self-describing:

| Field | Description |
|---|---|
| `timestamp` | UTC time of the call |
| `model` / `base_url` | Exact model identifier and API endpoint |
| `thinking` / `temperature` / `max_tokens` / `reasoning_max_tokens` | Exact parameters used |
| `language` / `dialect` | Language file stem (e.g. `fin_Latn`) and Belebele dialect tag |
| `link` / `question_number` | Stable cross-language question ID |
| `error_type` | `wrong_answer`, `unparseable`, or `api_error` |
| `correct_label` / `correct_text` | The right answer letter and its full text |
| `predicted_label` / `raw_response` | What the model returned |
| `passage` / `question` / `choices` | Full question content in the tested language |
| `english` | Same content in English (omitted when language is `eng_Latn`) |
| `elapsed_seconds` | Latency for this call |

## Results

All results: generative, 0-shot. Prompt template and evaluation details: [belebele.md](belebele.md)

### gemma-4-26b-a4b-it via openrouter

```
Model: google/gemma-4-26b-a4b-it (no thinking)
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      870     30       0   96.7%
est_Latn        900    900      802     98       0   89.1%
fin_Latn        900    900      833     67       0   92.6%
swe_Latn        900    900      833     67       0   92.6%
------------------------------------------------------------
TOTAL          3600   3600     3338    262           92.7%
```

### Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6` via OpenRouter)

```
Model: anthropic/claude-sonnet-4.6
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      855     45       0   95.0%
est_Latn        900    900      815     85       0   90.6%
fin_Latn        900    900      830     70       0   92.2%
swe_Latn        900    900      800    100       0   88.9%
------------------------------------------------------------
TOTAL          3600   3600     3300    300           91.7%
```

```
Model: anthropic/claude-sonnet-4.6 with thinking
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      856     44       0   95.1%
est_Latn        900    900      811     89       0   90.1%
fin_Latn        900    900      828     72       0   92.0%
swe_Latn        900    900      805     95       0   89.4%
------------------------------------------------------------
TOTAL          3600   3600     3300    300           91.7%
```

Adding thinking to sonnet 4.6 didn't much affect the results, but it made running this somewhat slower.

### Gemma-4 E4B Q8_0 (`gemma-4-E4B-it-Q8_0.gguf`, 9.02 GB, LM Studio 0.4.9, Apple M2 Pro)

```
Model: Gemma-4 E4B Q8_0
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      810     90       0   90.0%
est_Latn        900    900      664    236       0   73.8%
fin_Latn        900    900      766    134       0   85.1%
swe_Latn        900    900      785    115       0   87.2%
------------------------------------------------------------
TOTAL          3600   3600     3025    575           84.0%
```

```
Model: Gemma-4 E4B Q8_0 with thinking
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      864     36       0   96.0%
est_Latn        900    900      791    109       0   87.9%
fin_Latn        900    900      830     70       0   92.2%
swe_Latn        900    900      833     67       0   92.6%
------------------------------------------------------------
TOTAL          3600   3600     3318    282           92.2%
```

Adding thinking to locally running Gemma-4 E4B Q8_0 significantly improved performance, and made the run time at least 20x longer.

### Claude Haiku 4.5 (`anthropic/claude-haiku-4.5` via OpenRouter)

```
Model: anthropic/claude-haiku-4.5
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      766    134       0   85.1%
est_Latn        900    900      718    182       0   79.8%
fin_Latn        900    900      761    139       0   84.6%
swe_Latn        900    900      773    127       0   85.9%
------------------------------------------------------------
TOTAL          3600   3600     3018    582           83.8%
```

```
Model: anthropic/claude-haiku-4.5 with thinking
Language       Done  Total  Correct  Wrong  Errors     Acc
------------------------------------------------------------
eng_Latn        900    900      742    158       0   82.4%
est_Latn        900    900      637    263       0   70.8%
fin_Latn        900    900      685    215       0   76.1%
swe_Latn        900    900      731    169       0   81.2%
------------------------------------------------------------
TOTAL          3600   3600     2795    805           77.6%
```

Yes, haiku with thinking performed worse.

## Analysis

`analyze_wrong_answers.py` joins all run data with the Belebele source data and produces per-question statistics. It reads from `runs/*/state.json` (new format) with automatic fallback to legacy `wrong_answers_*.jsonl` files at the repo root. Run with:

```bash
uv run analyze_wrong_answers.py
```

Outputs to `stats/`: `accuracy.csv`, `hardest_questions.csv`, `language_flip.csv`, `model_disagreement.csv`, `SUMMARY.md`.

### Questions wrong by all 4 models in all 4 languages

7 questions were answered incorrectly by every model in every language (16/16 failure rate). These are likely flawed benchmark questions rather than genuine model failures.

| q_no | Question | Link |
|------|----------|------|
| 2 | How long have humans been magnifying objects using lenses? | [Telescopes](https://en.wikibooks.org/wiki/High_School_Earth_Science/Telescopes) |
| 1 | What can be found at The Giza Plateau? | [Great Pyramid](https://en.wikibooks.org/wiki/The_Seven_Wonders_of_the_World/The_Great_Pyramid) |
| 2 | Based on the information given in the passage, what was not mentioned in the Atlanta Journal-Constitution's report? | [Beverly Hall](https://en.wikinews.org/wiki/Beverly_Hall,_indicted_public_school_superintendent,_dies_aged_68) |
| 1 | While visiting the location described in the passage, which of the following would be deemed as inappropriate? | [Auschwitz-Birkenau](https://en.wikivoyage.org/wiki/Auschwitz-Birkenau) |
| 2 | Which statement does not accurately describe auxiliary languages? | [Auxiliary languages](https://en.wikivoyage.org/wiki/Auxiliary_languages) |
| 1 | According to the passage, which scenario would be ideal for a traveler planning to take a bus from the inter-district station? | [Thimphu](https://en.wikivoyage.org/wiki/Thimphu) |
| 2 | Which of the following facts about Timbuktu is true? | [Timbuktu](https://en.wikivoyage.org/wiki/Timbuktu) |

#### Example bad question: Telescopes

> Humans have been making and using lenses for magnification for thousands and thousands of years. However, the first true telescopes were made in Europe in the late 16th century. These telescopes used a combination of two lenses to make distant objects appear both nearer and larger.

**Question:** How long have humans been magnifying objects using lenses?

- A. For a thousand years
- B. Since the late 16th century ← benchmark's "correct" answer
- C. For thousands of years ← what the passage actually says
- D. Since the early 16th century

The passage explicitly states humans have used lenses for magnification for "thousands and thousands of years", making **C** the correct reading. The benchmark marks **B** as correct, which describes when *telescopes* were invented — a different thing entirely. Every model answered C and was penalised for it.

### Language flip (English-correct → non-English wrong)

| Model | Questions flipped |
|-------|-------------------|
| anthropic/claude-haiku-4.5 (thinking) | 335 |
| local Gemma-4 E4B Q8_0 (thinking) | 133 |
| google/gemma-4-26b-a4b-it | 120 |
| anthropic/claude-sonnet-4.6 (thinking) | 118 |

Haiku has nearly 3× as many language flips as the other models — it is far more dependent on English phrasing than the others.

### Model disagreement per language

| Language | Questions where exactly 1 or 3 models differ |
|----------|----------------------------------------------|
| est_Latn | 265 |
| fin_Latn | 198 |
| eng_Latn | 154 |
| swe_Latn | 142 |

Estonian produces the most inter-model disagreement — it is the language where models most often diverge from each other.

### Unparseable responses

These are counted as wrong in all accuracy figures (matching `eval_state.json`).

| Model | Unparseable count |
|-------|-------------------|
| anthropic/claude-haiku-4.5 (thinking) | 695 |
| anthropic/claude-sonnet-4.6 (thinking) | 194 |
| google/gemma-4-26b-a4b-it | 40 |
| local Gemma-4 E4B Q8_0 (thinking) | 4 |

Haiku produces long explanatory responses instead of a single letter far more often than the other models. Its true "wrong reasoning" rate is substantially lower than its headline accuracy suggests.
