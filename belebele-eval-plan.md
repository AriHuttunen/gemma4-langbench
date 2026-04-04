# Belebele Benchmark Evaluation: Gemma 4 E4B on Mac M1

Evaluate Gemma 4 E4B (instruction-tuned) on multilingual reading comprehension
across English, Finnish, Swedish, and Estonian using the Belebele benchmark.

**Hardware**: Mac M1, 16 GB unified memory
**Model**: Gemma 4 E4B-it, Q8_0 quantization (~8.7 GB GGUF)
**Inference**: llama.cpp server via Homebrew

---

## Status

Steps 0–4 are **done and verified**. The model runs locally at ~29 t/s
generation on M1 and the llama-server serves completions with logprobs.

Steps 5–7 (running Belebele via lm-evaluation-harness) are **blocked** due
to compatibility issues between Gemma 4 (released 2026-04-02) and available
evaluation tooling. See the "What didn't work" section at the end. A custom
evaluation script is needed as the next step.

---

## 0. Set up the Python environment ✅

```bash
uv venv belebele-env
source belebele-env/bin/activate.fish   # .fish for fish shell
uv python pin 3.13
```

## 1. Install llama.cpp ✅

```bash
brew install llama.cpp
```

This installs `llama-server`, `llama-cli`, and related binaries with Metal
(Apple Silicon GPU) support enabled by default.

## 2. Download the model ✅

Authenticate with Hugging Face (needed for gated repos like google/gemma-4):

```bash
uvx --from huggingface_hub hf auth login
```

Download the Q8_0 GGUF (8.66 GB) and the tokenizer:

```bash
# Model weights (Unsloth repo, ungated)
uvx --from huggingface_hub --with hf_transfer hf download \
  unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q8_0.gguf \
  --local-dir ~/models/gemma-4-E4B-it-GGUF

# Tokenizer (google repo, gated — needs auth above)
uvx --from huggingface_hub --with hf_transfer hf download \
  google/gemma-4-E4B-it \
  --include "tokenizer*" --include "special_tokens*" \
  --local-dir ~/models/gemma-4-E4B-it-tokenizer
```

> **RAM note**: Q8_0 at 8.7 GB leaves ~7 GB for macOS + KV cache. Belebele
> samples are short (~300–400 tokens each), so this is comfortable. If you see
> swapping, fall back to Q4_K_M (5.0 GB) — download
> `gemma-4-E4B-it-Q4_K_M.gguf` instead and substitute it in all commands below.

## 3. Smoke-test the model ✅

```bash
llama-cli \
  -m ~/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf \
  -p "Mikä on Suomen pääkaupunki?" \
  -n 64
```

Confirmed working: model responds "Suomen pääkaupunki on **Helsinki**."
at ~29 t/s generation, ~46 t/s prompt processing on M1.

## 4. Start the llama.cpp server ✅

```bash
llama-server \
  -m ~/models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096
```

- `-c 4096`: Context window. 4K tokens is plenty for Belebele's short passages.
  Keeps KV cache small on 16 GB RAM.

Leave this running in its own terminal. Verified working — the completions
endpoint returns logprobs:

```bash
curl -s http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Helsinki is", "max_tokens": 5, "logprobs": 5}' \
  | python3 -m json.tool
```

## 5. Run the Belebele evaluation ❌ BLOCKED

This is where tooling breaks down. See "What didn't work" below.

**Next step**: Write a custom Python evaluation script that talks directly to
the llama-server `/v1/completions` endpoint, loads Belebele from Hugging Face,
and scores loglikelihood per answer choice. The server works, the logprobs
work — the only missing piece is the scoring harness.

---

## What didn't work

### lm-evaluation-harness: `local-completions` model type

```bash
uv pip install "lm-eval[api]" transformers
```

```bash
uv run lm_eval \
  --model local-completions \
  --model_args "model=gemma-4-E4B-it,base_url=http://127.0.0.1:8080/v1,tokenizer_backend=huggingface,tokenizer=google/gemma-4-E4B-it" \
  --tasks belebele_eng_Latn \
  --limit 10 --batch_size 1 \
  --output_path ~/results/belebele-test
```

**Result**: `404 Client Error: Not Found for url: http://127.0.0.1:8080/v1`.
The model type constructs API calls that don't match llama-server's endpoint
layout. Variations with and without `/v1` in the base URL all failed.

### lm-evaluation-harness: `gguf` model type

```bash
uv run lm_eval \
  --model gguf \
  --model_args "base_url=http://127.0.0.1:8080" \
  --tasks belebele_eng_Latn \
  --limit 10 --batch_size 1 \
  --output_path ~/results/belebele-test
```

**Result**: Connects and runs, but every request warns `Invalid logprobs data.
Expected 'logprobs' to contain 'token_logprobs' list.` The `gguf` model type
expects the legacy OpenAI logprobs format (flat `token_logprobs` array), but
llama-server returns the newer format (nested `content` array with objects).
All scores come back as zero.

### lm-evaluation-harness: HF backend with GGUF

```bash
uv pip install torch gguf accelerate

uv run lm_eval \
  --model hf \
  --model_args "pretrained=$HOME/models/gemma-4-E4B-it-GGUF,gguf_file=gemma-4-E4B-it-Q8_0.gguf,tokenizer=$HOME/models/gemma-4-E4B-it-tokenizer" \
  --tasks belebele_eng_Latn \
  --device mps --batch_size 1 \
  --output_path ~/results/belebele-test
```

**Result**: `ValueError: GGUF model with architecture gemma4 is not supported
yet.` Gemma 4 was released 2026-04-02; the `transformers` library does not yet
have a GGUF loader for the gemma4 architecture.

### Root cause

Gemma 4 is two days old. The three integration paths between llama.cpp and
lm-eval all have version mismatches: endpoint routing, logprobs format, and
architecture support. These will likely resolve in weeks as the ecosystem
catches up, but as of 2026-04-04 none of them work end to end.
