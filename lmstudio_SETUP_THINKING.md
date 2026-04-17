# Enabling Gemma 4 Thinking Mode in LM Studio

Two changes are needed in LM Studio's model settings for your E4B model. Both are in **My Models → gemma-4-E4B-it → Advanced Settings**.

## 1. Jinja Template

Add this as the **first line** of the Jinja prompt template:

```
{%- set enable_thinking = true %}
```

This sets the `enable_thinking` variable that the template already checks. When true, it injects the `<|think|>` token into the system turn, which triggers the model to produce a reasoning channel before answering.

## 2. Reasoning Parsing

Change the reasoning parsing delimiters from the DeepSeek-style defaults to Gemma 4's channel format:

| Field | Value |
|-------|-------|
| Start | `<\|channel>thought` |
| End   | `<channel\|>` |

This tells LM Studio to extract the thinking block and return it as `reasoning_content` in the API response, keeping `content` clean.

## Verifying it works

After reloading the model, test with a curl request:

```bash
curl -s http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "loaded-model",
    "messages": [{"role": "user", "content": "What is 17 * 23?"}],
    "max_tokens": 2048,
    "temperature": 0
  }' | python3 -m json.tool
```

In the response, `reasoning_content` should be non-empty and `content` should contain only the final answer.

## Script change

With thinking enabled at the LM Studio level, the only change needed in `eval_belebele.py` is increasing `max_tokens` to give the model room to reason:

```python
max_tokens=2048,  # was 10000 but reasoning needs ~500-1000, not 10k
```

No prompt or message structure changes are required — the Jinja template handles injection.

## Known issues

There is an active llama.cpp bug where Gemma 4 thinking mode can produce `<unused49>` token floods. This is mostly reported on 26B-A4B and 31B but may affect E4B. If you see it, restart the LM Studio server. If it persists, revert the Jinja change and run without thinking.
