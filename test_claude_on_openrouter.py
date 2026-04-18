# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""
API smoke test for Claude models via OpenRouter.

Verifies that:
  - Basic chat completions work (temperature=0)
  - Extended thinking works (temperature=1, reasoning enabled)

The OpenAI SDK silently drops the `reasoning` and `reasoning_details` fields
from the parsed message object, so we inspect the raw HTTP response JSON
directly to confirm whether thinking is active.

Usage:
    set -x OPENROUTER_API_KEY sk-or-...   # fish
    uv run test_claude_on_openrouter.py
"""
import os
import time
from openai import OpenAI

MODEL = "anthropic/claude-sonnet-4.6"
QUESTION = "What is the sum of all prime numbers less than 100?"  # correct answer: 1060

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

CASES = [
    (
        "without thinking",
        {"temperature": 0, "max_tokens": 512},
    ),
    (
        "with thinking",
        {
            "temperature": 1,
            "max_tokens": 2500,
            "extra_body": {"reasoning": {"type": "enabled", "max_tokens": 2000}},
        },
    ),
]

for label, kwargs in CASES:
    print(f"── {label} ──")
    t0 = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": QUESTION}],
        **kwargs,
    )
    elapsed = time.perf_counter() - t0

    data = raw.http_response.json()
    msg = data["choices"][0]["message"]
    reasoning = msg.get("reasoning") or ""

    print(f"  model:    {data.get('model', MODEL)}")
    print(f"  elapsed:  {elapsed:.2f}s")
    print(f"  thinking: {'yes' if reasoning else 'no'} ({len(reasoning)} chars)")
    if reasoning:
        print(f"  excerpt:  {reasoning[:200].strip()}")
    print(f"  answer:   {msg.get('content', '').strip()[:200]}")
    print()
