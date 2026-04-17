# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Evaluate a model on the first N Belebele questions via LM Studio."""

import argparse
import json
import time
from openai import OpenAI
from pathlib import Path

LABELS = ["A", "B", "C", "D"]

INSTRUCTION = (
    "Read the passage, the query, and the choices. "
    "Output only the letter (A, B, C, or D) corresponding to the correct answer."
)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


def build_prompt(item: dict) -> str:
    return (
        f"{INSTRUCTION}\n"
        f"###\n"
        f"Passage:\n{item['flores_passage']}\n"
        f"###\n"
        f"Query:\n{item['question']}\n"
        f"###\n"
        f"Choices:\n"
        f"(A) {item['mc_answer1']}\n"
        f"(B) {item['mc_answer2']}\n"
        f"(C) {item['mc_answer3']}\n"
        f"(D) {item['mc_answer4']}\n"
        f"###\n"
        f"Answer:"
    )


def parse_answer(text: str) -> str | None:
    text = text.strip().upper()
    for label in LABELS:
        if text.startswith(f"({label})") or text.startswith(label):
            return label
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on Belebele.")
    parser.add_argument("-n", type=int, default=100, help="Max questions (default: 100)")
    parser.add_argument("-l", "--lang", default="eng_Latn", help="Language code (default: eng_Latn)")
    args = parser.parse_args()

    dataset = Path(f"data/belebele/{args.lang}.jsonl")
    print(f"Language: {args.lang} | Max questions: {args.n}")

    items = []
    with open(dataset) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            items.append(json.loads(line))

    correct = 0
    times = []
    for i, item in enumerate(items, 1):
        prompt = build_prompt(item)
        expected = LABELS[int(item["correct_answer_num"]) - 1]

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model="loaded-model",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10000,
            temperature=0,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        raw = response.choices[0].message.content
        predicted = parse_answer(raw)

        status = "OK" if predicted == expected else "WRONG"
        if predicted is None:
            status = "PARSE_ERROR"
        if predicted == expected:
            correct += 1

        print(f"Q{i}: expected={expected} predicted={predicted} raw={raw!r} [{status}]")
        if i % 10 == 0:
            wrong = i - correct
            print(f"  --- {i} questions: {correct} correct, {wrong} wrong, {correct/i*100:.1f}% ---")
            batch = times[-10:]
            timing_summary = (
                f"  --- Q{i-9}-Q{i}: min={min(batch):.2f}s max={max(batch):.2f}s "
                f"avg={sum(batch)/len(batch):.2f}s total={sum(batch):.1f}s "
                f"| cumulative avg={sum(times)/len(times):.2f}s ---"
            )
            print(timing_summary)

    print(f"\nLanguage: {args.lang}")
    print(f"Accuracy: {correct}/{len(items)} ({correct/len(items)*100:.0f}%)")


if __name__ == "__main__":
    main()
