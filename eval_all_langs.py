# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Evaluate a model on Belebele across all available languages, round-robin."""

import argparse
import json
import os
import signal
import sys
import time
from openai import OpenAI
from pathlib import Path

DATA_DIR = Path("data/belebele")
STATE_FILE = DATA_DIR / ".eval_state.json"
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


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def render(langs: list[str], state: dict, totals: dict, elapsed: float | None):
    """Render the stats table in-place."""
    lines = []
    lines.append(f"{'Language':<12} {'Done':>6} {'Total':>6} {'Correct':>8} {'Wrong':>6} {'Errors':>7} {'Acc':>7}")
    lines.append("-" * 60)
    total_done = 0
    total_correct = 0
    total_total = 0
    for lang in langs:
        s = state.get(lang, {})
        done = s.get("done", 0)
        correct = s.get("correct", 0)
        wrong = s.get("wrong", 0)
        errors = s.get("errors", 0)
        total = totals[lang]
        acc = f"{correct/done*100:.1f}%" if done > 0 else "-"
        lines.append(f"{lang:<12} {done:>6} {total:>6} {correct:>8} {wrong:>6} {errors:>7} {acc:>7}")
        total_done += done
        total_correct += correct
        total_total += total
    lines.append("-" * 60)
    overall_acc = f"{total_correct/total_done*100:.1f}%" if total_done > 0 else "-"
    lines.append(f"{'TOTAL':<12} {total_done:>6} {total_total:>6} {total_correct:>8} {total_done-total_correct:>6} {'':>7} {overall_acc:>7}")
    lines.append(f"\nLast question: {elapsed:.2f}s" if elapsed is not None else "")
    # Move cursor to top of table and overwrite
    output = "\n".join(lines) + "\n"
    num_lines = output.count("\n")
    if render.prev_lines > 0:
        sys.stdout.write(f"\033[{render.prev_lines}A\033[J")
    sys.stdout.write(output)
    sys.stdout.flush()
    render.prev_lines = num_lines

render.prev_lines = 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate all languages on Belebele.")
    parser.add_argument("-n", type=int, default=900, help="Max questions per language (default: 900)")
    parser.add_argument("--reset", action="store_true", help="Reset saved progress")
    args = parser.parse_args()

    # Discover languages
    lang_files = sorted(DATA_DIR.glob("*.jsonl"))
    langs = [f.stem for f in lang_files]
    if not langs:
        print("No datasets found in data/belebele/")
        return

    # Load all questions
    data: dict[str, list] = {}
    totals: dict[str, int] = {}
    for lang, path in zip(langs, lang_files):
        with open(path) as f:
            items = [json.loads(line) for line in f]
        data[lang] = items[:args.n]
        totals[lang] = len(data[lang])

    # Load or reset state
    if args.reset:
        state = {}
    else:
        state = load_state()

    # Handle Ctrl+C gracefully
    stopping = False
    def handle_signal(sig, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, handle_signal)

    # Round-robin
    while not stopping:
        made_progress = False
        for lang in langs:
            if stopping:
                break
            s = state.setdefault(lang, {"done": 0, "correct": 0, "wrong": 0, "errors": 0})
            idx = s["done"]
            if idx >= len(data[lang]):
                continue

            made_progress = True
            item = data[lang][idx]
            prompt = build_prompt(item)
            expected = LABELS[int(item["correct_answer_num"]) - 1]

            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model="loaded-model",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )
                raw = response.choices[0].message.content
                predicted = parse_answer(raw)
            except Exception:
                s["errors"] += 1
                s["done"] += 1
                predicted = None
                save_state(state)
                render(langs, state, totals, time.perf_counter() - t0)
                continue

            elapsed = time.perf_counter() - t0

            if predicted == expected:
                s["correct"] += 1
            else:
                s["wrong"] += 1
            s["done"] += 1

            save_state(state)
            render(langs, state, totals, elapsed)

        if not made_progress:
            break

    # Final state
    save_state(state)
    if stopping:
        print("\nStopped. Progress saved. Run again to resume.")


if __name__ == "__main__":
    main()
