# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///
"""Evaluate a model on Belebele across all available languages, round-robin."""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

DATA_DIR = Path("data/belebele")
RUNS_DIR = Path("runs")
LABELS = ["A", "B", "C", "D"]

INSTRUCTION = (
    "Read the passage, the query, and the choices. "
    "Output only the letter (A, B, C, or D) corresponding to the correct answer."
)


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


def parse_answer(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip().upper()
    for label in LABELS:
        if text.startswith(f"({label})") or text.startswith(label):
            return label
    return None


def safe_run_id(model: str, thinking: bool) -> str:
    safe = model.replace("/", "_")
    return f"{safe}_thinking" if thinking else safe


def run_dir_for(run_id: str) -> Path:
    return RUNS_DIR / run_id


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def prompt_sha() -> str:
    return hashlib.sha256(INSTRUCTION.encode()).hexdigest()[:16]


def meta_api_kwargs(thinking: bool) -> dict:
    """Clean parameter record for the state meta block."""
    if thinking:
        return {"temperature": 1, "max_tokens": 2500, "reasoning_max_tokens": 2000}
    return {"temperature": 0, "max_tokens": 2048, "reasoning_max_tokens": 0}


def model_api_kwargs(thinking: bool) -> dict:
    """Full kwargs dict for the OpenAI client."""
    if thinking:
        return {
            "temperature": 1,
            "max_tokens": 2500,
            "extra_body": {"reasoning": {"type": "enabled", "max_tokens": 2000}},
        }
    return {"temperature": 0, "max_tokens": 2048}


def q_key(item: dict) -> str:
    return f"{item['link']}|{item['question_number']}"


def find_next_item(items: list, lang_results: dict) -> dict | None:
    for item in items:
        if q_key(item) not in lang_results:
            return item
    return None


def derived_counts(lang_results: dict) -> dict:
    done = len(lang_results)
    correct = sum(1 for r in lang_results.values() if r["outcome"] == "correct")
    errors = sum(1 for r in lang_results.values() if r["outcome"] == "api_error")
    wrong = done - correct - errors
    return {"done": done, "correct": correct, "wrong": wrong, "errors": errors}


def mark_completed(state: dict, langs: list, n: int) -> None:
    if state["meta"].get("completed_at"):
        return
    if all(len(state["languages"].get(lang, {})) >= n for lang in langs):
        state["meta"]["completed_at"] = datetime.now(timezone.utc).isoformat()


def append_log(log_path: Path, entry: dict) -> None:
    with log_path.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_log(run_log_path: Path, msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with run_log_path.open("a") as f:
        f.write(f"{ts} {msg}\n")


def english_block(item: dict) -> dict:
    correct_num = int(item["correct_answer_num"])
    return {
        "passage": item["flores_passage"],
        "question": item["question"],
        "choices": {
            "A": item["mc_answer1"],
            "B": item["mc_answer2"],
            "C": item["mc_answer3"],
            "D": item["mc_answer4"],
        },
        "correct_label": LABELS[correct_num - 1],
        "correct_text": item[f"mc_answer{correct_num}"],
    }


def build_log_entry(
    lang: str,
    item: dict,
    model: str,
    base_url: str,
    error_type: str,
    predicted: str | None,
    raw_response: str | None,
    elapsed: float | None,
    thinking: bool = False,
    akm: dict | None = None,
    english_item: dict | None = None,
    error_message: str | None = None,
) -> dict:
    correct_num = int(item["correct_answer_num"])
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": base_url,
        "thinking": thinking,
        "temperature": (akm or {}).get("temperature"),
        "max_tokens": (akm or {}).get("max_tokens"),
        "reasoning_max_tokens": (akm or {}).get("reasoning_max_tokens"),
        "language": lang,
        "dialect": item.get("dialect"),
        "link": item.get("link"),
        "question_number": item.get("question_number"),
        "error_type": error_type,
        "correct_label": LABELS[correct_num - 1],
        "correct_text": item[f"mc_answer{correct_num}"],
        "predicted_label": predicted,
        "raw_response": raw_response,
        "error_message": error_message,
        "passage": item["flores_passage"],
        "question": item["question"],
        "choices": {
            "A": item["mc_answer1"],
            "B": item["mc_answer2"],
            "C": item["mc_answer3"],
            "D": item["mc_answer4"],
        },
        "elapsed_seconds": elapsed,
    }
    if english_item is not None and lang != "eng_Latn":
        entry["english"] = english_block(english_item)
    return entry


def load_state(run_dir: Path) -> dict:
    path = run_dir / "state.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"meta": {}, "languages": {}}


def save_state(state: dict, run_dir: Path) -> None:
    path = run_dir / "state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


recent_times: list[float] = []


def render(
    langs: list[str],
    state: dict,
    totals: dict,
    elapsed,
    model: str,
    thinking: bool,
    dry_run: bool = False,
):
    lines = []
    tag = " [thinking]" if thinking else ""
    if dry_run:
        tag += " [dry-run]"
    lines.append(f"Model: {model}{tag}")
    lines.append(
        f"{'Language':<12} {'Done':>6} {'Total':>6} {'Correct':>8} {'Wrong':>6} {'Errors':>7} {'Acc':>7}"
    )
    lines.append("-" * 60)
    total_done = 0
    total_correct = 0
    total_total = 0
    for lang in langs:
        counts = derived_counts(state["languages"].get(lang, {}))
        done = counts["done"]
        correct = counts["correct"]
        wrong = counts["wrong"]
        errors = counts["errors"]
        total = totals[lang]
        acc = f"{correct / done * 100:.1f}%" if done > 0 else "-"
        lines.append(
            f"{lang:<12} {done:>6} {total:>6} {correct:>8} {wrong:>6} {errors:>7} {acc:>7}"
        )
        total_done += done
        total_correct += correct
        total_total += total
    lines.append("-" * 60)
    overall_acc = f"{total_correct / total_done * 100:.1f}%" if total_done > 0 else "-"
    lines.append(
        f"{'TOTAL':<12} {total_done:>6} {total_total:>6} {total_correct:>8} {total_done - total_correct:>6} {'':>7} {overall_acc:>7}"
    )
    if elapsed is not None:
        if isinstance(elapsed, list) and elapsed:
            recent_times.extend(elapsed)
        elif isinstance(elapsed, (int, float)):
            recent_times.append(elapsed)
        last10 = recent_times[-10:]
        if last10:
            lines.append(f"\n{'':>12} {'min':>8} {'avg':>8} {'max':>8}")
            batch = (
                elapsed
                if isinstance(elapsed, list) and elapsed
                else [elapsed]
                if isinstance(elapsed, (int, float))
                else []
            )
            if batch:
                lines.append(
                    f"{'Last batch:':<12} {min(batch):>7.2f}s {sum(batch) / len(batch):>7.2f}s {max(batch):>7.2f}s"
                )
            lines.append(
                f"{'Last 10:':<12} {min(last10):>7.2f}s {sum(last10) / len(last10):>7.2f}s {max(last10):>7.2f}s"
            )
    else:
        lines.append("")
    output = "\n".join(lines) + "\n"
    num_lines = output.count("\n")
    if render.prev_lines > 0:
        sys.stdout.write(f"\033[{render.prev_lines}A\033[J")
    sys.stdout.write(output)
    sys.stdout.flush()
    render.prev_lines = num_lines


render.prev_lines = 0


async def query_model(async_client, model: str, prompt: str, api_kwargs: dict):
    t0 = time.perf_counter()
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **api_kwargs,
    )
    elapsed = time.perf_counter() - t0
    msg = response.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    return msg.content, reasoning, elapsed


def run_sequential(
    langs, data, state, totals, run_dir, model, api_kwargs, akm, thinking,
    base_url, api_key, log_path, eng_index, rl, dry_run,
):
    """Sequential round-robin for local models."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    n = max(totals.values(), default=0)

    stopping = False

    def handle_signal(sig, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, handle_signal)

    while not stopping:
        made_progress = False
        for lang in langs:
            if stopping:
                break
            lang_results = state["languages"].setdefault(lang, {})
            item = find_next_item(data[lang], lang_results)
            if item is None:
                continue

            made_progress = True
            key = q_key(item)
            prompt = build_prompt(item)
            expected = LABELS[int(item["correct_answer_num"]) - 1]

            t0 = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **api_kwargs,
                )
                raw = response.choices[0].message.content
                predicted = parse_answer(raw)
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"\nError ({lang}): {e}", file=sys.stderr)
                lang_results[key] = {"outcome": "api_error", "predicted": None, "correct": expected, "elapsed": elapsed}
                if not dry_run:
                    run_log(rl, f"ERROR lang={lang} key={key} {e}")
                    eng = eng_index.get((item.get("link"), item.get("question_number")))
                    append_log(log_path, build_log_entry(
                        lang, item, model, base_url, "api_error",
                        None, None, elapsed,
                        thinking=thinking, akm=akm,
                        english_item=eng, error_message=str(e),
                    ))
                    mark_completed(state, langs, n)
                    save_state(state, run_dir)
                render(langs, state, totals, elapsed, model, thinking, dry_run)
                continue

            elapsed = time.perf_counter() - t0
            if predicted == expected:
                outcome = "correct"
            elif predicted is None:
                outcome = "unparseable"
            else:
                outcome = "wrong_answer"

            lang_results[key] = {"outcome": outcome, "predicted": predicted, "correct": expected, "elapsed": elapsed}

            if outcome != "correct" and not dry_run:
                eng = eng_index.get((item.get("link"), item.get("question_number")))
                append_log(log_path, build_log_entry(
                    lang, item, model, base_url, outcome,
                    predicted, raw, elapsed,
                    thinking=thinking, akm=akm,
                    english_item=eng,
                ))

            if not dry_run:
                mark_completed(state, langs, n)
                save_state(state, run_dir)
            render(langs, state, totals, elapsed, model, thinking, dry_run)

        if not made_progress:
            break

    if not dry_run:
        save_state(state, run_dir)
    if stopping:
        msg = "Stopped." if dry_run else "Stopped. Progress saved. Run again to resume."
        print(f"\n{msg}")


async def run_parallel(
    langs, data, state, totals, run_dir, model, api_kwargs, akm, thinking,
    base_url, api_key, log_path, eng_index, rl, dry_run,
):
    """Parallel requests for cloud APIs — one request per language concurrently."""
    async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    n = max(totals.values(), default=0)

    stopping = False

    def handle_signal(sig, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, handle_signal)

    while not stopping:
        pending = []
        for lang in langs:
            lang_results = state["languages"].setdefault(lang, {})
            item = find_next_item(data[lang], lang_results)
            if item is not None:
                prompt = build_prompt(item)
                expected = LABELS[int(item["correct_answer_num"]) - 1]
                pending.append((lang, item, prompt, expected))

        if not pending:
            break

        tasks = [
            query_model(async_client, model, prompt, api_kwargs)
            for _, _, prompt, _ in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_times = []
        for (lang, item, _, expected), result in zip(pending, results):
            lang_results = state["languages"][lang]
            key = q_key(item)
            if isinstance(result, Exception):
                print(f"\nError ({lang}): {result}", file=sys.stderr)
                lang_results[key] = {"outcome": "api_error", "predicted": None, "correct": expected, "elapsed": None}
                if not dry_run:
                    run_log(rl, f"ERROR lang={lang} key={key} {result}")
                    eng = eng_index.get((item.get("link"), item.get("question_number")))
                    append_log(log_path, build_log_entry(
                        lang, item, model, base_url, "api_error",
                        None, None, None,
                        thinking=thinking, akm=akm,
                        english_item=eng, error_message=str(result),
                    ))
            else:
                raw, reasoning, elapsed = result
                batch_times.append(elapsed)
                if not dry_run:
                    run_log(rl, f"RESPONSE lang={lang} reasoning_chars={len(reasoning) if reasoning else 0}")
                predicted = parse_answer(raw)
                if predicted == expected:
                    outcome = "correct"
                elif predicted is None:
                    outcome = "unparseable"
                else:
                    outcome = "wrong_answer"

                lang_results[key] = {"outcome": outcome, "predicted": predicted, "correct": expected, "elapsed": elapsed}

                if outcome != "correct" and not dry_run:
                    eng = eng_index.get((item.get("link"), item.get("question_number")))
                    append_log(log_path, build_log_entry(
                        lang, item, model, base_url, outcome,
                        predicted, raw, elapsed,
                        thinking=thinking, akm=akm,
                        english_item=eng,
                    ))

        if not dry_run:
            mark_completed(state, langs, n)
            save_state(state, run_dir)
        render(langs, state, totals, batch_times, model, thinking, dry_run)

    if not dry_run:
        save_state(state, run_dir)
    if stopping:
        msg = "Stopped." if dry_run else "Stopped. Progress saved. Run again to resume."
        print(f"\n{msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on Belebele across all available languages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # LM Studio (local server on localhost:1234)
  uv run eval_all_langs_v2.py --local
  uv run eval_all_langs_v2.py --local --thinking

  # OpenRouter
  uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6
  uv run eval_all_langs_v2.py --model anthropic/claude-sonnet-4.6 --thinking
  uv run eval_all_langs_v2.py --model google/gemma-4-26b-a4b-it --thinking -n 100
""",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--local",
        action="store_true",
        help="Use LM Studio on localhost:1234 with the currently loaded model",
    )
    source.add_argument(
        "--model",
        metavar="MODEL",
        help="OpenRouter model id (e.g. anthropic/claude-sonnet-4.6)",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable extended thinking (temperature=1, reasoning budget 2000 tokens)",
    )
    parser.add_argument(
        "-n", type=int, default=900, help="Max questions per language (default: 900)"
    )
    parser.add_argument("--reset", action="store_true", help="Wipe runs/<run_id>/ and start fresh")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run evaluation without writing any files",
    )
    args = parser.parse_args()

    if args.local:
        model = "loaded-model"
        base_url = "http://localhost:1234/v1"
        api_key = "lm-studio"
        parallel = False
    else:
        model = args.model
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: OPENROUTER_API_KEY env var required for OpenRouter models")
            sys.exit(1)
        parallel = True

    api_kwargs = model_api_kwargs(args.thinking)
    akm = meta_api_kwargs(args.thinking)

    run_id = safe_run_id(model, args.thinking)
    run_dir = run_dir_for(run_id)
    log_path = run_dir / "wrong_answers.jsonl"
    rl = run_dir / "run.log"

    lang_files = sorted(DATA_DIR.glob("*.jsonl"))
    langs = [f.stem for f in lang_files]
    if not langs:
        print("No datasets found in data/belebele/")
        return

    data: dict[str, list] = {}
    totals: dict[str, int] = {}
    for lang, path in zip(langs, lang_files):
        with open(path) as f:
            items = [json.loads(line) for line in f]
        data[lang] = items[: args.n]
        totals[lang] = len(data[lang])

    eng_index: dict[tuple, dict] = {
        (item.get("link"), item.get("question_number")): item
        for item in data.get("eng_Latn", [])
    }

    if args.reset and run_dir.exists():
        shutil.rmtree(run_dir)

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run or args.reset:
        state: dict = {"meta": {}, "languages": {}}
    else:
        state = load_state(run_dir)

    now = datetime.now(timezone.utc).isoformat()
    if not state["meta"]:
        state["meta"] = {
            "model": model,
            "base_url": base_url,
            "thinking": args.thinking,
            "api_kwargs": akm,
            "n_per_language": args.n,
            "prompt_sha256": prompt_sha(),
            "git_sha": get_git_sha(),
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
        }
    else:
        state["meta"].update({
            "model": model,
            "base_url": base_url,
            "thinking": args.thinking,
            "api_kwargs": akm,
            "n_per_language": args.n,
            "updated_at": now,
        })

    print(f"Model:    {model}")
    print(f"Source:   {base_url}")
    print(f"Thinking: {'on' if args.thinking else 'off'}")
    print(f"Mode:     {'parallel (all languages at once)' if parallel else 'sequential (one language at a time)'}")
    print(f"Per lang: {args.n} questions")
    if args.dry_run:
        print("Files:    none (dry run)")
    else:
        print(f"Dir:      {run_dir}/")
    print()

    if not args.dry_run:
        run_log(rl, f"START model={model} thinking={args.thinking} base_url={base_url} parallel={parallel} n={args.n} reset={args.reset}")

    render(langs, state, totals, None, model, args.thinking, args.dry_run)

    runner_args = (
        langs, data, state, totals, run_dir, model, api_kwargs, akm, args.thinking,
        base_url, api_key, log_path, eng_index, rl, args.dry_run,
    )

    if parallel:
        asyncio.run(run_parallel(*runner_args))
    else:
        run_sequential(*runner_args)

    if not args.dry_run:
        run_log(rl, "DONE")


if __name__ == "__main__":
    main()
