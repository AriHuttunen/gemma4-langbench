#!/usr/bin/env python3
"""Analyze wrong_answers_*.jsonl logs, joining with Belebele source data.

Outputs to stats/:
  accuracy.csv            — per-model per-language accuracy (asserted vs eval_state)
  hardest_questions.csv   — 900 questions ranked by total failure count across models × langs
  language_flip.csv       — questions a model got right in English but wrong in est/fin/swe
  model_disagreement.csv  — per lang: which models got each question right vs wrong
  SUMMARY.md              — human-readable rollup
"""

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data" / "belebele"
STATS_DIR = REPO_ROOT / "stats"
LANGUAGES = ["eng_Latn", "est_Latn", "fin_Latn", "swe_Latn"]
LANG_SHORT = {"eng_Latn": "eng", "est_Latn": "est", "fin_Latn": "fin", "swe_Latn": "swe"}

# wrong_answers filename stem → eval_state key (only needed where they differ)
EVAL_STATE_MAP = {
    "local_Gemma-4_E4B_Q8_0_think": "loaded-model",
}


def load_belebele():
    """Return (qdata, reverse_idx).

    qdata[(link, q_no)][lang] = {question, passage}
    reverse_idx[lang][(link, question_text)] = question_no
    """
    qdata = {}
    reverse_idx = {lang: {} for lang in LANGUAGES}

    for lang in LANGUAGES:
        path = DATA_DIR / f"{lang}.jsonl"
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                link = r["link"]
                q_no = int(r["question_number"])
                q_text = r["question"]
                key = (link, q_no)
                if key not in qdata:
                    qdata[key] = {}
                qdata[key][lang] = {"question": q_text, "passage": r["flores_passage"]}
                reverse_idx[lang][(link, q_text)] = q_no

    return qdata, reverse_idx


def load_wrong_answers(reverse_idx):
    """Return (outcome, wrong_counts, model_ids).

    outcome[(link, q_no)][(model_id, lang)] = error_type string
    wrong_counts[model_id][lang][error_type] = int
    model_ids = list of model_id strings (from filenames, in sorted order)
    """
    outcome = defaultdict(dict)
    wrong_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    files = sorted(glob.glob(str(REPO_ROOT / "wrong_answers_*.jsonl")))
    if not files:
        print("ERROR: no wrong_answers_*.jsonl files found", file=sys.stderr)
        sys.exit(1)

    model_ids = [Path(f).stem[len("wrong_answers_"):] for f in files]
    unmatched = []

    for fpath, model_id in zip(files, model_ids):
        with open(fpath) as f:
            for lineno, line in enumerate(f, 1):
                r = json.loads(line)
                lang = r["language"]
                if lang not in LANGUAGES:
                    continue
                link = r["link"]
                q_text = r["question"]
                error_type = r["error_type"]

                q_no = reverse_idx[lang].get((link, q_text))
                if q_no is None:
                    q_no = reverse_idx[lang].get((link, q_text.strip()))
                if q_no is None:
                    unmatched.append((fpath, lineno, lang, link, q_text[:60]))
                    continue

                outcome[(link, q_no)][(model_id, lang)] = error_type
                wrong_counts[model_id][lang][error_type] += 1

    if unmatched:
        print(f"WARNING: {len(unmatched)} wrong-answer records could not be matched:", file=sys.stderr)
        for u in unmatched[:5]:
            print(f"  {u}", file=sys.stderr)

    return outcome, wrong_counts, model_ids


def load_eval_states(model_ids):
    """Return dict[model_id][lang] = {done, correct, wrong, errors}."""
    states = {}
    for model_id in model_ids:
        eval_key = EVAL_STATE_MAP.get(model_id, model_id)
        path = DATA_DIR / f".eval_state_{eval_key}.json"
        if path.exists():
            with open(path) as f:
                states[model_id] = json.load(f)
        else:
            print(f"WARNING: no eval_state for model '{model_id}' (tried {path})", file=sys.stderr)
    return states


def is_wrong(outcome, link, q_no, model_id, lang):
    return (model_id, lang) in outcome.get((link, q_no), {})


def main():
    STATS_DIR.mkdir(exist_ok=True)

    print("Loading Belebele source data...")
    qdata, reverse_idx = load_belebele()
    all_keys = sorted(qdata.keys())
    assert len(all_keys) == 900, f"Expected 900 questions, got {len(all_keys)}"

    print("Loading wrong-answer logs...")
    outcome, wrong_counts, model_ids = load_wrong_answers(reverse_idx)
    print(f"  Models: {model_ids}")

    eval_states = load_eval_states(model_ids)

    # -------------------------------------------------------------------------
    # 1. Accuracy CSV (with assertion vs eval_state)
    # -------------------------------------------------------------------------
    print("Writing accuracy.csv...")
    acc_rows = []
    for model_id in model_ids:
        for lang in LANGUAGES:
            n_wrong_log = sum(wrong_counts[model_id][lang].values())
            if model_id in eval_states and lang in eval_states[model_id]:
                st = eval_states[model_id][lang]
                n_wrong_state = st["wrong"] + st.get("errors", 0)
                if n_wrong_log != n_wrong_state:
                    print(
                        f"ASSERTION FAILED: {model_id}/{lang}: "
                        f"log={n_wrong_log} vs eval_state={n_wrong_state}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            n_correct = 900 - n_wrong_log
            acc_rows.append({
                "model": model_id,
                "language": lang,
                "correct": n_correct,
                "wrong": n_wrong_log,
                "accuracy": f"{n_correct / 900:.4f}",
            })

    _write_csv(STATS_DIR / "accuracy.csv",
               ["model", "language", "correct", "wrong", "accuracy"], acc_rows)
    print("  Accuracy assertions passed.")

    # -------------------------------------------------------------------------
    # 2. Hardest questions CSV (all 900, sorted by n_wrong_total desc)
    # -------------------------------------------------------------------------
    print("Writing hardest_questions.csv...")
    hq_rows = []
    for (link, q_no) in all_keys:
        eng_q = qdata[(link, q_no)]["eng_Latn"]["question"]
        lang_wrong = {
            lang: sum(1 for mid in model_ids if is_wrong(outcome, link, q_no, mid, lang))
            for lang in LANGUAGES
        }
        hq_rows.append({
            "link": link,
            "question_number": q_no,
            "eng_question": eng_q,
            "n_wrong_total": sum(lang_wrong.values()),
            "n_wrong_eng": lang_wrong["eng_Latn"],
            "n_wrong_est": lang_wrong["est_Latn"],
            "n_wrong_fin": lang_wrong["fin_Latn"],
            "n_wrong_swe": lang_wrong["swe_Latn"],
        })
    hq_rows.sort(key=lambda r: (-r["n_wrong_total"], r["link"], r["question_number"]))

    _write_csv(STATS_DIR / "hardest_questions.csv",
               ["link", "question_number", "eng_question",
                "n_wrong_total", "n_wrong_eng", "n_wrong_est", "n_wrong_fin", "n_wrong_swe"],
               hq_rows)

    # -------------------------------------------------------------------------
    # 3. Language flip CSV
    # -------------------------------------------------------------------------
    print("Writing language_flip.csv...")
    non_eng = [l for l in LANGUAGES if l != "eng_Latn"]
    flip_rows = []
    for model_id in model_ids:
        for (link, q_no) in all_keys:
            if is_wrong(outcome, link, q_no, model_id, "eng_Latn"):
                continue  # only interested where English was correct
            wrong_langs = [l for l in non_eng if is_wrong(outcome, link, q_no, model_id, l)]
            if not wrong_langs:
                continue
            flip_rows.append({
                "model": model_id,
                "link": link,
                "question_number": q_no,
                "eng_question": qdata[(link, q_no)]["eng_Latn"]["question"],
                "wrong_in": ",".join(LANG_SHORT[l] for l in wrong_langs),
                "n_wrong_langs": len(wrong_langs),
            })
    flip_rows.sort(key=lambda r: (r["model"], -r["n_wrong_langs"], r["link"], r["question_number"]))

    _write_csv(STATS_DIR / "language_flip.csv",
               ["model", "link", "question_number", "eng_question", "wrong_in", "n_wrong_langs"],
               flip_rows)

    # -------------------------------------------------------------------------
    # 4. Model disagreement CSV
    # -------------------------------------------------------------------------
    print("Writing model_disagreement.csv...")
    dis_rows = []
    for lang in LANGUAGES:
        for (link, q_no) in all_keys:
            correct_models = [mid for mid in model_ids if not is_wrong(outcome, link, q_no, mid, lang)]
            wrong_models = [mid for mid in model_ids if is_wrong(outcome, link, q_no, mid, lang)]
            dis_rows.append({
                "language": lang,
                "link": link,
                "question_number": q_no,
                "question": qdata[(link, q_no)][lang]["question"],
                "n_correct": len(correct_models),
                "correct_models": "|".join(correct_models),
                "wrong_models": "|".join(wrong_models),
            })
    dis_rows.sort(key=lambda r: (r["language"], r["n_correct"], r["link"], r["question_number"]))

    _write_csv(STATS_DIR / "model_disagreement.csv",
               ["language", "link", "question_number", "question",
                "n_correct", "correct_models", "wrong_models"],
               dis_rows)

    # -------------------------------------------------------------------------
    # 5. SUMMARY.md
    # -------------------------------------------------------------------------
    print("Writing SUMMARY.md...")

    flip_by_model = defaultdict(int)
    for row in flip_rows:
        flip_by_model[row["model"]] += 1

    disagree_by_lang = defaultdict(int)
    for row in dis_rows:
        if row["n_correct"] in (1, 3):
            disagree_by_lang[row["language"]] += 1

    unparse_by_model = {
        mid: sum(wrong_counts[mid][l]["unparseable"] for l in LANGUAGES)
        for mid in model_ids
    }

    lines = [
        "# Belebele Evaluation Summary\n",
        "",
        "4 models × 4 languages × 900 questions = 3,600 prompts per model.\n",
        "",
        "## Accuracy by Model and Language",
        "",
        "| Model | eng_Latn | est_Latn | fin_Latn | swe_Latn |",
        "|-------|----------|----------|----------|----------|",
    ]
    for mid in model_ids:
        cells = []
        for lang in LANGUAGES:
            row = next(r for r in acc_rows if r["model"] == mid and r["language"] == lang)
            cells.append(f"{float(row['accuracy']) * 100:.1f}%")
        lines.append(f"| {mid} | {' | '.join(cells)} |")

    lines += [
        "",
        "## Top 20 Hardest Questions (across all models × languages)",
        "",
        "n_wrong_total is out of 16 (4 models × 4 languages).",
        "",
        "| # | q_no | n_wrong | eng | est | fin | swe | Question | Link |",
        "|---|------|---------|-----|-----|-----|-----|----------|------|",
    ]
    for i, row in enumerate(hq_rows[:20], 1):
        q_trunc = row["eng_question"][:70].replace("|", "\\|")
        lines.append(
            f"| {i} | {row['question_number']} | {row['n_wrong_total']} "
            f"| {row['n_wrong_eng']} | {row['n_wrong_est']} "
            f"| {row['n_wrong_fin']} | {row['n_wrong_swe']} "
            f"| {q_trunc} | [link]({row['link']}) |"
        )

    lines += [
        "",
        "## Language Flip: English-correct → Non-English wrong",
        "",
        "Questions where the model answered English correctly but failed in ≥1 of est/fin/swe.",
        "",
        "| Model | Flip count |",
        "|-------|------------|",
    ]
    for mid in model_ids:
        lines.append(f"| {mid} | {flip_by_model[mid]} |")

    lines += [
        "",
        "## Model Disagreement (unique outlier per language)",
        "",
        "Questions where exactly 1 or 3 of 4 models got it right — one model stands out.",
        "",
        "| Language | Disagreement count |",
        "|----------|--------------------|",
    ]
    for lang in LANGUAGES:
        lines.append(f"| {lang} | {disagree_by_lang[lang]} |")

    lines += [
        "",
        "## Note: Unparseable Responses",
        "",
        "Counted as wrong in all accuracy figures above (matches eval_state.json).",
        "",
        "| Model | Unparseable count |",
        "|-------|-------------------|",
    ]
    for mid in model_ids:
        lines.append(f"| {mid} | {unparse_by_model[mid]} |")

    lines += [
        "",
        "## Output Files",
        "",
        "- `accuracy.csv` — per-model per-language accuracy (verified against eval_state)",
        "- `hardest_questions.csv` — all 900 questions sorted by failure count across models×langs",
        "- `language_flip.csv` — questions eng-correct but non-eng-wrong, per model",
        "- `model_disagreement.csv` — per-question per-language which models got it right/wrong",
        "",
    ]

    with open(STATS_DIR / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines))

    print(f"Done. Files written to {STATS_DIR}/")


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
