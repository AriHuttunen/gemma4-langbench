# /// script
# requires-python = ">=3.12"
# dependencies = ["datasets"]
# ///
"""Download Belebele benchmark datasets for target languages."""

from datasets import load_dataset
from pathlib import Path

LANGUAGES = ["eng_Latn", "fin_Latn", "swe_Latn", "est_Latn"]
OUTPUT_DIR = Path("data/belebele")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in LANGUAGES:
        print(f"Downloading {lang}...")
        ds = load_dataset("facebook/belebele", lang, split="test")
        path = OUTPUT_DIR / f"{lang}.jsonl"
        ds.to_json(path)
        print(f"  Saved {len(ds)} examples to {path}")


if __name__ == "__main__":
    main()
