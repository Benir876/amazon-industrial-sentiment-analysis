"""
=============================================================================
COMP 262 - Phase 2 | LLM Task 1: Review Summarization
=============================================================================

Goal:
Select 10 reviews longer than 100 words and summarize each one to ~50 words
using a Hugging Face model hosted locally.
=============================================================================
"""

from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_PATH = Path("results/metrics/phase2_test_predictions_all_models.csv")
OUTPUT_PATH = Path("results/metrics/review_summaries.csv")


def load_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df):,} reviews")
    return df


def select_long_reviews(df: pd.DataFrame, min_words: int = 100, n: int = 10) -> pd.DataFrame:
    if "reviewText" not in df.columns:
        raise ValueError("Column 'reviewText' not found. Make sure model_evaluation.py saved it.")

    df = df.copy()
    df["review_word_count"] = df["reviewText"].fillna("").astype(str).str.split().str.len()
    long_reviews = df[df["review_word_count"] > min_words].copy()

    if len(long_reviews) < n:
        print(f"[WARN] Only {len(long_reviews)} reviews longer than {min_words} words were found.")
        return long_reviews

    selected = long_reviews.head(n).copy()
    print(f"[INFO] Selected {len(selected)} reviews longer than {min_words} words")
    return selected


def summarize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Loading summarization model...")

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    summaries = []
    for i, row in df.iterrows():
        text = str(row["reviewText"])

        # Keep input manageable for the model
        trimmed_text = " ".join(text.split()[:300])
        prompt = (
            "Summarize the following product review in about 50 words, "
            "keeping the main opinion, product quality, and user experience:\n\n"
            f"{trimmed_text}"
        )

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            output_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                min_new_tokens=35,
                do_sample=False
            )
            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            summary = f"ERROR: {e}"

        summaries.append(summary)
        print(f"[DONE] Review {len(summaries)} summarized")

    out = df.copy()
    out["summary_50_words"] = summaries
    return out


def main():
    df = load_data(INPUT_PATH)
    selected = select_long_reviews(df, min_words=100, n=10)
    summarized = summarize_reviews(selected)

    summarized.to_csv(OUTPUT_PATH, index=False)

    print("\nFirst 2 summaries (formatted):")
    preview = summarized.head(2)

    for idx, row in preview.iterrows():
        original = str(row.get("reviewText", ""))
        summary = str(row.get("summary_50_words", ""))

        print("\n" + "=" * 80)
        print("ORIGINAL REVIEW:")
        print("-" * 80)
        print(original)
        print("\nSUMMARY (~50 words):")
        print("-" * 80)
        print(summary)
        print("=" * 80)

    print(f"\n[SAVED] Summaries → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()