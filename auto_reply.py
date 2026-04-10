"""
=============================================================================
COMP 262 - Phase 2 | LLM Task 2: Auto Reply System
=============================================================================

Goal:
Automatically generate a customer service style response to a review
that contains a question.
=============================================================================
"""

from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

INPUT_PATH = Path("results/metrics/phase2_test_predictions_all_models.csv")
OUTPUT_PATH = Path("results/metrics/auto_replies.csv")


# -----------------------------------------------------------------------------
# STEP 1: Load data
# -----------------------------------------------------------------------------
def load_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df):,} reviews")
    return df


# -----------------------------------------------------------------------------
# STEP 2: Select reviews that contain questions
# -----------------------------------------------------------------------------
def select_question_reviews(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Select reviews that are likely to contain product/customer questions.
    Uses a strict filter first, then progressively relaxes it if too few matches exist.
    """
    if "reviewText" not in df.columns:
        raise ValueError("Column 'reviewText' not found.")

    df = df.copy()
    review_series = df["reviewText"].fillna("").astype(str).str.strip()
    review_lower = review_series.str.lower()

    has_qmark = review_series.str.contains("?", na=False, regex=False)

    strong_question_patterns = [
        "does this", "does it", "can this", "can it", "will this", "will it",
        "is this", "is it", "would this", "would it", "could this", "could it",
        "how do i", "how does", "what is", "what are", "works with", "work with",
        "compatible with", "fit ", "fits ", "use with", "safe for", "good for"
    ]
    medium_question_patterns = [
        "how ", "what ", "why ", "when ", "where ", "which ", "do ", "does ",
        "can ", "will ", "is ", "are ", "could ", "would ", "should "
    ]

    has_strong_question_phrase = review_lower.apply(
        lambda x: any(p in x for p in strong_question_patterns)
    )
    has_medium_question_phrase = review_lower.apply(
        lambda x: any(p in x for p in medium_question_patterns)
    )

    excluded_starts = [
        "what can you say", "how we love", "who knew", "guess what", "you know how"
    ]
    bad_opening = review_lower.apply(lambda x: any(x.startswith(p) for p in excluded_starts))

    noisy_patterns = [
        "aggg", "volunteer for this crap", "wtf", "omg", "!!!", "????", "crap"
    ]
    is_noisy = review_lower.apply(lambda x: any(p in x for p in noisy_patterns))

    word_count = review_series.str.split().str.len()
    medium_length = word_count.between(6, 80)
    relaxed_length = word_count.between(4, 100)
    ends_with_q = review_series.str.rstrip().str.endswith("?")

    # Pass 1: strict
    strict_df = df[
        has_qmark
        & has_strong_question_phrase
        & ~bad_opening
        & ~is_noisy
        & medium_length
        & ends_with_q
    ].copy()

    if len(strict_df) >= n:
        selected = strict_df.head(n).copy()
        print(f"[INFO] Selected {len(selected)} strict question-based reviews")
        return selected

    # Pass 2: relaxed but still question-oriented
    relaxed_df = df[
        has_qmark
        & (has_strong_question_phrase | has_medium_question_phrase)
        & ~bad_opening
        & ~is_noisy
        & relaxed_length
    ].copy()

    if len(relaxed_df) >= n:
        selected = relaxed_df.head(n).copy()
        print(f"[INFO] Selected {len(selected)} relaxed question-based reviews")
        return selected

    # Pass 3: fallback to any literal-question reviews that are not noisy
    fallback_df = df[
        has_qmark
        & ~bad_opening
        & ~is_noisy
        & relaxed_length
    ].copy()

    if len(fallback_df) == 0:
        raise ValueError("No usable question-like reviews found.")

    selected = fallback_df.head(min(n, len(fallback_df))).copy()
    print(f"[WARN] Only {len(selected)} fallback question-like reviews found after relaxing filters")
    return selected


# -----------------------------------------------------------------------------
# STEP 3: Generate replies using FLAN-T5
# -----------------------------------------------------------------------------
def generate_replies(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Loading model for auto-reply...")

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    replies = []

    for i, row in df.iterrows():
        review = str(row["reviewText"])

        prompt = (
            "You are a customer support representative for an online store. "
            "Write a short, polite, professional reply to the customer's question. "
            "Do not copy the review. Do not summarize it. Do not repeat the same wording. "
            "Answer clearly in 2 to 4 sentences. If the review does not provide enough information, "
            "respond helpfully and suggest checking the product specifications or contacting support.\n\n"
            f"Customer review:\n{review}\n\n"
            "Customer support reply:"
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
                max_new_tokens=100,
                min_new_tokens=25,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.2,
                length_penalty=1.0
            )

            reply = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        except Exception as e:
            reply = f"ERROR: {e}"

        replies.append(reply)
        print(f"[DONE] Reply {len(replies)} generated")

    out = df.copy()
    out["auto_reply"] = replies

    return out


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    df = load_data(INPUT_PATH)
    questions = select_question_reviews(df, n=5)
    replies = generate_replies(questions)

    replies.to_csv(OUTPUT_PATH, index=False)

    print("\nSample replies (filtered question reviews):")

    for _, row in replies.head(2).iterrows():
        print("\n" + "=" * 80)
        print("CUSTOMER REVIEW:")
        print("-" * 80)
        print(row["reviewText"])

        print("\nAUTO REPLY:")
        print("-" * 80)
        print(row["auto_reply"])
        print("=" * 80)

    print(f"\n[SAVED] Auto replies → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()