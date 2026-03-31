"""
=============================================================================
COMP 262 - Phase 1 / Phase 2 
Task: Text Pre-processing & Labeling
Dataset: Amazon Industrial & Scientific Reviews
=============================================================================
This script now supports BOTH phases:

Phase 1 deliverables covered:
  - Deliverable 2a: Label data (Positive / Neutral / Negative)
  - Deliverable 2b: Column selection with justification
  - Deliverable 2c: Outlier check on text
  - Deliverable 4:  Pre-process text for VADER and TextBlob
  - Deliverable 5:  Random sample of 1,000 reviews

Phase 2 support added:
  - Create a stratified subset for Machine Learning experiments
  - Create an ML-ready text column
  - Save outputs dedicated to Phase 2 modelling
=============================================================================
COLUMN SELECTION RATIONALE:
  - 'reviewText' : Main free-form review body with the richest sentiment signal.
  - 'summary'    : Short headline that often carries concentrated sentiment.
  - 'overall'    : Numeric star rating used only for ground-truth labeling.
  - 'asin', 'reviewerID' : Retained as identifiers for traceability.

  Columns excluded:
  - 'verified', 'vote', 'image', 'style' : metadata not directly required for
    text sentiment modelling in this project stage.
=============================================================================
"""

import re
import string
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from pathlib import Path

# Download required NLTK resources (run once)
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("wordnet",   quiet=True)

from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Paths ───────────────────────────────────────────────────────────────────
INPUT_PATH       = Path("data/processed/base_reviews.csv")   # from Person 1
OUTPUT_PROCESSED = Path("data/processed")
OUTPUT_FIGURES   = Path("results/figures")
RANDOM_SEED      = 42
PHASE1_SAMPLE_SIZE = 1000
PHASE2_SAMPLE_SIZE = 3000

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ============================================================================
# 1. LOAD & VALIDATE INPUT
# ============================================================================

def load_base_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df):,} rows from {filepath.name}")
    print(f"       Columns: {list(df.columns)}")
    return df


# ============================================================================
# 2. LABELING  (Deliverable 2a)
# ============================================================================

def label_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'overall' star ratings to three sentiment classes:
        4, 5  →  Positive
        3     →  Neutral
        1, 2  →  Negative
    """
    def _map(rating):
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    df = df.copy()
    df["sentiment_label"] = df["overall"].apply(_map)

    print("\n" + "="*60)
    print("SENTIMENT LABEL DISTRIBUTION")
    print("="*60)
    counts = df["sentiment_label"].value_counts()
    pcts   = (counts / len(df) * 100).round(2)
    for label in ["Positive", "Neutral", "Negative"]:
        print(f"  {label:<10}: {counts[label]:>7,}  ({pcts[label]}%)")

    # --- Plot ---
    colors = {"Positive": "#55A868", "Neutral": "#4C72B0", "Negative": "#C44E52"}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values,
           color=[colors[l] for l in counts.index], edgecolor="white")
    ax.set_title("Sentiment Label Distribution (Full Dataset)", fontweight="bold")
    ax.set_ylabel("Count")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "06_label_distribution.png")
    plt.show()
    print("[SAVED] 06_label_distribution.png")

    return df


# ============================================================================
# 3. COLUMN SELECTION & OUTLIER CHECK (Deliverable 2b, 2c)
# ============================================================================

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for analysis + labeling.
    See module-level docstring for full justification.
    """
    KEEP = ["asin", "reviewerID", "overall", "sentiment_label",
            "reviewText", "summary",
            "review_word_len", "review_char_len", "summary_char_len"]
    available = [c for c in KEEP if c in df.columns]
    df = df[available].copy()
    print(f"\n[INFO] Working columns: {available}")
    return df


def text_outlier_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deliverable 2c: Flag and report text outliers.
      - Empty/null reviewText
      - Extremely short (<= 3 words)
      - Extremely long (>= 500 words)  — kept but flagged
    """
    print("\n" + "="*60)
    print("TEXT OUTLIER CHECK")
    print("="*60)

    df = df.copy()
    df["is_null_text"]  = df["reviewText"].isna()
    df["is_short_text"] = df["review_word_len"] <= 3
    df["is_long_text"]  = df["review_word_len"] >= 500

    null_ct  = df["is_null_text"].sum()
    short_ct = df["is_short_text"].sum()
    long_ct  = df["is_long_text"].sum()

    print(f"  Null / missing reviewText  : {null_ct:,}")
    print(f"  Very short (≤ 3 words)     : {short_ct:,}")
    print(f"  Very long  (≥ 500 words)   : {long_ct:,}")

    # For modelling, we DROP nulls and very-short reviews (< 3 words)
    before = len(df)
    df = df[~df["is_null_text"] & ~df["is_short_text"]].copy()
    after = len(df)
    print(f"\n  Dropped {before - after:,} outlier rows. Remaining: {after:,}")

    return df


# ============================================================================
# 4. PRE-PROCESSING PIPELINES  (Deliverable 4)
# ============================================================================

# ---------------- 4A. VADER preprocessing ----------------
# VADER RATIONALE:
#   VADER is designed to work on raw social-media style text. It handles:
#   - UPPERCASE (GREAT!, TERRIBLE)
#   - punctuation sequences (!!!, ???)
#   - emoticons (:), :(  )
#   - common internet slang
#   Therefore, for VADER we apply MINIMAL pre-processing:
#     ✔ Strip HTML tags (noise, not signal)
#     ✔ Replace URLs with a token
#     ✔ Concatenate summary + reviewText (both carry sentiment)
#     ✘ Do NOT lowercase  (VADER uses case to detect emphasis)
#     ✘ Do NOT remove punctuation (!!!, ??? affect polarity score)
#     ✘ Do NOT remove stop words (negations like "not" are critical)
#     ✘ Do NOT stem/lemmatize (VADER uses its own lexicon)

def preprocess_for_vader(text: str, summary: str = "") -> str:
    """
    Minimal cleaning for VADER.
    Combine summary headline + review body for richer context.
    """
    if pd.isna(text):
        text = ""
    if pd.isna(summary):
        summary = ""

    combined = f"{summary}. {text}" if summary else text

    # Remove HTML tags
    combined = re.sub(r"<[^>]+>", " ", combined)
    # Replace URLs
    combined = re.sub(r"http\S+|www\.\S+", "[URL]", combined)
    # Collapse excessive whitespace
    combined = re.sub(r"\s+", " ", combined).strip()

    return combined


# ---------------- 4B. TextBlob preprocessing ----------------
# TextBlob RATIONALE:
#   TextBlob uses a pattern-based approach and performs best on clean,
#   grammatically normalised text. Unlike VADER it is less sensitive to
#   capitalisation and punctuation. Therefore we apply MORE aggressive cleaning:
#     ✔ Lowercase
#     ✔ Remove HTML, URLs, special characters
#     ✔ Remove punctuation (TextBlob does not benefit from !!!)
#     ✔ Tokenise and remove stop words (reduce noise)
#     ✔ Lemmatise (reduce vocabulary size, unify morphological variants)
#     ✘ Do NOT remove negations ("not", "never") — handled by negation-aware step

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "can't",
                  "won't", "don't", "doesn't", "isn't", "wasn't", "aren't",
                  "weren't", "wouldn't", "couldn't", "shouldn't", "n't"}

STOP_WORDS_NO_NEGATION = STOP_WORDS - NEGATION_WORDS   # keep negations


def preprocess_for_textblob(text: str, summary: str = "") -> str:
    """
    Full NLP pipeline for TextBlob.
    """
    if pd.isna(text):
        text = ""
    if pd.isna(summary):
        summary = ""

    combined = f"{summary} {text}"

    # 1. Lowercase
    combined = combined.lower()

    # 2. Remove HTML tags
    combined = re.sub(r"<[^>]+>", " ", combined)

    # 3. Remove URLs
    combined = re.sub(r"http\S+|www\.\S+", " ", combined)

    # 4. Remove non-alphabetic characters (keep spaces)
    combined = re.sub(r"[^a-z\s]", " ", combined)

    # 5. Tokenise
    tokens = word_tokenize(combined)

    # 6. Remove stop words (preserving negations)
    tokens = [t for t in tokens if t not in STOP_WORDS_NO_NEGATION]

    # 7. Lemmatise
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if len(t) > 1]

    return " ".join(tokens)


# ---------------- 4C. ML preprocessing ----------------
# ML RATIONALE:
#   For TF-IDF + Logistic Regression / Naive Bayes we want a normalized text
#   representation with reduced noise and a compact vocabulary:
#     ✔ Lowercase
#     ✔ Remove HTML tags and URLs
#     ✔ Remove punctuation / digits
#     ✔ Remove stop words while preserving negations
#     ✔ Lemmatise tokens
#   This output becomes the dedicated feature text for vectorisation.

def preprocess_for_ml(text: str, summary: str = "") -> str:
    """
    Cleaning pipeline for classic ML models using TF-IDF features.
    """
    if pd.isna(text):
        text = ""
    if pd.isna(summary):
        summary = ""

    combined = f"{summary} {text}".strip().lower()
    combined = re.sub(r"<[^>]+>", " ", combined)
    combined = re.sub(r"http\S+|www\.\S+", " ", combined)
    combined = re.sub(r"[^a-z\s]", " ", combined)

    tokens = word_tokenize(combined)
    tokens = [t for t in tokens if t not in STOP_WORDS_NO_NEGATION]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if len(t) > 1]

    return " ".join(tokens)


# ============================================================================
# 5. APPLY PIPELINES TO DATAFRAME
# ============================================================================

def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all pipelines, add feature columns, and drop ML-empty rows."""
    df = df.copy()

    print("\n[INFO] Applying VADER preprocessing ...")
    df["text_vader"] = df.apply(
        lambda r: preprocess_for_vader(r["reviewText"], r.get("summary", "")), axis=1
    )

    print("[INFO] Applying TextBlob preprocessing ...")
    df["text_textblob"] = df.apply(
        lambda r: preprocess_for_textblob(r["reviewText"], r.get("summary", "")), axis=1
    )

    print("[INFO] Applying ML preprocessing ...")
    df["text_ml"] = df.apply(
        lambda r: preprocess_for_ml(r["reviewText"], r.get("summary", "")), axis=1
    )

    # Length after preprocessing (sanity check)
    df["vader_word_len"]    = df["text_vader"].str.split().str.len()
    df["textblob_word_len"] = df["text_textblob"].str.split().str.len()
    df["ml_word_len"]       = df["text_ml"].str.split().str.len()

    before = len(df)
    df = df[df["ml_word_len"] > 0].copy()
    after = len(df)
    print(f"[INFO] Dropped {before - after:,} rows with empty ML text after cleaning.")

    print(f"\n  VADER    avg token length : {df['vader_word_len'].mean():.1f}")
    print(f"  TextBlob avg token length : {df['textblob_word_len'].mean():.1f}")
    print(f"  ML       avg token length : {df['ml_word_len'].mean():.1f}")

    return df


# ============================================================================
# 6. RANDOM SAMPLE OF 1,000 REVIEWS  (Deliverable 5)
# ============================================================================

def _stratified_sample(df: pd.DataFrame, sample_size: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create a stratified sample preserving the sentiment class proportions.
    """
    random.seed(seed)
    np.random.seed(seed)

    sample = (
        df.groupby("sentiment_label", group_keys=False)
        .apply(lambda x: x.sample(
            n=min(len(x), int(sample_size * len(x) / len(df))),
            random_state=seed
        ))
    )

    if len(sample) < sample_size:
        remaining = df.drop(sample.index).sample(sample_size - len(sample), random_state=seed)
        sample = pd.concat([sample, remaining])

    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sample


def sample_1000(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Deliverable 5: Stratified random sample of 1,000 reviews,
    proportional to the class distribution to avoid severe imbalance.
    """
    sample = _stratified_sample(df, sample_size=PHASE1_SAMPLE_SIZE, seed=seed)

    print("\n" + "="*60)
    print(f"STRATIFIED SAMPLE — {PHASE1_SAMPLE_SIZE:,} reviews  (seed={seed})")
    print("="*60)
    print(sample["sentiment_label"].value_counts())

    return sample


def sample_phase2_subset(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create the Phase 2 stratified subset for ML experiments.
    """
    sample = _stratified_sample(df, sample_size=PHASE2_SAMPLE_SIZE, seed=seed)

    print("\n" + "="*60)
    print(f"STRATIFIED SAMPLE — {PHASE2_SAMPLE_SIZE:,} reviews for Phase 2  (seed={seed})")
    print("="*60)
    print(sample["sentiment_label"].value_counts())

    return sample


# ============================================================================
# 7. SAVE OUTPUTS
# ============================================================================

def save_outputs(
    df_full: pd.DataFrame,
    df_sample_phase1: pd.DataFrame,
    df_sample_phase2: pd.DataFrame,
) -> None:
    full_path         = OUTPUT_PROCESSED / "preprocessed_full.csv"
    sample_phase1_path = OUTPUT_PROCESSED / "sample_1000.csv"
    sample_phase2_path = OUTPUT_PROCESSED / "phase2_subset_3000.csv"

    df_full.to_csv(full_path, index=False)
    df_sample_phase1.to_csv(sample_phase1_path, index=False)
    df_sample_phase2.to_csv(sample_phase2_path, index=False)

    print(f"\n[SAVED] Full preprocessed dataset → {full_path}  ({len(df_full):,} rows)")
    print(f"[SAVED] 1,000-review sample         → {sample_phase1_path}")
    print(f"[SAVED] Phase 2 subset (3,000)     → {sample_phase2_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # 1. Load Person 1 output
    df = load_base_data(INPUT_PATH)

    # 2. Label
    df = label_sentiment(df)

    # 3. Select relevant columns
    df = select_columns(df)

    # 4. Text outlier check
    df = text_outlier_check(df)

    # 5. Apply preprocessing pipelines
    df = apply_preprocessing(df)

    # 6. Draw the 1,000-review sample for Phase 1
    sample_phase1 = sample_1000(df)

    # 7. Draw the 3,000-review subset for Phase 2
    sample_phase2 = sample_phase2_subset(df)

    # 8. Save
    save_outputs(df, sample_phase1, sample_phase2)

    print("\n✅  Person 2 complete.")
    print("    Outputs:")
    print("      data/processed/preprocessed_full.csv")
    print("      data/processed/sample_1000.csv")
    print("      data/processed/phase2_subset_3000.csv")
