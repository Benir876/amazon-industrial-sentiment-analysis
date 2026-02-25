"""
=============================================================================
COMP 262 - Phase 1 | PERSON 1: Data Engineer
Task: Data Loading & Exploratory Data Analysis (EDA)
Dataset: Amazon Industrial & Scientific Reviews
=============================================================================
Deliverables covered:
  - Deliverable 1: Dataset data exploration (counts, averages, distributions,
    review lengths, outliers, duplicates)
=============================================================================
"""

import os
import json
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Plotting style ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Paths ───────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = Path("Industrial_and_Scientific_5.json/Industrial_and_Scientific_5.json")   # 5-core
OUTPUT_FIGURES  = Path("results/figures")
OUTPUT_PROCESSED= Path("data/processed")

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_PROCESSED.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_amazon_reviews(filepath: Path) -> pd.DataFrame:
    """
    Load Amazon review JSON-gz file into a DataFrame.
    Supports both .json.gz and plain .json files.
    """
    records = []
    opener = gzip.open if str(filepath).endswith(".gz") else open
    with opener(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df):,} reviews from {filepath.name}")
    return df


# ============================================================================
# 2. INITIAL DATA AUDIT
# ============================================================================

def initial_audit(df: pd.DataFrame) -> None:
    """Print shape, dtypes, null counts, and sample rows."""
    print("\n" + "="*60)
    print("INITIAL DATA AUDIT")
    print("="*60)
    print(f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns      : {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nNull Counts:\n{df.isnull().sum()}")
    print(f"\nNull Percentage:\n{(df.isnull().sum() / len(df) * 100).round(2)}")
    print(f"\nSample rows (3):")
    print(df.head(3).to_string())


# ============================================================================
# 3. BASIC STATISTICS
# ============================================================================

def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deliverable 1a: Counts and averages.
    Returns a summary DataFrame.
    """
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)

    stats = {
        "Total Reviews"           : len(df),
        "Unique Products (asin)"  : df["asin"].nunique(),
        "Unique Reviewers"        : df["reviewerID"].nunique(),
        "Rating Mean"             : df["overall"].mean(),
        "Rating Median"           : df["overall"].median(),
        "Rating Std Dev"          : df["overall"].std(),
        "Avg Review Text Length"  : df["reviewText"].dropna().str.len().mean(),
        "Avg Summary Length"      : df["summary"].dropna().str.len().mean(),
        "Date Range (unixTime)"   : f"{df['unixReviewTime'].min()} – {df['unixReviewTime'].max()}"
            if "unixReviewTime" in df.columns else "N/A",
    }

    for k, v in stats.items():
        val = f"{v:,.2f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else v)
        print(f"  {k:<35}: {val}")

    # Rating value counts
    print(f"\nRating Distribution (counts):\n{df['overall'].value_counts().sort_index()}")

    return pd.DataFrame.from_dict(stats, orient="index", columns=["Value"])


# ============================================================================
# 4. DISTRIBUTION ANALYSIS
# ============================================================================

def plot_rating_distribution(df: pd.DataFrame) -> None:
    """Deliverable 1a/1b: Bar chart of star ratings."""
    counts = df["overall"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(int), counts.values,
                  color=sns.color_palette("muted", 5), edgecolor="white")
    ax.bar_label(bars, fmt="{:,.0f}", padding=3, fontsize=9)
    ax.set_title("Distribution of Star Ratings", fontsize=13, fontweight="bold")
    ax.set_xlabel("Star Rating")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "01_rating_distribution.png")
    plt.show()
    print("[SAVED] 01_rating_distribution.png")


def plot_reviews_per_product(df: pd.DataFrame) -> None:
    """Deliverable 1b/1c: Distribution of review counts per product."""
    reviews_per_product = df.groupby("asin").size()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histogram
    axes[0].hist(reviews_per_product, bins=50, color="#4C72B0", edgecolor="white", log=True)
    axes[0].set_title("Reviews per Product (log scale)", fontweight="bold")
    axes[0].set_xlabel("Number of Reviews")
    axes[0].set_ylabel("Number of Products (log)")

    # Top 20 most-reviewed products
    top20 = reviews_per_product.nlargest(20).sort_values()
    axes[1].barh(top20.index, top20.values, color="#DD8452")
    axes[1].set_title("Top 20 Most-Reviewed Products", fontweight="bold")
    axes[1].set_xlabel("Review Count")
    axes[1].set_ylabel("Product ASIN")
    axes[1].tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "02_reviews_per_product.png")
    plt.show()
    print("[SAVED] 02_reviews_per_product.png")


def plot_reviews_per_user(df: pd.DataFrame) -> None:
    """Deliverable 1d: Distribution of reviews per user."""
    reviews_per_user = df.groupby("reviewerID").size()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(reviews_per_user, bins=50, color="#55A868", edgecolor="white", log=True)
    ax.set_title("Distribution of Reviews per User (log scale)", fontweight="bold")
    ax.set_xlabel("Number of Reviews by a Single User")
    ax.set_ylabel("Number of Users (log)")

    # Annotate key percentiles
    for p in [50, 90, 99]:
        val = np.percentile(reviews_per_user, p)
        ax.axvline(val, linestyle="--", linewidth=1, color="red", alpha=0.7)
        ax.text(val + 0.2, ax.get_ylim()[1] * 0.6, f"P{p}={val:.0f}", fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "03_reviews_per_user.png")
    plt.show()
    print("[SAVED] 03_reviews_per_user.png")


# ============================================================================
# 5. REVIEW LENGTH ANALYSIS
# ============================================================================

def analyze_review_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deliverable 1e/1f: Review lengths, outlier detection, and length stats.
    Returns df with new length columns added.
    """
    df = df.copy()
    df["review_char_len"]  = df["reviewText"].fillna("").str.len()
    df["review_word_len"]  = df["reviewText"].fillna("").str.split().str.len()
    df["summary_char_len"] = df["summary"].fillna("").str.len()

    print("\n" + "="*60)
    print("REVIEW LENGTH STATISTICS")
    print("="*60)
    for col in ["review_char_len", "review_word_len", "summary_char_len"]:
        print(f"\n{col}:")
        print(df[col].describe().apply(lambda x: f"{x:,.2f}"))

    # --- Box plot for word lengths ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].boxplot(df["review_word_len"], vert=False, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    axes[0].set_title("Word Count Distribution (reviewText)", fontweight="bold")
    axes[0].set_xlabel("Word Count")

    # Log-scale histogram
    axes[1].hist(df["review_word_len"], bins=100, color="#C44E52", edgecolor="white", log=True)
    axes[1].set_title("Word Count Histogram (log scale)", fontweight="bold")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency (log)")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "04_review_lengths.png")
    plt.show()
    print("[SAVED] 04_review_lengths.png")

    return df


def detect_outliers(df: pd.DataFrame) -> None:
    """
    Deliverable 1e: Flag extreme length outliers using IQR method.
    """
    print("\n" + "="*60)
    print("OUTLIER DETECTION (IQR Method — Word Count)")
    print("="*60)
    wl = df["review_word_len"]
    Q1, Q3 = wl.quantile(0.25), wl.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    outliers = df[(wl < lower) | (wl > upper)]
    print(f"  IQR bounds   : [{lower:.1f}, {upper:.1f}]")
    print(f"  Total outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")
    print(f"  Empty reviews (0 words): {(wl == 0).sum():,}")
    print(f"  Very long (>500 words) : {(wl > 500).sum():,}")


# ============================================================================
# 6. DUPLICATE DETECTION
# ============================================================================

def check_duplicates(df: pd.DataFrame) -> None:
    """Deliverable 1g: Check for duplicate rows and duplicate review texts."""
    print("\n" + "="*60)
    print("DUPLICATE DETECTION")
    print("="*60)

    # Convert any dict/list columns to strings so pandas can hash them
    df_hashable = df.copy()
    for col in df_hashable.columns:
        if df_hashable[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df_hashable[col] = df_hashable[col].astype(str)

    exact_dupes = df_hashable.duplicated().sum()
    print(f"  Exact duplicate rows         : {exact_dupes:,}")

    # Duplicate on (reviewerID, asin) = same user reviewed same product twice
    user_product_dupes = df.duplicated(subset=["reviewerID", "asin"]).sum()
    print(f"  Duplicate (user, product)    : {user_product_dupes:,}")

    # Duplicate review text
    text_dupes = df.duplicated(subset=["reviewText"]).sum()
    print(f"  Duplicate reviewText entries : {text_dupes:,}")

    # Show a sample of text duplicates if they exist
    if text_dupes > 0:
        sample = (df[df.duplicated(subset=["reviewText"], keep=False)]
                  .sort_values("reviewText")
                  .head(4)[["reviewerID", "asin", "overall", "reviewText"]])
        print(f"\n  Sample duplicated texts:\n{sample.to_string()}")


# ============================================================================
# 7. TEMPORAL ANALYSIS (bonus)
# ============================================================================

def plot_reviews_over_time(df: pd.DataFrame) -> None:
    """Bonus: Show review volume over time."""
    if "reviewTime" not in df.columns:
        print("[SKIP] No 'reviewTime' column found.")
        return

    df = df.copy()
    df["date"] = pd.to_datetime(df["reviewTime"], format="%m %d, %Y", errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("year_month").size()

    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, color="#4C72B0", linewidth=1.5)
    ax.set_title("Review Volume Over Time (Monthly)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "05_reviews_over_time.png")
    plt.show()
    print("[SAVED] 05_reviews_over_time.png")


# ============================================================================
# 8. SAVE CLEANED BASE DATAFRAME
# ============================================================================

def save_base_dataframe(df: pd.DataFrame) -> None:
    """
    Save the enriched (length columns added) DataFrame for Person 2 to consume.
    """
    out_path = OUTPUT_PROCESSED / "base_reviews.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] Base DataFrame → {out_path}  ({len(df):,} rows)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # --- Load ---
    df = load_amazon_reviews(RAW_DATA_PATH)

    # --- Audit ---
    initial_audit(df)

    # --- Stats ---
    stats_df = basic_statistics(df)

    # --- Visualizations ---
    plot_rating_distribution(df)
    plot_reviews_per_product(df)
    plot_reviews_per_user(df)

    # --- Length analysis ---
    df = analyze_review_lengths(df)
    detect_outliers(df)

    # --- Duplicates ---
    check_duplicates(df)

    # --- Temporal ---
    plot_reviews_over_time(df)

    # --- Save for Person 2 ---
    save_base_dataframe(df)

    print("\n✅  Person 1 complete. Output saved to data/processed/base_reviews.csv")