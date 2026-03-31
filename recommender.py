"""
=============================================================================
COMP 262 - Phase 2 | Recommender System based on Reviews
Author: Luis Mateo
=============================================================================

Idea:
Enhance product ratings using sentiment predictions from the ML model.

Steps:
1. Load Phase 2 predictions
2. Adjust ratings based on sentiment
3. Aggregate per product (asin)
4. Recommend top products
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path("results/metrics/phase2_test_predictions_all_models.csv")
OUTPUT_PATH = Path("results/metrics/recommendations.csv")
FIGURES_PATH = Path("results/figures")
FIGURES_PATH.mkdir(parents=True, exist_ok=True)
def plot_rating_distributions(df: pd.DataFrame) -> None:
    """
    Plot the distribution of original vs adjusted ratings (before vs after).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original ratings
    axes[0].hist(df["original_rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor="black")
    axes[0].set_title("Original Rating Distribution", fontweight="bold")
    axes[0].set_xlabel("Original Rating")
    axes[0].set_ylabel("Number of Reviews")
    axes[0].set_xticks([1, 2, 3, 4, 5])

    # Adjusted ratings
    axes[1].hist(df["adjusted_rating"], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], edgecolor="black")
    axes[1].set_title("Adjusted Rating Distribution", fontweight="bold")
    axes[1].set_xlabel("Adjusted Rating")
    axes[1].set_ylabel("Number of Reviews")
    axes[1].set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()

    out_file = FIGURES_PATH / "14_before_after_rating_distribution.png"
    plt.savefig(out_file)
    plt.show()

    print(f"[SAVED] Before vs After rating distribution → {out_file}")


def adjust_rating(row):
    """
    Adjust rating using predicted sentiment.
    Simple heuristic:
      positive → +1
      neutral  →  0
      negative → -1
    """
    rating = row.get("original_rating", 3)  # fallback if not present
    sentiment = row["logreg_balanced_grid_pred"]

    if sentiment == "positive":
        rating += 1
    elif sentiment == "negative":
        rating -= 1

    return max(1, min(5, rating))  # clamp between 1–5


def build_recommender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build product ranking based on adjusted ratings.
    We keep additional statistics to make recommendations more reliable.
    """

    # If original rating not present, simulate from labels (fallback)
    if "original_rating" not in df.columns:
        label_map = {
            "positive": 5,
            "neutral": 3,
            "negative": 1
        }
        df["original_rating"] = df["true_label"].map(label_map)

    df["adjusted_rating"] = df.apply(adjust_rating, axis=1)

    # Aggregate by product (recommended path)
    if "asin" in df.columns:
        grouped = (
            df.groupby("asin")
            .agg(
                review_count=("asin", "count"),
                avg_original_rating=("original_rating", "mean"),
                avg_adjusted_rating=("adjusted_rating", "mean")
            )
            .reset_index()
        )

        grouped = grouped.sort_values(
            by=["avg_adjusted_rating", "review_count"],
            ascending=[False, False]
        )
    else:
        grouped = pd.DataFrame({
            "review_count": [len(df)],
            "avg_original_rating": [df["original_rating"].mean()],
            "avg_adjusted_rating": [df["adjusted_rating"].mean()]
        })
        grouped = grouped.sort_values(
            by=["avg_adjusted_rating", "review_count"],
            ascending=[False, False]
        )

    return grouped


def main():
    print("[INFO] Loading predictions...")
    df = pd.read_csv(INPUT_PATH)

    print(f"[INFO] Loaded {len(df):,} rows")

    print("[INFO] Building recommender...")
    recommendations = build_recommender(df)

    plot_rating_distributions(df)

    recommendations.to_csv(OUTPUT_PATH, index=False)

    print("\nTop 10 recommended items:")
    print(recommendations.head(10).to_string(index=False))

    print(f"\n[SAVED] Recommendations → {OUTPUT_PATH}")
    print("[SAVED] Figure → results/figures/14_before_after_rating_distribution.png")


if __name__ == "__main__":
    main()
