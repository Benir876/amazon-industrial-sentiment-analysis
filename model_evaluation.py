"""
=============================================================================
COMP 262 - Phase 1 | Sherwayne 3: ML / Validation Lead
Task: Lexicon Sentiment Modeling + Evaluation
Dataset: Amazon Industrial & Scientific Reviews (1,000-review sample)
=============================================================================
Deliverables covered:
  - Deliverable 3:  Lexicon package study & justification (VADER + TextBlob)
  - Deliverable 6:  Build two sentiment analysis models
  - Deliverable 7:  Validate results & comparison table
=============================================================================
LEXICON SELECTION RATIONALE (Deliverable 3):
  Selected: VADER and TextBlob
  Rejected: SentiWordNet

  VADER:
    • Specifically designed for social-media / short informal text — product
      reviews share many of the same characteristics (abbreviations, slang,
      emoticons, capitalization for emphasis).
    • Produces a compound score in [-1, +1] that maps directly to sentiment
      strength, making thresholding straightforward.
    • Requires NO training data; it is unsupervised and deterministic.
    • Handles negations, booster words, and punctuation natively.

  TextBlob:
    • Uses the Pattern library lexicon; returns polarity [-1, +1] and
      subjectivity [0, 1].  The subjectivity score is a useful secondary
      feature for filtering objective statements.
    • Works well on clean, normalised text — a good contrast to VADER's
      minimal-preprocessing approach, enabling a fair comparison of how
      text cleaning affects performance.
    • Lightweight and interpretable; aligns with the academic scope of Phase 1.

  Why NOT SentiWordNet:
    • Requires Part-of-Speech tagging and word-sense disambiguation for every
      token, making it considerably slower and more complex.
    • Its synset-level scoring introduces ambiguity for informal review text
      where words are used colloquially rather than in their primary dictionary
      sense.  Performance on product reviews has been shown to be inferior to
      both VADER and TextBlob in the literature.
=============================================================================
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Paths ───────────────────────────────────────────────────────────────────
SAMPLE_PATH    = Path("data/processed/sample_1000.csv")
OUTPUT_FIGURES = Path("results/figures")
OUTPUT_METRICS = Path("results/metrics")

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["Positive", "Neutral", "Negative"]


# ============================================================================
# 1. LOAD SAMPLE
# ============================================================================

def load_sample(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded sample: {len(df):,} rows")
    print(f"       Label distribution:\n{df['sentiment_label'].value_counts()}\n")
    return df


# ============================================================================
# 2. VADER MODEL  (Deliverable 6a)
# ============================================================================

vader_analyzer = SentimentIntensityAnalyzer()

def vader_predict(text: str) -> str:
    """
    VADER classification using compound score thresholds.
    Thresholds follow the original Hutto & Gilbert (2014) paper:
        compound >=  0.05  →  Positive
        compound <= -0.05  →  Negative
        otherwise          →  Neutral
    """
    scores = vader_analyzer.polarity_scores(str(text))
    c = scores["compound"]
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Apply VADER to the 'text_vader' column (minimally preprocessed)."""
    print("[INFO] Running VADER predictions ...")
    df = df.copy()
    df["vader_compound"] = df["text_vader"].apply(
        lambda t: vader_analyzer.polarity_scores(str(t))["compound"]
    )
    df["vader_pred"] = df["vader_compound"].apply(
        lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral")
    )
    return df


# ============================================================================
# 3. TEXTBLOB MODEL  (Deliverable 6b)
# ============================================================================

def textblob_predict(text: str) -> str:
    """
    TextBlob polarity classification.
    Thresholds:
        polarity >  0.05  →  Positive
        polarity < -0.05  →  Negative
        otherwise         →  Neutral
    (Using symmetric ±0.05 dead-band to assign Neutral,
     mirroring VADER for a fair apples-to-apples comparison.)
    """
    blob = TextBlob(str(text))
    p = blob.sentiment.polarity
    if p > 0.05:
        return "Positive"
    elif p < -0.05:
        return "Negative"
    else:
        return "Neutral"


def run_textblob(df: pd.DataFrame) -> pd.DataFrame:
    """Apply TextBlob to the 'text_textblob' column (fully preprocessed)."""
    print("[INFO] Running TextBlob predictions ...")
    df = df.copy()
    blob_results = df["text_textblob"].apply(
        lambda t: TextBlob(str(t)).sentiment
    )
    df["tb_polarity"]     = blob_results.apply(lambda s: s.polarity)
    df["tb_subjectivity"] = blob_results.apply(lambda s: s.subjectivity)
    df["tb_pred"] = df["tb_polarity"].apply(
        lambda p: "Positive" if p > 0.05 else ("Negative" if p < -0.05 else "Neutral")
    )
    return df


# ============================================================================
# 4. EVALUATION UTILITIES  (Deliverable 7)
# ============================================================================

def evaluate_model(y_true: pd.Series, y_pred: pd.Series,
                   model_name: str) -> dict:
    """Compute accuracy, precision, recall, F1 per class and macro-average."""
    labels = LABEL_ORDER

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="macro",
                           zero_division=0)
    rec  = recall_score(y_true, y_pred, labels=labels, average="macro",
                        zero_division=0)
    f1   = f1_score(y_true, y_pred, labels=labels, average="macro",
                    zero_division=0)

    print(f"\n{'='*60}")
    print(f"EVALUATION — {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy (macro)  : {acc:.4f}")
    print(f"  Precision (macro) : {prec:.4f}")
    print(f"  Recall (macro)    : {rec:.4f}")
    print(f"  F1-score (macro)  : {f1:.4f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_true, y_pred, labels=labels, zero_division=0)}")

    return {
        "Model"              : model_name,
        "Accuracy"           : round(acc,  4),
        "Precision (macro)"  : round(prec, 4),
        "Recall (macro)"     : round(rec,  4),
        "F1-score (macro)"   : round(f1,   4),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series,
                          model_name: str, filename: str) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / filename)
    plt.show()
    print(f"[SAVED] {filename}")


# ============================================================================
# 5. SCORE DISTRIBUTION PLOTS
# ============================================================================

def plot_score_distributions(df: pd.DataFrame) -> None:
    """
    Side-by-side KDE plots of VADER compound scores and TextBlob polarity
    scores, coloured by true sentiment label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    palette = {"Positive": "#55A868", "Neutral": "#4C72B0", "Negative": "#C44E52"}

    for label in LABEL_ORDER:
        subset = df[df["sentiment_label"] == label]
        axes[0].hist(subset["vader_compound"], bins=40, alpha=0.55,
                     label=label, color=palette[label], density=True)
        axes[1].hist(subset["tb_polarity"], bins=40, alpha=0.55,
                     label=label, color=palette[label], density=True)

    axes[0].axvline( 0.05, color="black", linestyle="--", linewidth=1, label="threshold ±0.05")
    axes[0].axvline(-0.05, color="black", linestyle="--", linewidth=1)
    axes[1].axvline( 0.05, color="black", linestyle="--", linewidth=1, label="threshold ±0.05")
    axes[1].axvline(-0.05, color="black", linestyle="--", linewidth=1)

    axes[0].set_title("VADER Compound Score Distribution", fontweight="bold")
    axes[0].set_xlabel("Compound Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].set_title("TextBlob Polarity Distribution", fontweight="bold")
    axes[1].set_xlabel("Polarity Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "09_score_distributions.png")
    plt.show()
    print("[SAVED] 09_score_distributions.png")


# ============================================================================
# 6. COMPARISON TABLE  (Deliverable 7)
# ============================================================================

def build_comparison_table(vader_metrics: dict, tb_metrics: dict) -> pd.DataFrame:
    """
    Deliverable 7: Produce the side-by-side comparison table.
    """
    comparison = pd.DataFrame([vader_metrics, tb_metrics]).set_index("Model")

    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison.to_string())

    # Highlight winner per metric
    print("\nWinner per metric:")
    for col in comparison.columns:
        winner = comparison[col].idxmax()
        print(f"  {col:<25}: {winner}")

    # Save as CSV
    out_path = OUTPUT_METRICS / "comparison_table.csv"
    comparison.to_csv(out_path)
    print(f"\n[SAVED] Comparison table → {out_path}")

    return comparison


def plot_comparison_bar(comparison: pd.DataFrame) -> None:
    """Grouped bar chart comparing the two models across all metrics."""
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, comparison.loc["VADER",    metrics], width,
                   label="VADER",    color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, comparison.loc["TextBlob", metrics], width,
                   label="TextBlob", color="#DD8452", edgecolor="white")

    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)

    ax.set_title("VADER vs TextBlob — Phase 1 Performance Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "10_model_comparison.png")
    plt.show()
    print("[SAVED] 10_model_comparison.png")


# ============================================================================
# 7. ERROR ANALYSIS (bonus — useful for report discussion)
# ============================================================================

def error_analysis(df: pd.DataFrame, model_pred_col: str,
                   model_name: str, n: int = 5) -> None:
    """
    Show sample misclassified reviews for each true class.
    Useful for qualitative analysis in the project report.
    """
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS — {model_name} (n={n} samples per class)")
    print(f"{'='*60}")
    errors = df[df["sentiment_label"] != df[model_pred_col]]
    for true_label in LABEL_ORDER:
        subset = errors[errors["sentiment_label"] == true_label]
        print(f"\n  True: {true_label} → predicted incorrectly ({len(subset)} total)")
        for _, row in subset.head(n).iterrows():
            preview = str(row["reviewText"])[:120].replace("\n", " ")
            print(f"    [{row[model_pred_col]}] {preview}...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # 1. Load the 1,000-review sample from Person 2
    df = load_sample(SAMPLE_PATH)

    # Verify required columns exist
    required = ["text_vader", "text_textblob", "sentiment_label", "reviewText"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns (run person2 first): {missing}")

    # 2. VADER model
    df = run_vader(df)

    # 3. TextBlob model
    df = run_textblob(df)

    # 4. Evaluate both
    vader_metrics = evaluate_model(df["sentiment_label"], df["vader_pred"], "VADER")
    tb_metrics    = evaluate_model(df["sentiment_label"], df["tb_pred"],    "TextBlob")

    # 5. Confusion matrices
    plot_confusion_matrix(df["sentiment_label"], df["vader_pred"],
                          "VADER",    "07_confusion_vader.png")
    plot_confusion_matrix(df["sentiment_label"], df["tb_pred"],
                          "TextBlob", "08_confusion_textblob.png")

    # 6. Score distributions
    plot_score_distributions(df)

    # 7. Comparison table + chart
    comparison = build_comparison_table(vader_metrics, tb_metrics)
    plot_comparison_bar(comparison)

    # 8. Error analysis (qualitative)
    error_analysis(df, "vader_pred",  "VADER")
    error_analysis(df, "tb_pred",     "TextBlob")

    # 9. Save full results
    out_path = Path("results/metrics/predictions_sample_1000.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] Full predictions → {out_path}")

    print("\n✅  Sherwayne complete.")
    print("    Key outputs:")
    print("      results/metrics/comparison_table.csv")
    print("      results/metrics/predictions_sample_1000.csv")
    print("      results/figures/07_confusion_vader.png")
    print("      results/figures/08_confusion_textblob.png")
    print("      results/figures/09_score_distributions.png")
    print("      results/figures/10_model_comparison.png")