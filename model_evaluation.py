"""
=============================================================================
COMP 262 - Phase 1 / Phase 2 | Sherwayne 3: ML / Validation Lead
Task: Lexicon Sentiment Modeling + Machine Learning Evaluation
Dataset: Amazon Industrial & Scientific Reviews
=============================================================================
This script now supports:
  - Phase 1 lexicon evaluation (VADER + TextBlob)
  - Phase 2 machine learning evaluation (TF-IDF + Logistic Regression + Naive Bayes)
  - Final apples-to-apples comparison on the same Phase 2 test split
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── Paths ───────────────────────────────────────────────────────────────────
SAMPLE_PATH      = Path("data/processed/sample_1000.csv")
PHASE2_PATH      = Path("data/processed/phase2_subset_3000.csv")
OUTPUT_FIGURES   = Path("results/figures")
OUTPUT_METRICS   = Path("results/metrics")
OUTPUT_MODELS    = Path("results/models")

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["positive", "neutral", "negative"]


# ============================================================================
# 1. LOAD SAMPLE
# ============================================================================

def load_sample(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded Phase 1 sample: {len(df):,} rows")
    print(f"       Label distribution:\n{df['sentiment_label'].value_counts()}\n")
    return df


def load_phase2_subset(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded Phase 2 subset: {len(df):,} rows")
    print(f"       Label distribution:\n{df['sentiment_label'].value_counts()}\n")
    return df

def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize label text to lowercase and strip extra spaces."""
    return series.astype(str).str.strip().str.lower()


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
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    """Apply VADER to the 'text_vader' column (minimally preprocessed)."""
    print("[INFO] Running VADER predictions ...")
    df = df.copy()
    df["vader_compound"] = df["text_vader"].apply(
        lambda t: vader_analyzer.polarity_scores(str(t))["compound"]
    )
    df["vader_pred"] = df["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
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
        return "positive"
    elif p < -0.05:
        return "negative"
    else:
        return "neutral"


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
        lambda p: "positive" if p > 0.05 else ("negative" if p < -0.05 else "neutral")
    )
    return df


# ============================================================================
# 4. MACHINE LEARNING MODELS  (Phase 2)
# ============================================================================

def prepare_phase2_data(df: pd.DataFrame):
    """
    Build TF-IDF representation and stratified train/test split.
    Also preserve product/rating metadata so later stages (recommender)
    can aggregate predictions by product.
    """
    required = ["text_ml", "sentiment_label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Phase 2: {missing}")

    # Keep optional metadata if present
    metadata_cols = [c for c in ["asin", "overall", "reviewText"] if c in df.columns]

    X_text = df["text_ml"].fillna("")
    y = df["sentiment_label"]
    metadata = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)

    X_train_text, X_test_text, y_train, y_test, meta_train, meta_test = train_test_split(
        X_text,
        y,
        metadata,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("[INFO] TF-IDF vectorization complete")
    print(f"       Train shape: {X_train.shape}")
    print(f"       Test shape : {X_test.shape}")

    return X_train_text, X_test_text, X_train, X_test, y_train, y_test, vectorizer, meta_train, meta_test


def run_logistic_regression(X_train, X_test, y_train):
    print("[INFO] Training Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


# --- New function: Balanced Logistic Regression with GridSearchCV ---
def run_balanced_logistic_gridsearch(X_train, X_test, y_train):
    """
    Logistic Regression with class imbalance handling + GridSearchCV.
    We optimize using F1 macro because the dataset is imbalanced
    and all three sentiment classes matter.
    """
    print("[INFO] Running GridSearchCV for balanced Logistic Regression ...")

    param_grid = {
        "C": [0.1, 1.0, 5.0, 10.0],
        "solver": ["liblinear", "lbfgs"],
    }

    base_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    print(f"[INFO] Best GridSearch params: {grid.best_params_}")
    print(f"[INFO] Best CV F1-macro: {grid.best_score_:.4f}")

    return best_model, preds, grid.best_params_, grid.best_score_


def run_naive_bayes(X_train, X_test, y_train):
    print("[INFO] Training Multinomial Naive Bayes ...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, preds


def run_lexicon_on_testset(X_test_text: pd.Series) -> pd.DataFrame:
    """
    Run VADER and TextBlob on the SAME Phase 2 test set for apples-to-apples comparison.
    """
    test_df = pd.DataFrame({"text_for_lexicon": X_test_text}).copy()
    test_df["vader_compound"] = test_df["text_for_lexicon"].apply(
        lambda t: vader_analyzer.polarity_scores(str(t))["compound"]
    )
    test_df["vader_pred"] = test_df["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )
    test_df["tb_polarity"] = test_df["text_for_lexicon"].apply(
        lambda t: TextBlob(str(t)).sentiment.polarity
    )
    test_df["tb_pred"] = test_df["tb_polarity"].apply(
        lambda p: "positive" if p > 0.05 else ("negative" if p < -0.05 else "neutral")
    )
    return test_df


# ============================================================================
# 5. EVALUATION UTILITIES
# ============================================================================

def evaluate_model(y_true: pd.Series, y_pred: pd.Series,
                   model_name: str) -> dict:
    """Compute accuracy, precision, recall, F1 per class and macro-average."""
    labels = LABEL_ORDER
    y_true = pd.Series(y_true).astype(str).str.strip().str.lower()
    y_pred = pd.Series(y_pred).astype(str).str.strip().str.lower()

    print(f"[DEBUG] y_true labels: {sorted(y_true.unique().tolist())}")
    print(f"[DEBUG] y_pred labels: {sorted(y_pred.unique().tolist())}")

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
    y_true = pd.Series(y_true).astype(str).str.strip().str.lower()
    y_pred = pd.Series(y_pred).astype(str).str.strip().str.lower()
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
# 6. SCORE DISTRIBUTION PLOTS
# ============================================================================

def plot_score_distributions(df: pd.DataFrame) -> None:
    """
    Side-by-side KDE plots of VADER compound scores and TextBlob polarity
    scores, coloured by true sentiment label.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    palette = {"positive": "#55A868", "neutral": "#4C72B0", "negative": "#C44E52"}

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
# 7. COMPARISON TABLE  (Deliverable 7)
# ============================================================================

def build_comparison_table(*metric_dicts: dict) -> pd.DataFrame:
    """
    Build a side-by-side comparison table for any number of models.
    """
    comparison = pd.DataFrame(list(metric_dicts)).set_index("Model")

    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(comparison.to_string())

    numeric_cols = comparison.select_dtypes(include=["number"]).columns
    print("\nWinner per metric:")
    for col in numeric_cols:
        winner = comparison[col].idxmax()
        print(f"  {col:<25}: {winner}")

    out_path = OUTPUT_METRICS / "comparison_table_all_models.csv"
    comparison.to_csv(out_path)
    print(f"\n[SAVED] Comparison table → {out_path}")

    return comparison


def plot_comparison_bar(comparison: pd.DataFrame) -> None:
    """Bar chart comparing all models across the main metrics."""
    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"]
    ax = comparison[metrics].plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "10_model_comparison_all_models.png")
    plt.show()
    print("[SAVED] 10_model_comparison_all_models.png")


# ============================================================================
# 8. ERROR ANALYSIS (bonus — useful for report discussion)
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
    # ------------------------------------------------------------------
    # Phase 1: Lexicon evaluation on 1,000-review sample
    # ------------------------------------------------------------------
    df_phase1 = load_sample(SAMPLE_PATH)
    df_phase1["sentiment_label"] = normalize_labels(df_phase1["sentiment_label"])

    required_phase1 = ["text_vader", "text_textblob", "sentiment_label", "reviewText"]
    missing_phase1 = [c for c in required_phase1 if c not in df_phase1.columns]
    if missing_phase1:
        raise ValueError(f"Missing Phase 1 columns: {missing_phase1}")

    df_phase1 = run_vader(df_phase1)
    df_phase1 = run_textblob(df_phase1)

    vader_metrics_phase1 = evaluate_model(df_phase1["sentiment_label"], df_phase1["vader_pred"], "VADER (Phase 1)")
    tb_metrics_phase1    = evaluate_model(df_phase1["sentiment_label"], df_phase1["tb_pred"], "TextBlob (Phase 1)")

    plot_confusion_matrix(df_phase1["sentiment_label"], df_phase1["vader_pred"],
                          "VADER (Phase 1)", "07_confusion_vader.png")
    plot_confusion_matrix(df_phase1["sentiment_label"], df_phase1["tb_pred"],
                          "TextBlob (Phase 1)", "08_confusion_textblob.png")

    plot_score_distributions(df_phase1)
    error_analysis(df_phase1, "vader_pred", "VADER (Phase 1)")
    error_analysis(df_phase1, "tb_pred", "TextBlob (Phase 1)")

    out_phase1 = OUTPUT_METRICS / "predictions_sample_1000.csv"
    df_phase1.to_csv(out_phase1, index=False)
    print(f"\n[SAVED] Full Phase 1 predictions → {out_phase1}")

    # ------------------------------------------------------------------
    # Phase 2: ML evaluation + apples-to-apples comparison on same test set
    # ------------------------------------------------------------------
    df_phase2 = load_phase2_subset(PHASE2_PATH)
    df_phase2["sentiment_label"] = normalize_labels(df_phase2["sentiment_label"])
    X_train_text, X_test_text, X_train, X_test, y_train, y_test, vectorizer, meta_train, meta_test = prepare_phase2_data(df_phase2)

    # Baseline ML models (first run / before tuning)
    logreg_model, logreg_pred = run_logistic_regression(X_train, X_test, y_train)
    nb_model, nb_pred         = run_naive_bayes(X_train, X_test, y_train)

    logreg_metrics = evaluate_model(y_test, logreg_pred, "Logistic Regression (baseline)")
    nb_metrics     = evaluate_model(y_test, nb_pred, "Naive Bayes")

    plot_confusion_matrix(y_test, logreg_pred, "Logistic Regression (baseline)", "11_confusion_logreg_baseline.png")
    plot_confusion_matrix(y_test, nb_pred,     "Naive Bayes",                     "12_confusion_nb.png")

    # Tuned Logistic Regression (after class imbalance handling + grid search)
    tuned_logreg_model, tuned_logreg_pred, best_params, best_cv_f1 = run_balanced_logistic_gridsearch(
        X_train, X_test, y_train
    )
    tuned_logreg_metrics = evaluate_model(y_test, tuned_logreg_pred, "Logistic Regression (balanced + grid)")
    plot_confusion_matrix(
        y_test,
        tuned_logreg_pred,
        "Logistic Regression (balanced + grid)",
        "13_confusion_logreg_balanced_grid.png"
    )

    lexicon_test_df = run_lexicon_on_testset(X_test_text)
    vader_metrics_phase2 = evaluate_model(y_test, lexicon_test_df["vader_pred"], "VADER (Phase 2 test)")
    tb_metrics_phase2    = evaluate_model(y_test, lexicon_test_df["tb_pred"],    "TextBlob (Phase 2 test)")

    comparison = build_comparison_table(
        vader_metrics_phase2,
        tb_metrics_phase2,
        logreg_metrics,
        tuned_logreg_metrics,
        nb_metrics
    )
    plot_comparison_bar(comparison)

    phase2_predictions = pd.DataFrame({
        "text_ml": X_test_text.values,
        "true_label": y_test.values,
        "vader_pred": lexicon_test_df["vader_pred"].values,
        "textblob_pred": lexicon_test_df["tb_pred"].values,
        "logreg_baseline_pred": logreg_pred,
        "logreg_balanced_grid_pred": tuned_logreg_pred,
        "nb_pred": nb_pred,
    }, index=X_test_text.index)

    if "asin" in meta_test.columns:
        phase2_predictions["asin"] = meta_test["asin"]
    if "overall" in meta_test.columns:
        phase2_predictions["original_rating"] = meta_test["overall"]
    if "reviewText" in meta_test.columns:
        phase2_predictions["reviewText"] = meta_test["reviewText"]
    out_phase2 = OUTPUT_METRICS / "phase2_test_predictions_all_models.csv"
    phase2_predictions.to_csv(out_phase2, index=False)
    print(f"[INFO] Phase 2 prediction columns: {list(phase2_predictions.columns)}")
    print(f"[SAVED] Phase 2 test predictions → {out_phase2}")

    gridsearch_summary = pd.DataFrame([
        {
            "best_params": str(best_params),
            "best_cv_f1_macro": round(best_cv_f1, 4)
        }
    ])
    gridsearch_path = OUTPUT_METRICS / "gridsearch_best_params.csv"
    gridsearch_summary.to_csv(gridsearch_path, index=False)
    print(f"[SAVED] GridSearch summary → {gridsearch_path}")

    print("\n✅ Model evaluation complete.")
    print("   Key outputs:")
    print("     results/metrics/predictions_sample_1000.csv")
    print("     results/metrics/phase2_test_predictions_all_models.csv")
    print("     results/metrics/comparison_table_all_models.csv")
    print("     results/figures/07_confusion_vader.png")
    print("     results/figures/08_confusion_textblob.png")
    print("     results/metrics/gridsearch_best_params.csv")
    print("     results/figures/11_confusion_logreg_baseline.png")
    print("     results/figures/12_confusion_nb.png")
    print("     results/figures/13_confusion_logreg_balanced_grid.png")
    print("     results/figures/10_model_comparison_all_models.png")