# Amazon Sentiment Analysis & Recommender System

## Overview
End-to-end NLP system that analyzes customer reviews, improves sentiment classification using machine learning, and builds a recommendation system enhanced with textual sentiment.

This project uses the Amazon Industrial & Scientific review domain and combines lexicon-based methods, supervised ML models, recommendation logic, and LLM-based review support tasks.

## Dataset
Before running the project, you must download the dataset manually.

Source:
Amazon Review Data (2018) by Jianmo Ni (UCSD)
[https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html)

Required category:
`Amazon Industrial & Scientific`

Place the downloaded file in:

```text
data/raw/Industrial_and_Scientific.json.gz
```

The pipeline expects that file path when running the data processing scripts.

## Features
- Lexicon-based sentiment analysis (VADER, TextBlob)
- Machine learning models (Logistic Regression, Naive Bayes)
- Class imbalance handling + Grid Search optimization
- Recommender system using adjusted ratings
- LLM integration (summarization + auto-reply)

## Results
- Logistic Regression (tuned) improved F1-score significantly
- Better handling of neutral and negative classes
- Enhanced product ranking using sentiment

## Tech Stack
- Python
- Scikit-learn
- Pandas / NumPy
- Transformers (HuggingFace)

## Project Structure
```text
amazon-industrial-sentiment-analysis/
├── data/
│   ├── raw/
│   │   └── Industrial_and_Scientific.json.gz
│   └── processed/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── data_exploration.py
├── preprocessing.py
├── model_evaluation.py
├── recommender.py
├── summarization.py
├── auto_reply.py
├── README.md
└── .gitignore
```

## Setup
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies.
4. Download the dataset and place it under `data/raw/`.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn nltk scikit-learn textblob vaderSentiment transformers
```

## How To Run
Run the project in this order:

1. Data processing / exploration
   ```bash
   python3 data_exploration.py
   ```
   This loads the raw Amazon reviews dataset, performs exploratory analysis, and creates the base processed file used by the next step.

2. Pre-processing
   ```bash
   python3 preprocessing.py
   ```
   This labels sentiment, applies the text cleaning pipelines, generates the Phase 1 sample of 1,000 reviews, and creates the Phase 2 stratified subset of 3,000 reviews.

3. Modeling / evaluation
   ```bash
   python3 model_evaluation.py
   ```
   This evaluates VADER and TextBlob, then trains and compares the ML models on the Phase 2 subset.

4. Recommender system
   ```bash
   python3 recommender.py
   ```
   This uses sentiment-enhanced predictions to adjust ratings and rank products.

5. LLM summarization
   ```bash
   python3 summarization.py
   ```
   This generates summaries from reviews or model outputs, depending on the script configuration.

6. LLM auto-reply
   ```bash
   python3 auto_reply.py
   ```
   This generates automated responses based on review content and sentiment context.

## Main Outputs
- `data/processed/base_reviews.csv`
- `data/processed/preprocessed_full.csv`
- `data/processed/sample_1000.csv`
- `data/processed/phase2_subset_3000.csv`
- `results/figures/`
- `results/metrics/`
- `results/models/`

## Notes
- The raw dataset is not included in the repository and must be downloaded separately.
- Generated datasets, metrics, figures, and trained models should remain local and should not be committed to Git.
