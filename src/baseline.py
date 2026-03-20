"""
baseline.py
-----------
TF-IDF + Logistic Regression baseline for fake news detection.
Serves as the comparison point for the RoBERTa primary model.

Usage:
    python src/baseline.py \
        --train_csv data/processed/train_processed.csv \
        --valid_csv data/processed/valid_processed.csv \
        --test_csv  data/processed/test_processed.csv
"""

import argparse             # For parsing command-line arguments
import logging              # For logging information
import json                 # For reading/writing JSON files
from pathlib import Path    # For handling file paths

import numpy as np          # For numerical operations
import pandas as pd         # For data manipulation
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF vectorization
from sklearn.linear_model import LogisticRegression          # For logistic regression model
from sklearn.metrics import (
    accuracy_score, 
    f1_score,
    classification_report,
    confusion_matrix
)   # For evaluation metrics
from sklearn.pipeline import Pipeline               # For creating a pipeline of transformations and model
from sklearn.preprocessing import StandardScaler    # For feature scaling
from scipy.sparse import hstack, csr_matrix         # For handling sparse matrices
import joblib                                       # For saving and loading models 

logging.basicConfig(                                     # Configure logging to display time, level, and message
    level = logging.INFO,                                # Set logging level to INFO to capture important information
    format = "%(asctime)s [%(levelname)s] %(message)s"   # Set the logging format to include timestamp, log level, and message
)
logger = logging.getLogger(__name__)             # Create a logger for this module

# Feature columns 
TEXT_COL = "statement_clean"  # Column containing the cleaned text statements
LABEL_COL = "label_binary"    # Column containing the binary labels (0 for fake, 1 for real)

# Engineered numeric features from liar_preprocessing.py
NUMERIC_FEATURES = [
    "statement_len",
    "statement_char_len",
    "exclamation_count",
    "question_count",
    "capitalized_word_ratio",
    "speaker_lie_rate",
    "speaker_total_statements",
    "is_political_context"
]  # List of engineered numeric features to be included in the model

# Categorical features to one-hot encode
CATEGORICAL_FEATURES = [
    "party",
    "speaker_job",
    "state_info"
]   

# Data loading
def load_split(path : str) -> pd.DataFrame:
    df = pd.read_csv(path)  # Load the CSV file into a DataFrame
    for col in NUMERIC_FEATURES:  # Ensure numeric features are treated as floats
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}. Run liar_preprocessing.py first.")
    return df

# Feature construction
class FeatureBuilder:
    """
    Builds a combined feature matrix from:
    1. TF-IDF on statement text
    2. Engineered numeric features
    3. One-hot encoded categorical features 
    """

    def __init__(
        self,
        max_features: int = 50_000, # Limit TF-IDF features to top 50k to manage memory
        ngram_range: tuple = (1, 2), # Use unigrams and bigrams for TF-IDF
        sublinear_tf: bool = True,     # Use sublinear TF scaling (1 + log(tf))
    ):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=2,
        )
        self.scaler = StandardScaler()
        self.cat_columns_ = None  # set after fit
 
    def fit_transform(self, df: pd.DataFrame):
        # 1. TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(df[TEXT_COL].fillna(""))
 
        # 2. Numeric features
        numeric_matrix = self.scaler.fit_transform(
            df[NUMERIC_FEATURES].fillna(0).values
        )
 
        # 3. Categorical one-hot
        cat_dummies = pd.get_dummies(
            df[CATEGORICAL_FEATURES].fillna("unknown"), drop_first=False
        )
        self.cat_columns_ = cat_dummies.columns.tolist()
        cat_matrix = csr_matrix(cat_dummies.values.astype(float))
 
        return hstack([tfidf_matrix, csr_matrix(numeric_matrix), cat_matrix])
 
    def transform(self, df: pd.DataFrame):
        tfidf_matrix = self.tfidf.transform(df[TEXT_COL].fillna(""))
 
        numeric_matrix = self.scaler.transform(
            df[NUMERIC_FEATURES].fillna(0).values
        )
 
        cat_dummies = pd.get_dummies(
            df[CATEGORICAL_FEATURES].fillna("unknown"), drop_first=False
        )
        # Align columns to training set (handle unseen categories)
        cat_dummies = cat_dummies.reindex(columns=self.cat_columns_, fill_value=0)
        cat_matrix = csr_matrix(cat_dummies.values.astype(float))
 
        return hstack([tfidf_matrix, csr_matrix(numeric_matrix), cat_matrix])
    
# Evaluation
def evaluate(y_true, y_pred, split_name: str = "test") -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    f1_binary = f1_score(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)
 
    logger.info("── %s results ──", split_name.upper())
    logger.info("Accuracy : %.4f", acc)
    logger.info("F1 Macro : %.4f", f1)
    logger.info("F1 Binary: %.4f", f1_binary)
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=["Fake", "Real"]))
 
    return {
        "split": split_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "f1_binary": round(f1_binary, 4),
        "confusion_matrix": cm.tolist(),
    }

# Error analysis
def error_analysis(df: pd.DataFrame, y_pred: np.ndarray, n: int = 10) -> pd.DataFrame:
    """
    Return a DataFrame of misclassified examples with key metadata.
    Useful for manual inspection and the error analysis report section.
    """
    errors = df.copy()
    errors["predicted"] = y_pred
    errors["true"] = df[LABEL_COL].values
    errors = errors[errors["predicted"] != errors["true"]]
 
    cols = [
        "statement", "true", "predicted",
        "speaker", "party", "speaker_lie_rate",
        "capitalized_word_ratio", "statement_len",
    ]
    available = [c for c in cols if c in errors.columns]
    return errors[available].head(n)

# Main training routine
def train_baseline(
    train_csv: str,
    valid_csv: str,
    test_csv: str,
    output_dir: str = "models/baseline",
    C: float = 1.0,
    max_iter: int = 1000,
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
) -> dict:
    """
    Full training + evaluation pipeline for the TF-IDF baseline.
 
    Returns:
        dict of evaluation metrics for all splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Load data
    logger.info("Loading data...")
    df_train = load_split(train_csv)
    df_valid = load_split(valid_csv)
    df_test  = load_split(test_csv)
 
    y_train = df_train[LABEL_COL].values
    y_valid = df_valid[LABEL_COL].values
    y_test  = df_test[LABEL_COL].values
 
    # Build features
    logger.info("Building features...")
    builder = FeatureBuilder(max_features=max_features, ngram_range=ngram_range)
    X_train = builder.fit_transform(df_train)
    X_valid = builder.transform(df_valid)
    X_test  = builder.transform(df_test)
 
    logger.info("Feature matrix shape — train: %s", X_train.shape)
 
    # Train
    logger.info("Training Logistic Regression (C=%.4f)...", C)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="saga",
        random_state=42,
    )
    clf.fit(X_train, y_train)
 
    # Evaluate
    results = {}
    for name, X, y, df in [
        ("train", X_train, y_train, df_train),
        ("valid", X_valid, y_valid, df_valid),
        ("test",  X_test,  y_test,  df_test),
    ]:
        y_pred = clf.predict(X)
        results[name] = evaluate(y, y_pred, split_name=name)
 
        if name == "test":
            errors = error_analysis(df, y_pred)
            error_path = output_dir / "error_analysis.csv"
            errors.to_csv(error_path, index=False)
            logger.info("Error analysis saved → %s", error_path)
 
    # Save model + artifacts
    joblib.dump(clf,     output_dir / "logistic_regression.joblib")
    joblib.dump(builder, output_dir / "feature_builder.joblib")
 
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
 
    logger.info("Model saved → %s", output_dir)
    return results

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline")
    parser.add_argument("--train_csv", default="data/processed/train_processed.csv")
    parser.add_argument("--valid_csv", default="data/processed/valid_processed.csv")
    parser.add_argument("--test_csv",  default="data/processed/test_processed.csv")
    parser.add_argument("--output_dir", default="models/baseline")
    parser.add_argument("--C",          type=float, default=1.0)
    parser.add_argument("--max_iter",   type=int,   default=3000)
    parser.add_argument("--max_features", type=int, default=50_000)
    args = parser.parse_args()
 
    train_baseline(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        C=args.C,
        max_iter=args.max_iter,
        max_features=args.max_features,
    )

# Pipelines:
# 1. Data loading → load_split()
# 2. Feature construction → FeatureBuilder
# 3. Model training → LogisticRegression
# 4. Evaluation → evaluate()
