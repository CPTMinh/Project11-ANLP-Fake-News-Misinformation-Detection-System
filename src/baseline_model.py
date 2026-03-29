"""
baseline_model.py
-----------------
Classical NLP baselines for fake news detection on the LIAR dataset.

Models:
  1. TF-IDF + Logistic Regression  (primary baseline)
  2. TF-IDF + LinearSVC            (secondary baseline)

Both use GridSearchCV for hyperparameter tuning and are saved
as pickled sklearn pipelines.

Usage (from project root):
    python src/baseline_model.py
    python src/baseline_model.py --config configs/baseline_config.yaml
"""

from __future__ import annotations      # For Python 3.10+ type hinting of class methods

import argparse                         # For command-line argument parsing
import logging                          # For logging progress and errors
import pickle                           # For saving and loading trained models
from pathlib import Path                # For convenient path handling  

import numpy as np                      # For numerical operations
import pandas as pd                     # For data manipulation and analysis
import yaml                             # For loading YAML configuration files
from sklearn.feature_extraction.text import TfidfVectorizer # For converting text to TF-IDF features
from sklearn.linear_model import LogisticRegression         # For Logistic Regression classifier
from sklearn.model_selection import GridSearchCV            # For hyperparameter tuning with cross-validation
from sklearn.pipeline import Pipeline                       # For creating a machine learning pipeline
from sklearn.svm import LinearSVC                           # For Linear Support Vector Classifier

from evaluate import (
    compute_metrics,
    error_analysis,
    plot_confusion_matrix,
    print_classification_report,
    print_comparison_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Config loading

def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Data loading

def load_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train / valid / test CSVs from config paths."""
    data_dir = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(data_dir / cfg["data"]["train_file"])
    valid = pd.read_csv(data_dir / cfg["data"]["valid_file"])
    test  = pd.read_csv(data_dir / cfg["data"]["test_file"])
    logger.info(
        "Loaded splits — train: %d, valid: %d, test: %d",
        len(train), len(valid), len(test),
    )
    return train, valid, test


def get_xy(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> tuple[pd.Series, pd.Series]:
    """Extract feature (X) and label (y) columns, dropping NaN rows."""
    df = df.dropna(subset=[text_col, label_col])
    return df[text_col], df[label_col].astype(int)

# Hyperparameter grid builder

def _build_param_grid(model_cfg: dict, prefix_tfidf: str, prefix_clf: str) -> list[dict]:
    """
    Build a list of param-grid dicts for GridSearchCV from flat YAML config.

    prefix_tfidf : e.g. 'tfidf__'
    prefix_clf   : e.g. 'clf__'

    Note: YAML loads sequences as Python lists, but sklearn expects tuples
    for parameters like ngram_range. Any list-of-lists value is converted
    to a list-of-tuples here.
    """
    def _coerce(v):
        """Convert a list value to a tuple (handles nested lists like ngram_range)."""
        if isinstance(v, list):
            return [tuple(item) if isinstance(item, list) else item for item in v]
        return v

    tfidf_params = {
        f"{prefix_tfidf}{k}": _coerce(v)
        for k, v in model_cfg["tfidf"].items()
    }
    clf_params = {
        f"{prefix_clf}{k}": _coerce(v)
        for k, v in model_cfg["classifier"].items()
    }
    return [{**tfidf_params, **clf_params}]

# Model 1: TF-IDF + Logistic Regression

class TfidfLRModel:
    """TF-IDF Vectorizer + Logistic Regression with GridSearchCV tuning."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["logistic_regression"]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf",   LogisticRegression()),
        ])
        param_grid = _build_param_grid(
            self.cfg,
            prefix_tfidf="tfidf__",
            prefix_clf="clf__",
        )
        self.grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=self.cfg.get("cv_folds", 5),
            scoring=self.cfg.get("scoring", "f1_macro"),
            n_jobs=self.cfg.get("n_jobs", -1),
            verbose=1,
        )
        self.best_estimator_: Pipeline | None = None

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> None:
        logger.info("Training TF-IDF + Logistic Regression (GridSearchCV)...")
        self.grid_search.fit(X_train, y_train)
        self.best_estimator_ = self.grid_search.best_estimator_
        logger.info("Best params: %s", self.grid_search.best_params_)
        logger.info("Best CV %s: %.4f", self.cfg.get("scoring"), self.grid_search.best_score_)

    def predict(self, X: pd.Series) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.best_estimator_.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.best_estimator_, f)
        logger.info("LR model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        with open(path, "rb") as f:
            return pickle.load(f)

# Model 2: TF-IDF + LinearSVC

class TfidfSVMModel:
    """TF-IDF Vectorizer + LinearSVC with GridSearchCV tuning."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["svm"]
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf",   LinearSVC()),
        ])
        param_grid = _build_param_grid(
            self.cfg,
            prefix_tfidf="tfidf__",
            prefix_clf="clf__",
        )
        self.grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=self.cfg.get("cv_folds", 5),
            scoring=self.cfg.get("scoring", "f1_macro"),
            n_jobs=self.cfg.get("n_jobs", -1),
            verbose=1,
        )
        self.best_estimator_: Pipeline | None = None

    def fit(self, X_train: pd.Series, y_train: pd.Series) -> None:
        logger.info("Training TF-IDF + LinearSVC (GridSearchCV)...")
        self.grid_search.fit(X_train, y_train)
        self.best_estimator_ = self.grid_search.best_estimator_
        logger.info("Best params: %s", self.grid_search.best_params_)
        logger.info("Best CV %s: %.4f", self.cfg.get("scoring"), self.grid_search.best_score_)

    def predict(self, X: pd.Series) -> np.ndarray:
        if self.best_estimator_ is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        return self.best_estimator_.predict(X)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.best_estimator_, f)
        logger.info("SVM model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> Pipeline:
        with open(path, "rb") as f:
            return pickle.load(f)

# End-to-end experiment runner

def run_baseline_experiment(config_path: str | Path = "configs/baseline_config.yaml") -> dict:
    """
    Full baseline experiment:
      1. Load data
      2. Train LR + SVM with GridSearchCV
      3. Evaluate on test set
      4. Save models & confusion matrices
      5. Print comparison table + error analysis

    Returns
    -------
    dict with keys 'lr' and 'svm', each containing metrics dicts.
    """
    cfg = load_config(config_path)

    # Data
    train_df, valid_df, test_df = load_data(cfg)
    text_col  = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    # Combine train + valid for final baseline training (common practice)
    trainval_df = pd.concat([train_df, valid_df], ignore_index=True)

    X_trainval, y_trainval = get_xy(trainval_df, text_col, label_col)
    X_test, y_test         = get_xy(test_df,     text_col, label_col)

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir  = Path(cfg["output"]["models_dir"])

    all_results: dict[str, dict] = {}

    # Logistic Regression
    lr_model = TfidfLRModel(cfg)
    lr_model.fit(X_trainval, y_trainval)
    y_pred_lr = lr_model.predict(X_test)

    print_classification_report(y_test, y_pred_lr, model_name="TF-IDF + Logistic Regression")
    plot_confusion_matrix(
        y_test, y_pred_lr,
        model_name="TF-IDF + LR",
        save_path=results_dir / "confusion_matrix_lr.png",
        show=False,
    )
    lr_model.save(models_dir / "baseline_lr.pkl")
    all_results["TF-IDF + LR"] = compute_metrics(y_test, y_pred_lr)

    # Error analysis for LR
    test_df_clean = test_df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    lr_errors = error_analysis(
        test_df_clean, y_test.values, y_pred_lr, text_col="statement"
    )
    lr_errors.to_csv(results_dir / "error_analysis_lr.csv", index=False)
    logger.info("LR error analysis saved to %s", results_dir / "error_analysis_lr.csv")

    # LinearSVC
    svm_model = TfidfSVMModel(cfg)
    svm_model.fit(X_trainval, y_trainval)
    y_pred_svm = svm_model.predict(X_test)

    print_classification_report(y_test, y_pred_svm, model_name="TF-IDF + LinearSVC")
    plot_confusion_matrix(
        y_test, y_pred_svm,
        model_name="TF-IDF + SVM",
        save_path=results_dir / "confusion_matrix_svm.png",
        show=False,
    )
    svm_model.save(models_dir / "baseline_svm.pkl")
    all_results["TF-IDF + SVM"] = compute_metrics(y_test, y_pred_svm)

    svm_errors = error_analysis(
        test_df_clean, y_test.values, y_pred_svm, text_col="statement"
    )
    svm_errors.to_csv(results_dir / "error_analysis_svm.csv", index=False)
    logger.info("SVM error analysis saved to %s", results_dir / "error_analysis_svm.csv")

    # Summary
    print_comparison_table(all_results)

    return all_results

# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.yaml",
        help="Path to baseline YAML config",
    )
    args = parser.parse_args()
    run_baseline_experiment(config_path=args.config)