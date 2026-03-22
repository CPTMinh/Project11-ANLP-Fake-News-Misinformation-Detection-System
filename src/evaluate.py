"""
evaluate.py
-----------
Shared evaluation utilities for the Fake News Detection project.
Used by both baseline_model.py and roberta_train.py.

Provides:
  - compute_metrics()         -> dict of classification metrics
  - print_classification_report()
  - plot_confusion_matrix()   -> saves figure
  - error_analysis()          -> DataFrame of misclassified examples
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Label display names for binary classification
LABEL_NAMES = ["Fake (0)", "Real (1)"]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "macro",
) -> dict[str, float]:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like of int
    y_pred : array-like of int
    average : str
        Averaging strategy for F1/precision/recall. Default 'macro'.

    Returns
    -------
    dict with keys: accuracy, f1, precision, recall
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall":   float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


def print_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model_name: str = "Model",
) -> None:
    """Print a formatted sklearn classification report."""
    print(f"\n{'=' * 60}")
    print(f"  Classification Report — {model_name}")
    print('=' * 60)
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"], zero_division=0))


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    model_name: str = "Model",
    label_names: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot and optionally save a normalised confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like of int
    model_name     : displayed in the figure title
    label_names    : x/y tick labels; defaults to LABEL_NAMES
    save_path      : if not None, saves the figure to this path
    show           : if True, calls plt.show()
    """
    if label_names is None:
        label_names = LABEL_NAMES

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in zip(
        axes,
        [cm, cm_norm],
        ["Count", "Row-Normalised"],
    ):
        fmt = "d" if title == "Count" else ".2f"
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"{model_name} — Confusion Matrix ({title})", fontsize=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    df: pd.DataFrame,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    text_col: str = "statement",
    n: int = 20,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of misclassified examples with diagnostic columns.

    Parameters
    ----------
    df        : original processed DataFrame (aligned with y_true / y_pred)
    y_true    : ground-truth labels
    y_pred    : model predictions
    text_col  : name of the raw text column to include
    n         : max number of error examples to return
    extra_cols: additional DataFrame columns to include

    Returns
    -------
    pd.DataFrame with columns:
        statement, true_label, predicted_label, label (6-class),
        statement_len, speaker_lie_rate, party, speaker, context
        (plus any extra_cols requested)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error_mask = y_true != y_pred
    error_df = df[error_mask].copy().reset_index(drop=True)
    error_df["true_label"] = y_true[error_mask]
    error_df["predicted_label"] = y_pred[error_mask]

    display_cols = [text_col, "true_label", "predicted_label", "label"]
    optional_cols = [
        "statement_len",
        "speaker_lie_rate",
        "party",
        "speaker",
        "context",
    ]
    if extra_cols:
        optional_cols += extra_cols

    available = [c for c in display_cols + optional_cols if c in error_df.columns]
    error_df = error_df[available].head(n)

    logger.info(
        "Error analysis: %d / %d examples misclassified (%.1f%%)",
        int(error_mask.sum()),
        len(y_true),
        100.0 * error_mask.mean(),
    )
    return error_df


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """
    Print a side-by-side model comparison table.

    Parameters
    ----------
    results : dict[model_name, metrics_dict]
        e.g. {'LR': {'accuracy': 0.62, 'f1': 0.61, ...}, 'RoBERTa': {...}}
    """
    df = pd.DataFrame(results).T
    df = df[["accuracy", "f1", "precision", "recall"]]
    df.columns = ["Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"]
    df = df.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    print("\n" + "=" * 70)
    print("  Model Comparison — Test Set")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")
