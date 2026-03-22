"""
tests/test_baseline_model.py
-----------------------------
Unit tests for the baseline models and evaluate utilities.

Run with:
    pytest tests/test_baseline_model.py -v
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Allow import from src/
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseline_model import TfidfLRModel, TfidfSVMModel, get_xy
from evaluate import compute_metrics, error_analysis, print_comparison_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CFG = {
    "logistic_regression": {
        "tfidf": {
            "ngram_range": [[1, 1]],
            "max_features": [1000],
            "sublinear_tf": [True],
        },
        "classifier": {
            "C": [1.0],
            "max_iter": [200],
            "solver": ["lbfgs"],
        },
        "cv_folds": 2,
        "scoring": "f1_macro",
        "n_jobs": 1,
    },
    "svm": {
        "tfidf": {
            "ngram_range": [[1, 1]],
            "max_features": [1000],
            "sublinear_tf": [True],
        },
        "classifier": {
            "C": [1.0],
            "max_iter": [500],
        },
        "cv_folds": 2,
        "scoring": "f1_macro",
        "n_jobs": 1,
    },
}


def make_fake_data(n: int = 40):
    """Create a tiny labelled text dataset."""
    np.random.seed(42)
    real_texts = [
        "the government signed a new economic policy",
        "scientists discover a new species of bird",
        "congress passed the infrastructure bill",
        "stock markets rose on positive earnings",
        "the president met with foreign leaders",
    ]
    fake_texts = [
        "SHOCKING: vaccines proven to cause mind control",
        "BREAKING: aliens landed in washington dc CONFIRMED",
        "the deep state is controlling your water supply",
        "5G towers are killing birds everywhere PROOF",
        "secret society controls every election EXPOSED",
    ]
    texts, labels = [], []
    for i in range(n // 2):
        texts.append(real_texts[i % len(real_texts)])
        labels.append(1)
        texts.append(fake_texts[i % len(fake_texts)])
        labels.append(0)
    return pd.Series(texts), pd.Series(labels)


# ---------------------------------------------------------------------------
# TfidfLRModel tests
# ---------------------------------------------------------------------------

class TestTfidfLRModel:
    def test_fit_predict_shape(self):
        X, y = make_fake_data()
        model = TfidfLRModel(MINIMAL_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_binary(self):
        X, y = make_fake_data()
        model = TfidfLRModel(MINIMAL_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_best_estimator_set_after_fit(self):
        X, y = make_fake_data()
        model = TfidfLRModel(MINIMAL_CFG)
        assert model.best_estimator_ is None
        model.fit(X, y)
        assert model.best_estimator_ is not None

    def test_predict_before_fit_raises(self):
        X, _ = make_fake_data()
        model = TfidfLRModel(MINIMAL_CFG)
        with pytest.raises(RuntimeError, match="not been trained"):
            model.predict(X)

    def test_save_creates_file(self, tmp_path):
        X, y = make_fake_data()
        model = TfidfLRModel(MINIMAL_CFG)
        model.fit(X, y)
        out = tmp_path / "model.pkl"
        model.save(out)
        assert out.exists()

    def test_load_returns_pipeline(self, tmp_path):
        from sklearn.pipeline import Pipeline
        X, y = make_fake_data()
        model = TfidfLRModel(MINIMAL_CFG)
        model.fit(X, y)
        out = tmp_path / "model.pkl"
        model.save(out)
        loaded = TfidfLRModel.load(out)
        assert isinstance(loaded, Pipeline)


# ---------------------------------------------------------------------------
# TfidfSVMModel tests
# ---------------------------------------------------------------------------

class TestTfidfSVMModel:
    def test_fit_predict_binary(self):
        X, y = make_fake_data()
        model = TfidfSVMModel(MINIMAL_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_shape(self):
        X, y = make_fake_data()
        model = TfidfSVMModel(MINIMAL_CFG)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = [0, 0, 1, 1, 0, 1]
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1"]       == pytest.approx(1.0)

    def test_metric_keys_present(self):
        y_true = [0, 1, 0, 1]
        y_pred = [1, 0, 0, 1]
        metrics = compute_metrics(y_true, y_pred)
        for key in ("accuracy", "f1", "precision", "recall"):
            assert key in metrics

    def test_values_in_range(self):
        y_true = [0, 1, 0, 1, 0, 0]
        y_pred = [0, 1, 1, 0, 0, 1]
        metrics = compute_metrics(y_true, y_pred)
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# get_xy tests
# ---------------------------------------------------------------------------

class TestGetXY:
    def test_drops_nan(self):
        df = pd.DataFrame({
            "text": ["hello", None, "world"],
            "label": [0, 1, 0],
        })
        X, y = get_xy(df, "text", "label")
        assert len(X) == 2
        assert len(y) == 2

    def test_label_dtype_int(self):
        df = pd.DataFrame({"text": ["a", "b"], "label": ["0", "1"]})
        _, y = get_xy(df, "text", "label")
        assert y.dtype == int


# ---------------------------------------------------------------------------
# error_analysis tests
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    def setup_method(self):
        self.df = pd.DataFrame({
            "statement":     ["s0", "s1", "s2", "s3"],
            "label":         ["false", "true", "pants-fire", "mostly-true"],
            "statement_len": [4, 3, 5, 2],
            "party":         ["democrat", "republican", "democrat", "independent"],
        })
        self.y_true = np.array([0, 1, 0, 1])
        self.y_pred = np.array([1, 1, 0, 0])  # errors at idx 0 and 3

    def test_returns_dataframe(self):
        result = error_analysis(self.df, self.y_true, self.y_pred, text_col="statement")
        assert isinstance(result, pd.DataFrame)

    def test_contains_only_errors(self):
        result = error_analysis(self.df, self.y_true, self.y_pred, text_col="statement")
        # idx 0 pred=1 true=0, idx 3 pred=0 true=1
        assert len(result) == 2

    def test_has_required_columns(self):
        result = error_analysis(self.df, self.y_true, self.y_pred, text_col="statement")
        assert "true_label"      in result.columns
        assert "predicted_label" in result.columns


# ---------------------------------------------------------------------------
# print_comparison_table tests
# ---------------------------------------------------------------------------

class TestPrintComparisonTable:
    def test_does_not_raise(self, capsys):
        results = {
            "LR":  {"accuracy": 0.62, "f1": 0.61, "precision": 0.60, "recall": 0.62},
            "SVM": {"accuracy": 0.65, "f1": 0.64, "precision": 0.63, "recall": 0.65},
        }
        print_comparison_table(results)  # should not raise
        captured = capsys.readouterr()
        assert "LR" in captured.out
        assert "SVM" in captured.out
