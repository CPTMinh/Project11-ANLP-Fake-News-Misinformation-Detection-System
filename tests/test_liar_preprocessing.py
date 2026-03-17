"""
tests/test_liar_preprocessing.py
---------------------------------
Unit tests for the LIAR dataset preprocessing pipeline.

Run with:
    pytest tests/test_liar_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.liar_preprocessing import (
    BINARY_LABEL_MAP,
    CREDIT_COLUMNS,
    LIAR_COLUMNS,
    ORDINAL_LABEL_MAP,
    _capitalized_ratio,
    clean_categorical,
    clean_text,
    engineer_features,
    handle_missing,
    map_labels,
)


# Fixtures

def make_sample_df(n: int = 6) -> pd.DataFrame:
    """Create a minimal valid LIAR-like DataFrame with one row per label."""
    labels = list(BINARY_LABEL_MAP.keys())  # all 6 labels
    rows = []
    for i, label in enumerate(labels):
        row = {
            "id": f"stmt{i}",
            "label": label,
            "statement": f"This is statement number {i} about SOME TOPIC.",
            "subject": "politics",
            "speaker": f"speaker_{i}",
            "speaker_job": "politician",
            "state_info": "california",
            "party": "democrat",
            "barely_true_counts": str(i),
            "false_counts": str(i + 1),
            "half_true_counts": str(i + 2),
            "mostly_true_counts": str(i + 3),
            "pants_on_fire_counts": str(i),
            "context": "in a press conference",
        }
        rows.append(row)
    return pd.DataFrame(rows)


# clean_text
class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_removes_non_ascii(self):
        result = clean_text("café naïve")
        assert "é" not in result
        assert "ï" not in result

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_empty_string(self):
        assert clean_text("") == ""


# clean_categorical

class TestCleanCategorical:
    def test_lowercases(self):
        s = pd.Series(["Democrat", "REPUBLICAN"])
        result = clean_categorical(s)
        assert list(result) == ["democrat", "republican"]

    def test_fills_nan(self):
        s = pd.Series([None, "democrat"])
        result = clean_categorical(s, fill="unknown")
        assert result.iloc[0] == "unknown"

    def test_fills_empty_string(self):
        s = pd.Series(["", "republican"])
        result = clean_categorical(s)
        assert result.iloc[0] == "unknown"


# map_labels
class TestMapLabels:
    def test_all_labels_mapped(self):
        df = make_sample_df()
        df = handle_missing(df)
        result = map_labels(df)
        assert set(result["label_binary"].unique()).issubset({0, 1})
        assert set(result["label_ordinal"].unique()).issubset(set(range(6)))

    def test_fake_labels_map_to_0(self):
        df = pd.DataFrame([{"label": lbl, "statement": "test"} | {c: "0" for c in CREDIT_COLUMNS}
                           for lbl in ["pants-fire", "false", "barely-true"]])
        df = handle_missing(df)
        result = map_labels(df)
        assert all(result["label_binary"] == 0)

    def test_real_labels_map_to_1(self):
        df = pd.DataFrame([{"label": lbl, "statement": "test"} | {c: "0" for c in CREDIT_COLUMNS}
                           for lbl in ["half-true", "mostly-true", "true"]])
        df = handle_missing(df)
        result = map_labels(df)
        assert all(result["label_binary"] == 1)

    def test_unknown_label_dropped(self):
        df = make_sample_df()
        df = handle_missing(df)
        # Inject a bad label
        bad_row = df.iloc[0].copy()
        bad_row["label"] = "made-up-label"
        df = pd.concat([df, bad_row.to_frame().T], ignore_index=True)
        result = map_labels(df)
        assert "made-up-label" not in result["label"].values


# handle_missing
class TestHandleMissing:
    def test_drops_missing_statement(self):
        df = make_sample_df()
        df.loc[0, "statement"] = None
        result = handle_missing(df)
        assert len(result) == len(df) - 1

    def test_credit_columns_filled_with_zero(self):
        df = make_sample_df()
        df.loc[0, "false_counts"] = None
        result = handle_missing(df)
        assert result.loc[0, "false_counts"] == 0

    def test_credit_columns_are_int(self):
        df = make_sample_df()
        result = handle_missing(df)
        for col in CREDIT_COLUMNS:
            assert result[col].dtype == int


# engineer_features
class TestEngineerFeatures:
    def setup_method(self):
        df = make_sample_df()
        df = handle_missing(df)
        df = map_labels(df)
        self.df = engineer_features(df)

    def test_statement_clean_exists(self):
        assert "statement_clean" in self.df.columns

    def test_statement_len_positive(self):
        assert (self.df["statement_len"] > 0).all()

    def test_speaker_lie_rate_in_range(self):
        assert self.df["speaker_lie_rate"].between(0.0, 1.0).all()

    def test_capitalized_word_ratio_in_range(self):
        assert self.df["capitalized_word_ratio"].between(0.0, 1.0).all()

    def test_is_political_context_binary(self):
        assert set(self.df["is_political_context"].unique()).issubset({0, 1})


# capitalized_ratio
class TestCapitalizedRatio:
    def test_all_caps(self):
        assert _capitalized_ratio("FAKE NEWS ALERT") == pytest.approx(1.0)

    def test_no_caps(self):
        assert _capitalized_ratio("this is normal text") == pytest.approx(0.0)

    def test_mixed(self):
        ratio = _capitalized_ratio("this is FAKE news")
        assert 0.0 < ratio < 1.0

    def test_empty_string(self):
        assert _capitalized_ratio("") == 0.0

    def test_none(self):
        assert _capitalized_ratio(None) == 0.0