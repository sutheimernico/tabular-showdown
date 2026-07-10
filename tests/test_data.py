"""Tests for the Adult dataset loading pipeline.

These tests run against the real data/adult.data and data/adult.test files
checked into the repo, not fixtures, so they double as a smoke test that the
pipeline still parses the actual UCI files correctly.
"""

from pathlib import Path

import pandas as pd
import pytest

from tabular_showdown.data import (
    CATEGORICAL_COLUMNS,
    COLUMNS,
    NUMERIC_COLUMNS,
    frozen_eval_set,
    load_adult,
    split_features_target,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TRAIN_ROWS = 32561
TEST_ROWS = 16281


def _row_fingerprints(df: pd.DataFrame) -> set[tuple[str, ...]]:
    """Stringified row tuples, used to check set membership/overlap between dataframes."""
    return set(map(tuple, df.astype(str).itertuples(index=False, name=None)))


@pytest.fixture(scope="module")
def loaded() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_adult(DATA_DIR)


def test_row_counts(loaded):
    train_df, test_df = loaded
    assert len(train_df) == TRAIN_ROWS
    assert len(test_df) == TEST_ROWS


def test_columns_match_spec(loaded):
    train_df, test_df = loaded
    assert list(train_df.columns) == COLUMNS
    assert list(test_df.columns) == COLUMNS


def test_no_gross_leakage_between_train_and_test(loaded):
    """Train and test should be almost entirely disjoint by content.

    Not exactly disjoint: the raw UCI extraction contains a small number of
    coincidentally identical rows across the two files (documented in
    adult.names as "Duplicate or conflicting instances"). We assert the
    overlap stays tiny (well under 1% of the test set) rather than zero, so
    the test still catches real pipeline bugs (e.g. accidentally loading the
    same file for both splits, which would push overlap into the thousands)
    without failing on this known, harmless quirk of the dataset.
    """
    train_df, test_df = loaded
    overlap = _row_fingerprints(train_df) & _row_fingerprints(test_df)
    assert len(overlap) < 0.01 * len(test_df)


def test_label_mapping(loaded):
    train_df, test_df = loaded
    for df in (train_df, test_df):
        assert set(df["income"].unique()) == {0, 1}
        positive_rate = df["income"].mean()
        assert 0.20 <= positive_rate <= 0.28


def test_no_literal_question_marks_and_missing_is_nan(loaded):
    train_df, test_df = loaded
    for df in (train_df, test_df):
        assert not (df.astype(str) == "?").any().any()
        # workclass, occupation and native_country are known to contain missing
        # values in this dataset; they must show up as NaN, not be imputed away.
        assert df[["workclass", "occupation", "native_country"]].isna().any().any()


def test_dtypes(loaded):
    train_df, test_df = loaded
    for df in (train_df, test_df):
        for col in CATEGORICAL_COLUMNS:
            assert isinstance(df[col].dtype, pd.CategoricalDtype)
        for col in NUMERIC_COLUMNS:
            assert df[col].dtype == "int64"
        assert df["income"].dtype == "int64"


def test_frozen_eval_set_is_deterministic(loaded):
    _, test_df = loaded
    first = frozen_eval_set(test_df)
    second = frozen_eval_set(test_df)
    pd.testing.assert_frame_equal(first, second)


def test_frozen_eval_set_size_and_sourced_from_test(loaded):
    _, test_df = loaded
    eval_df = frozen_eval_set(test_df, n=4000, seed=42)
    assert len(eval_df) == 4000
    assert eval_df.index.isin(test_df.index).all()
    # values must match the test set exactly at those indices, not just the index
    pd.testing.assert_frame_equal(eval_df, test_df.loc[eval_df.index])


def test_frozen_eval_set_is_stratified(loaded):
    _, test_df = loaded
    eval_df = frozen_eval_set(test_df, n=4000, seed=42)
    test_rate = test_df["income"].mean()
    eval_rate = eval_df["income"].mean()
    assert abs(test_rate - eval_rate) <= 0.02


def test_split_features_target(loaded):
    train_df, _ = loaded
    X, y = split_features_target(train_df)
    assert "income" not in X.columns
    assert list(X.columns) == COLUMNS[:-1]
    assert y.name == "income"
    assert len(X) == len(y) == len(train_df)
