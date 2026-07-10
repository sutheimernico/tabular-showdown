"""Load and clean the UCI Adult / Census Income dataset.

Provides the single source of truth for train/test loading so every
downstream milestone (baselines, TabPFN, the learning-size curve) sees the
exact same cleaned data and the exact same frozen evaluation subset.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

_INCOME_MAP = {"<=50K": 0, ">50K": 1}


def _read_raw(path: Path, skiprows: int = 0) -> pd.DataFrame:
    """Read one of the adult.* CSVs with no dtype assumptions yet.

    skipinitialspace strips the leading space UCI puts after every comma
    (" Private" -> "Private"); na_values="?" turns the literal missing-value
    marker into a real NaN.
    """
    return pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,
        na_values="?",
    )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # adult.test labels have a trailing period ("<=50K.", ">50K."); adult.data
    # doesn't, so rstrip is a no-op there. Normalize both to the same 0/1 int.
    df["income"] = df["income"].str.rstrip(".").map(_INCOME_MAP).astype("int64")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    for col in NUMERIC_COLUMNS:
        df[col] = df[col].astype("int64")

    return df


def load_adult(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean adult.data (train) and adult.test (test).

    adult.test starts with a junk header line ("|1x3 Cross validator"), hence
    skiprows=1 there. Missing values ("?") stay as NaN — no imputation here.
    """
    train_df = _clean(_read_raw(data_dir / "adult.data"))
    test_df = _clean(_read_raw(data_dir / "adult.test", skiprows=1))
    return train_df, test_df


def frozen_eval_set(test_df: pd.DataFrame, n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Deterministic, stratified fixed-size subsample of the test set.

    This is THE evaluation set for the whole benchmark: TabPFN v2 predicting
    on CPU over the full ~16k test rows is too slow to repeat at every point
    of the learning-size curve, so every model/run is scored on this same
    frozen 4k stratified subset to keep the comparison fair and fast.
    """
    eval_idx, _ = train_test_split(
        test_df.index,
        train_size=n,
        stratify=test_df["income"],
        random_state=seed,
    )
    return test_df.loc[eval_idx].sort_index()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into feature matrix X and target series y."""
    return df.drop(columns="income"), df["income"]
