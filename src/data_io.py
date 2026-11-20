"""Minimal CSV-reading helper for the instructor-provided Adult Income dataset.

This module intentionally does not clean, impute, encode, split, or fit any models.
The team owns its preprocessing decisions and must avoid fitting transformations
with held-out test records.
"""
from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]


def load_adult(path: str | Path = "data/adult.data") -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find {path}. Place the provided dataset at data/adult.data."
        )
    data = pd.read_csv(
        path, header=None, names=COLUMN_NAMES, skipinitialspace=True,
        na_values=["?"],
    )
    if data.shape[1] != len(COLUMN_NAMES):
        raise ValueError("Unexpected number of columns in Adult dataset")
    return data


if __name__ == "__main__":
    frame = load_adult()
    print(f"Loaded {len(frame):,} records and {len(frame.columns)} columns.")
    print("Data types:")
    print(frame.dtypes.to_string())
    print("Missing values per column:")
    print(frame.isna().sum().to_string())
    print("Target class counts:")
    print(frame["income"].value_counts(dropna=False).to_string())
