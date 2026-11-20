"""Deterministic, project-wide Adult Income cleaning and test holdout.

Matches the uploaded RQ2 script: remove missing rows and duplicates,
reset row order, then reserve the LAST rounded 15% of cleaned records.
The original order is not shuffled here. This helper does not score the test set.
"""
import pandas as pd


def development_and_holdout(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = data.dropna().drop_duplicates().reset_index(drop=True)
    n_test = int(round(0.15 * len(cleaned)))
    if n_test < 1 or len(cleaned) - n_test < 2:
        raise ValueError("Not enough clean rows to create development and 15% holdout sets")
    development = cleaned.iloc[:-n_test].copy()
    test_holdout = cleaned.iloc[-n_test:].copy()
    return development, test_holdout
