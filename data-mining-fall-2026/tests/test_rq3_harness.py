"""Mechanical checks for the AI-assisted RQ3 code scaffold.

These are software checks, not a statistical validation of the project results.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.rq3.compare import run
from src.data_io import COLUMN_NAMES


def test_smoke_generates_only_development_metrics(tmp_path: Path) -> None:
    n = 240
    rng = np.random.default_rng(5)
    data = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "workclass": rng.choice(["Private", "Self-emp"], n),
        "fnlwgt": rng.integers(10000, 350000, n),
        "education": rng.choice(["HS-grad", "Bachelors"], n),
        "education_num": rng.choice([9, 13], n),
        "marital_status": rng.choice(["Never-married", "Married-civ-spouse"], n),
        "occupation": rng.choice(["Sales", "Tech-support"], n),
        "relationship": rng.choice(["Husband", "Not-in-family"], n),
        "race": ["Other"] * n,
        "sex": rng.choice(["Male", "Female"], n),
        "capital_gain": np.zeros(n, dtype=int),
        "capital_loss": np.zeros(n, dtype=int),
        "hours_per_week": rng.integers(15, 65, n),
        "native_country": ["United-States"] * n,
        "income": ["<=50K" if i % 2 == 0 else ">50K" for i in range(n)],
    })[COLUMN_NAMES]
    source = tmp_path / "synthetic_adult.data"
    data.to_csv(source, header=False, index=False)
    config = {
        "test_split_seed": 12, "validation_split_seed": 14,
        "validation_fraction": 0.2, "output_directory": str(tmp_path / "results"),
        "model_parameters": {
            "decision_tree": {"max_depth": 3, "random_state": 14},
            "knn": {"n_neighbors": 3},
            "random_forest": {"n_estimators": 4, "max_depth": 3, "random_state": 14},
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    metrics_path = run(path, smoke=True, dataset_override=source)
    metrics = pd.read_csv(metrics_path)
    metadata = json.loads((tmp_path / "results" / "SMOKE_ONLY_rq3_run_metadata.json").read_text())
    assert set(metrics["model"]) == {
        "decision_tree", "knn", "random_forest", "majority_baseline", "random_baseline"
    }
    assert metrics["accuracy"].between(0, 1).all()
    assert metrics["f1_macro"].between(0, 1).all()
    assert metadata["reserved_test_rows"] == 36
    assert metadata["test_set_used_in_training_or_scoring"] is False
    assert metadata["development_train_rows"] + metadata["development_validation_rows"] == 204
