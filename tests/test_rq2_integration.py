"""Mechanical checks of RQ2-to-RQ3 interoperability; not final evaluation."""
import numpy as np
import pandas as pd

from experiments.rq2.discretize_3bins import equal_width_edges, equal_depth_edges, kmeans_1d_edges, assign
from experiments.rq3.compare import TrainFittedDiscretizer
from src.project_split import development_and_holdout


def test_15_percent_split_matches_uploaded_rq2_convention():
    df = pd.DataFrame({"age": range(1, 102), "income": ["<=50K"] * 101})
    df.loc[100, "age"] = np.nan  # missing row is dropped before computing holdout length
    df.loc[5] = df.loc[4]        # duplicate row is dropped as well
    dev, test = development_and_holdout(df)
    assert len(dev) + len(test) == 99
    assert len(test) == round(0.15 * 99)
    assert dev.index.max() + 1 == test.index.min()


def test_pipeline_uses_teammates_three_bin_edges():
    xt = pd.DataFrame({"age": [20, 22, 24, 40, 42, 44]})
    for key, fn in {
        "equal_width": equal_width_edges,
        "equal_depth": equal_depth_edges,
        "kmeans": kmeans_1d_edges,
    }.items():
        transform = TrainFittedDiscretizer(method=key).fit(xt)
        np.testing.assert_array_equal(
            transform.transform(xt)["age"].to_numpy(dtype=int),
            assign(xt["age"].to_numpy(dtype=float), fn(xt["age"].to_numpy(dtype=float), 3)),
        )


def test_rq3_transformer_fits_edges_only_on_training_values():
    train = pd.DataFrame({"age": [10, 20, 30, 40]})
    outlier_valid = pd.DataFrame({"age": [10000]})
    tr = TrainFittedDiscretizer(method="equal_width").fit(train)
    np.testing.assert_allclose(tr.cut_points_["age"], [20, 30])
    assert tr.transform(outlier_valid)["age"].iloc[0] == 2
    np.testing.assert_allclose(tr.cut_points_["age"], [20, 30])
