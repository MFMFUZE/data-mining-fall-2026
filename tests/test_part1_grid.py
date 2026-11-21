"""Mechanical checks for the Part 1 parameter grid."""
import numpy as np

from experiments.rq1.hyperparameter_grid import validate, inverse_square_weights, configurations


def test_grid_counts():
    assert validate() == {"decision_tree": 27, "knn": 36, "random_forest": 27}
    assert sum(len(list(configurations(name)))
               for name in ("decision_tree", "knn", "random_forest")) == 90


def test_inverse_square_weight_formula():
    np.testing.assert_allclose(
        inverse_square_weights(np.array([[0.0, 1.0, 2.0]])),
        np.array([[1.0, 0.5, 0.2]]),
    )
