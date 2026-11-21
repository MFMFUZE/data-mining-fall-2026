"""Part 1 hyperparameter candidates as concrete, scikit-learn-valid Python objects.

Model selection, experiment strategy and final parameter choice remain for the
team to decide. This module only enumerates the candidate values stated in
Section 3.3 and validates that they instantiate their estimators.
"""
from itertools import product
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def inverse_square_weights(distances):
    """Custom k-NN neighbor weights; always finite and strictly positive."""
    return 1.0 / (1.0 + np.square(distances))


PARAMETER_GRIDS = {
    "decision_tree": {
        "max_depth": (10, 20, 30),
        "max_features": ("sqrt", "log2", None),
        "criterion": ("gini", "entropy", "log_loss"),
    },
    "knn": {
        "n_neighbors": (3, 7, 15, 25),
        "metric": ("euclidean", "manhattan", "cosine"),
        "weights": ("uniform", "distance", inverse_square_weights),
    },
    "random_forest": {
        "n_estimators": (100, 200, 300),
        "max_depth": (10, 20, 30),
        "max_features": ("sqrt", "log2", None),
    },
}

CLASSIFIERS = {
    "decision_tree": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
}


def configurations(model):
    """Yield a dictionary for each distinct candidate combination."""
    grid = PARAMETER_GRIDS[model]
    for values in product(*(grid[key] for key in grid)):
        yield dict(zip(grid, values))


def validate():
    """Instantiate each config to catch invalid keyword/value representations."""
    counts = {}
    for model, cls in CLASSIFIERS.items():
        settings = list(configurations(model))
        for params in settings:
            cls(**params)
        counts[model] = len(settings)
    if counts != {"decision_tree": 27, "knn": 36, "random_forest": 27}:
        raise AssertionError(f"Unexpected configuration counts: {counts}")
    return counts


if __name__ == "__main__":
    result = validate()
    for label, count in result.items():
        print(f"{label}: {count}")
    print(f"TOTAL: {sum(result.values())}")
