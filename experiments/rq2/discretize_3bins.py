"""Section 3.5: equal-width, equal-depth and 1D k-means discretization (3 bins),
implemented from scratch, plus pairwise agreement between the methods.
Usage: python discretize_3bins.py adult.data.txt
"""
import sys
import numpy as np
import pandas as pd

COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
CONTINUOUS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
K = 3
SEED = 42


def equal_width_edges(x, k):
    # k bins of equal width between min and max -> k-1 interior cut points
    lo, hi = x.min(), x.max()
    return lo + (hi - lo) * np.arange(1, k) / k


def equal_depth_edges(x, k):
    # cut points at the 1/k, 2/k, ... quantiles so each bin holds ~n/k values
    # (if many values are tied, edges coincide and some bins end up empty)
    return np.quantile(x, np.arange(1, k) / k)


def kmeans_1d_edges(x, k, n_iter=100, seed=SEED):
    # Lloyd's algorithm in 1D; cut points are midpoints between sorted centers
    rng = np.random.default_rng(seed)
    centers = np.sort(rng.choice(np.unique(x), size=min(k, len(np.unique(x))), replace=False)).astype(float)
    for _ in range(n_iter):
        labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
        new = np.array([x[labels == j].mean() if np.any(labels == j) else centers[j]
                        for j in range(len(centers))])
        new = np.sort(new)
        if np.allclose(new, centers):
            break
        centers = new
    return (centers[:-1] + centers[1:]) / 2


def assign(x, edges):
    # bins are (-inf, e1], (e1, e2], (e2, inf) -> index 0 (low), 1 (mid), 2 (high).
    # Every method orders bins low->high, so bins are matched by order before comparing.
    # If edges coincide (heavy ties), the middle bin is simply empty.
    return np.searchsorted(edges, x, side="left")


METHODS = {"equal_width": equal_width_edges, "equal_depth": equal_depth_edges, "kmeans": kmeans_1d_edges}


def main(path):
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True, na_values="?")
    df = df.dropna().drop_duplicates().reset_index(drop=True)   # same cleaning as Section 3.4

    # Fit edges on development data only; apply the same edges to the test set
    # Test set = last 15% of the cleaned rows, as described in Section 3.4
    n_test = int(round(0.15 * len(df)))
    is_test = np.zeros(len(df), bool); is_test[-n_test:] = True
    dev, test = df[~is_test], df[is_test]

    binned = {}
    for m, fn in METHODS.items():
        out = df.copy()
        print(f"\n== {m} ==")
        for c in CONTINUOUS:
            edges = fn(dev[c].to_numpy(float), K)
            out[c] = assign(df[c].to_numpy(float), edges)
            counts = np.bincount(out.loc[~is_test, c], minlength=K)
            print(f"  {c:15s} edges={np.round(edges, 1)}  dev bin counts={counts.tolist()}")
        out.to_csv(f"adult_{m}_3bins.csv", index=False)
        binned[m] = out[~is_test]

    print("\n== Pairwise agreement (% of dev values in the same bin) ==")
    names = list(METHODS)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = binned[names[i]], binned[names[j]]
            per = {c: 100 * np.mean(a[c].to_numpy() == b[c].to_numpy()) for c in CONTINUOUS}
            per["OVERALL"] = 100 * np.mean(a[CONTINUOUS].to_numpy() == b[CONTINUOUS].to_numpy())
            rows.append(pd.Series(per, name=f"{names[i]} vs {names[j]}"))
    print(pd.DataFrame(rows).round(1).T.to_string())


if __name__ == "__main__":
    main(sys.argv[1])
