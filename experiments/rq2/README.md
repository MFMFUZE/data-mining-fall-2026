# RQ2 — Discretization

**Owner:** Kenneth Currier (research question). The uploaded 3-bin script was supplied by the group and is preserved at `discretize_3bins.py`.

The source implements equal width, equal depth and one-dimensional k-means, and prints bin counts and pairwise agreement. It does the same missing-value/duplicate removal as the group document and reserves the **last 15% of cleaned rows** as the holdout set. It writes three CSV datasets in the current working directory. Do not commit these files.

## Reproduce the reported Part 1 binning numbers

Run in the repository root with the Adult source saved locally as `data/adult.data`:

```bash
python experiments/rq2/discretize_3bins.py data/adult.data
```

The result should have 30,096 cleaned rows (25,582 development, 4,514 holdout) for the supplied file. Development-set overall bin agreement: 56.9% equal-width vs equal-depth; 81.6% equal-width vs k-means; 71.2% equal-depth vs k-means.

**Important:** the original script does **not** randomize rows before reserving the last 15%. Documented split labels should reflect this actual behavior. It fits discretization cut points on the full development set, which is appropriate for its Part 1 descriptive output but not a validation run for hyperparameter selection. In RQ3, `TrainFittedDiscretizer` reuses the same algorithms and fits edges on each training portion only.
