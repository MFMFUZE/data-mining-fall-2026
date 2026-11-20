# RQ1 — Model Selection

**Owner:** [Team member]

**Research question (from assignment):** How do different model-selection strategies affect which model/configuration is selected and the performance estimate that is reported?

Strategies specified in the handout: fixed holdout, repeated holdout (3 seeds), and 3-fold CV.

**Team-owned experiment design, code, logs, figures, and findings:** [Add here]

## Validated Part 1 hyperparameter grid

`hyperparameter_grid.py` enumerates 27 Decision Tree, 36 k-NN, and 27 Random Forest combinations (90 in total). The third k-NN weighting option is an explicitly defined Python callable `inverse_square_weights`. The default-p=2 Minkowski metric duplicates Euclidean, so the three listed metrics are Euclidean, Manhattan, and Cosine.

Run from repository root:

```bash
python -m experiments.rq1.hyperparameter_grid
```
