# Principles of Data Mining — Fall 2026

**Programming Project | Team of 3 | Status: Initial setup**

The team will use the provided **Adult Income** dataset to study how model selection, discretization, and classifier choices influence classification performance and performance estimates.

> This repository is a collaboration scaffold, **not** a finished evaluation plan or analysis. Each team member must decide, justify, implement, and understand their own experiments. Follow your instructor's AI-use and disclosure requirements.

## Team ownership

| Research question | Topic | Team member |
|---|---|---|
| RQ1 | Model selection: fixed holdout, repeated holdout (3 seeds), 3-fold CV | [Assign name] |
| RQ2 | Discretization: equal width, equal depth, 1D k-means; bins and continuous features | [Assign name] |
| RQ3 | Classifiers: Decision Tree, k-NN, Random Forest; performance, runtime, complexity, interpretability | [Assign name] |

All three members must understand the shared pipeline and combined findings.

## Deliverables

**Deliverable 1 — Initial experimental setup**

- Team information, contributions, and detailed AI-use statement.
- Description of each required classifier's strengths, limitations, and assumptions.
- Hyperparameter plan: parameters, at least 3 candidate values for each selected hyperparameter, explanation, and total combinations.
- Dataset audit: attribute types, distributions, missing values, duplicates, preprocessing rationale, and distribution plots for every attribute in the 15% test set **and** the remainder.
- Implement 3-bin equal-width, equal-depth, and 1D k-means discretization of continuous numeric features. Provide three datasets and pairwise percentage agreement of bin assignments.

**Deliverable 2 — Main project package and final report.** The supplied handout does not list a detailed final-report checklist; confirm specific submission requirements and dates with your instructor.

## Setup

1. Clone this repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate       # Windows PowerShell alternative
   python -m pip install -r requirements.txt
   ```
3. Put the **instructor-provided** Adult Income file in `data/adult.data` (rename the downloaded `adult.data(1).txt` if necessary). This file is intentionally excluded from version control.
4. Check that data loading works:
   ```bash
   python -m src.data_io
   ```
5. Use notebooks for exploration; move reusable code into `src/`, and keep each RQ's experiments in its own `experiments/` folder.

**Important:** Reserve 15% of all records as the test set before fitting data-dependent transformations. Do not use this held-out test set for model selection. The team must choose and justify its own implementation details.

## Folder layout

```text
data-mining-fall-2026/
├── .github/PULL_REQUEST_TEMPLATE.md
├── data/README.md
├── src/__init__.py
├── src/data_io.py
├── experiments/rq1/README.md
├── experiments/rq2/README.md
├── experiments/rq3/README.md
├── notebooks/README.md
├── results/README.md
├── CONTRIBUTING.md
├── requirements.txt
└── .gitignore
```

## Shared writing document

[Google Doc — Data Mining Group Project Template](https://docs.google.com/document/d/1Vu4EA3jQ-uCuMI-zkmookbP7Sr8KUZKKxVM3mdD3HPk/edit)

## Dataset source

Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
