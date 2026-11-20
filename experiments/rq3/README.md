# RQ3 — Model Performance

**Owner:** Daniel Bencomo  
**Course:** CAP 5771, Principles of Data Mining, Fall 2026

**Research question:** How do the classifiers compare in predictive performance, runtime, model size/complexity and interpretability?

## Status

This directory contains an AI-assisted, executed **software scaffold**, not an approved experiment design, verified final findings, or report discussion. The supplied project rules expressly prohibit AI-written evaluation plans, analysis, discussion, and recommendations. Your group must own those parts. Disclose AI-generated/restructured code.

## Run technical check

From the repository root:

```bash
python -m pip install -r requirements.txt
# Place the assigned Adult Income file in data/adult.data OR use --data
python -m experiments.rq3.compare --config experiments/rq3/smoke_config.json --smoke
```

For the original file name, append `--data 'path/to/adult.data(1).txt'`.

The code reserves 15% of the dataset as an untouched test set. It then creates a development-training/validation split; numerical and categorical preprocessing are fitted **only** on development-training rows. All reported metrics are on development validation, never the final test set. In smoke mode only a small portion of development records is used. Smoke output must **not** be cited as project conclusions.

## Fields recorded

- Accuracy, balanced accuracy, macro precision, macro recall, macro F1, positive-class (>50K) F1.
- One-run fit and prediction elapsed time (seconds). These are not stable time benchmarks; the team needs its own justified measurement approach.
- Serialized **entire fitted pipeline** size (bytes), which includes preprocessing objects. This is not a RAM footprint measurement.
- Model structure: Decision Tree nodes/depth/leaves; Random Forest tree count/total nodes/mean depth; k-NN stored training rows/encoded feature count. These are descriptive attributes, not interpretation scores.
- Majority-class and random-prediction baselines.

## Student-owned completion checklist

- [ ] Review the code line by line, test it, and fix issues.
- [ ] Agree on identical split and preprocessing conventions with RQ1 and RQ2; share held-out row IDs or a split manifest.
- [ ] Decide and personally justify model configurations, target metrics, timing methodology, and validation strategy.
- [ ] Run the group-approved setup, preserve exact code version, environment, seeds, and results.
- [ ] Check that generated metrics and complexity measures are correct and comparable.
- [ ] Write your own comparative analysis of predictive performance, runtime, model complexity and interpretability.
- [ ] Log all AI help in the contributions/disclosure section.

## Result placeholders

| Model | Validation metric(s) | Fit time | Predict time | Size | Complexity notes | Interpretability notes |
|---|---|---|---|---|---|---|
| Decision Tree | [your output] | [your output] | [your output] | [your output] | [inspect] | [your writing] |
| k-NN | [your output] | [your output] | [your output] | [your output] | [inspect] | [your writing] |
| Random Forest | [your output] | [your output] | [your output] | [your output] | [inspect] | [your writing] |
| Majority baseline | [your output] | [your output] | [your output] | [your output] | — | [your writing] |
| Random baseline | [your output] | [your output] | [your output] | [your output] | — | [your writing] |
