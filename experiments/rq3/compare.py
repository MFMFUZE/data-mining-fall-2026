"""RQ3 comparison tool. AI-assisted code: review and disclose before use.

This is a software scaffold, not the team's evaluation plan or analysis.
Never fits or scores on the reserved 15% test split.
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_io import load_adult
from src.project_split import development_and_holdout
from experiments.rq2.discretize_3bins import METHODS, assign, CONTINUOUS


CLASSIFIERS = {
    'decision_tree': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'random_forest': RandomForestClassifier,
}


# RQ2 uses hyphenated column labels; RQ3's data loader uses underscores.
RQ2_CONTINUOUS = tuple(c.replace("-", "_") for c in CONTINUOUS)


class TrainFittedDiscretizer(BaseEstimator, TransformerMixin):
    """Optional RQ2 three-bin representation, fitted on a training fold only.

    Reuses the actual equal-width, equal-depth and 1D k-means functions
    from the teammate's source script, rather than a separately coded method.
    """
    def __init__(self, method="equal_width", n_bins=3):
        self.method = method
        self.n_bins = n_bins

    def fit(self, X, y=None):
        if self.method not in METHODS:
            raise ValueError(f"Unknown discretization method: {self.method}")
        self.cut_points_ = {}
        for feature in RQ2_CONTINUOUS:
            if feature in X.columns:
                values = X[feature].dropna().to_numpy(dtype=float)
                if not len(values):
                    raise ValueError(f"No training values for {feature}")
                self.cut_points_[feature] = METHODS[self.method](values, self.n_bins)
        return self

    def transform(self, X):
        result = X.copy()
        for feature, edges in self.cut_points_.items():
            nonnull = result[feature].notna()
            if nonnull.any():
                result.loc[nonnull, feature] = assign(
                    result.loc[nonnull, feature].to_numpy(dtype=float), edges
                )
        return result


def make_pipeline(x, model, discretization_method=None):
    nums = x.select_dtypes(include='number').columns.tolist()
    cats = [c for c in x.columns if c not in nums]
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), nums),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cats),
    ], sparse_threshold=1.0)
    steps = []
    if discretization_method is not None:
        steps.append(('discretize', TrainFittedDiscretizer(method=discretization_method)))
    steps.extend([('preprocess', preprocessor), ('model', model)])
    return Pipeline(steps)


def complexity(model):
    clf = model.named_steps['model'] if isinstance(model, Pipeline) else model
    out = {'estimator': type(clf).__name__}
    if isinstance(clf, DecisionTreeClassifier):
        out.update(tree_nodes=int(clf.tree_.node_count), tree_depth=int(clf.get_depth()),
                   tree_leaves=int(clf.get_n_leaves()))
    elif isinstance(clf, RandomForestClassifier):
        out.update(n_trees=len(clf.estimators_),
                   total_tree_nodes=int(sum(t.tree_.node_count for t in clf.estimators_)),
                   mean_tree_depth=float(np.mean([t.get_depth() for t in clf.estimators_])))
    elif isinstance(clf, KNeighborsClassifier):
        out.update(training_rows_stored=int(clf.n_samples_fit_),
                   transformed_feature_count=int(clf.n_features_in_),
                   distance_metric=clf.effective_metric_)
    return out


def score(y, pred):
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average='macro', zero_division=0),
                macro_precision=precision_score(y, pred, average='macro', zero_division=0),
                macro_recall=recall_score(y, pred, average='macro', zero_division=0),
                f1_income_over_50k=f1_score(y, pred, pos_label='>50K', zero_division=0))


def run(config_file, dataset=None, smoke=False):
    config = json.loads(Path(config_file).read_text(encoding='utf-8'))
    data = load_adult(dataset or config.get('data_file', 'data/adult.data'))
    if set(data.income.unique()) != {'<=50K', '>50K'}:
        raise ValueError('Income values do not match the expected two classes.')
    # Match the uploaded RQ2 cleaning and final-row holdout exactly.
    development, test_holdout = development_and_holdout(data)
    representation = config.get('discretization_method')
    if representation not in (None, 'equal_width', 'equal_depth', 'kmeans'):
        raise ValueError('discretization_method must be null, equal_width, equal_depth, or kmeans')
    # Technical smoke only; downsample DEVELOPMENT data, never touch test data.
    if smoke and len(development) > 1600:
        development, _ = train_test_split(
            development, train_size=1600, stratify=development.income,
            random_state=int(config['validation_split_seed']))
    train, valid = train_test_split(
        development, test_size=float(config['validation_fraction']), stratify=development.income,
        random_state=int(config['validation_split_seed']))
    xt, yt = train.drop(columns='income'), train.income
    xv, yv = valid.drop(columns='income'), valid.income
    params = config['model_parameters']
    if set(params) != set(CLASSIFIERS):
        raise ValueError('Specify the three required classifier configurations.')
    models = {k: make_pipeline(xt, typ(**params[k]), representation)
              for k, typ in CLASSIFIERS.items()}
    models['majority_baseline'] = DummyClassifier(strategy='most_frequent')
    models['random_baseline'] = DummyClassifier(strategy='stratified',
                                               random_state=int(config['validation_split_seed']))
    rows, model_details = [], {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(xt, yt)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        predictions = model.predict(xv)
        pred_time = time.perf_counter() - start
        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=0)
        rows.append({'model': name, **score(yv, predictions),
                     'fit_seconds_single_run': fit_time,
                     'prediction_seconds_single_run': pred_time,
                     'serialized_pipeline_bytes': buffer.tell()})
        model_details[name] = complexity(model)
    out = Path(config['output_directory'])
    out.mkdir(parents=True, exist_ok=True)
    prefix = 'SMOKE_ONLY' if smoke else 'DEVELOPMENT_VALIDATION'
    csv = out / f'{prefix}_rq3_metrics.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    (out / f'{prefix}_complexity.json').write_text(json.dumps(model_details, indent=2))
    digest = hashlib.sha256(','.join(map(str, sorted(test_holdout.index))).encode()).hexdigest()
    meta = {'notice': 'Not final results; technical smoke test' if smoke else
            'Development validation only; never final test performance',
            'reserved_test_rows': len(test_holdout),
            'reserved_test_row_index_digest': digest,
            'development_training_rows': len(train), 'development_validation_rows': len(valid),
            'test_evaluated': False, 'split_method': 'last_15_percent_of_cleaned_rows',
            'discretization_method': representation, 'config': config}
    (out / f'{prefix}_metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'Wrote {csv}. Test set not evaluated.')
    return csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.config, dataset=args.data, smoke=args.smoke)
