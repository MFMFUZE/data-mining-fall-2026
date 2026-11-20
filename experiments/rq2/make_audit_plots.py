"""Generate reproducible Adult Income Part 1 distribution figures.

Usage: python make_audit_plots.py adult.data.txt OUTPUT_DIRECTORY
Uses the group's existing cleaning and holdout convention: drop missing rows,
then duplicates, then reserve the last round(15%) rows as test without shuffling.
Plots each attribute for development and test. Dataset statistics and plots
are created by this script, not manually inferred.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

COLS = ['age','workclass','fnlwgt','education','education_num','marital_status',
        'occupation','relationship','race','sex','capital_gain','capital_loss',
        'hours_per_week','native_country','income']


def prepare(source):
    raw = pd.read_csv(source, header=None, names=COLS,
                      skipinitialspace=True, na_values='?')
    removed_missing = int(raw.isna().any(axis=1).sum())
    no_missing = raw.dropna()
    removed_duplicates_after_missing = int(no_missing.duplicated().sum())
    clean = no_missing.drop_duplicates().reset_index(drop=True)
    n_test = int(round(0.15 * len(clean)))
    assert len(clean) == len(raw) - removed_missing - removed_duplicates_after_missing
    return raw, clean.iloc[:-n_test], clean.iloc[-n_test:], removed_missing, removed_duplicates_after_missing


def comparison_figure(feature, dev, test):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    numeric = pd.api.types.is_numeric_dtype(dev[feature])
    if numeric:
        vals = pd.concat([dev[feature], test[feature]])
        x = vals.to_numpy(dtype=float)
        lo, hi = float(np.min(x)), float(np.max(x))
        if np.isclose(lo, hi):
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, 31)
        for ax, frame, title in [(axs[0], dev, 'Development (85%)'),
                                 (axs[1], test, 'Reserved test (15%)')]:
            ax.hist(frame[feature].to_numpy(dtype=float), bins=edges,
                    weights=np.full(len(frame), 100 / len(frame)), edgecolor='none')
            ax.set(xlabel=feature, ylabel='Percent of split (%)', title=title)
            ax.set_xlim(lo, hi)
            ax.grid(axis='y', alpha=0.25)
    else:
        v = pd.concat([dev[feature], test[feature]]).astype(str)
        cats = v.value_counts().index.tolist()
        for ax, frame, title in [(axs[0], dev, 'Development (85%)'),
                                 (axs[1], test, 'Reserved test (15%)')]:
            vals = frame[feature].astype(str).value_counts().reindex(cats, fill_value=0)
            y = np.arange(len(cats))
            ax.barh(y, 100 * vals.to_numpy() / len(frame), height=0.8)
            ax.set_yticks(y, labels=cats, fontsize=7 if len(cats) > 12 else 8)
            ax.invert_yaxis()
            ax.set(xlabel='Percent of split (%)', title=title)
            ax.grid(axis='x', alpha=0.2)
        if len(cats) > 18:
            fig.set_size_inches(12, max(5, len(cats) * 0.22))
    fig.suptitle(f'Adult Income: {feature} — same cleaned dataset, fixed last-15% holdout', fontsize=11)
    return fig


def main(source, directory):
    out = Path(directory)
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    raw, dev, test, removed_missing, removed_dups = prepare(source)
    metrics = {
        'original_rows': len(raw),
        'rows_with_missing_values_removed': removed_missing,
        'duplicate_rows_original_file': int(raw.duplicated().sum()),
        'duplicates_removed_after_missing_rows': removed_dups,
        'clean_rows': len(dev) + len(test),
        'development_rows': len(dev),
        'reserved_test_rows': len(test),
        'holdout_method': 'Last round(15%) of cleaned rows; no additional shuffle',
        'attributes': COLS,
        'figures_generated': [],
    }
    import json
    with PdfPages(out / 'Part1_distribution_appendix.pdf') as pdf:
        for ix, feature in enumerate(COLS, 1):
            fig = comparison_figure(feature, dev, test)
            f = figures / f'{ix:02d}_{feature}.png'
            fig.savefig(f, dpi=150, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            metrics['figures_generated'].append(f.name)
    (out / 'figure_manifest.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='attributes' and k!='figures_generated'},indent=2))
    print(f'Created {len(metrics["figures_generated"])} comparison figures and a 15-page PDF appendix')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python make_audit_plots.py SOURCE_CSV OUTPUT_DIR')
    main(sys.argv[1], sys.argv[2])
