from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'

metrics = pd.read_csv(RESULTS / 'orthogonal_contrast_metrics.csv')
predictions = pd.read_csv(RESULTS / 'orthogonal_contrast_predictions.csv')
methods = ['Pooled_Contrast', 'Orthogonal_Contrast']
required_metrics = {'scope', 'held_out', 'n_genes', 'method', 'accuracy',
                    'macro_f1', 'dtp_recall', 'selected_jaccard_between_methods'}
required_predictions = {'scope', 'held_out', 'n_genes', 'method',
                        'sample_id', 'actual', 'predicted'}
if not required_metrics.issubset(metrics.columns):
    raise ValueError(f'Missing metrics columns: {required_metrics - set(metrics.columns)}')
if not required_predictions.issubset(predictions.columns):
    raise ValueError(f'Missing prediction columns: {required_predictions - set(predictions.columns)}')

# Figure 1: show the actual number of correct predictions per held-out cell line.
view = predictions.query('scope == "Internal_LOCO" and n_genes == 50').copy()
view['correct'] = view['actual'] == view['predicted']
counts = view.groupby(['held_out', 'method'])['correct'].sum().unstack()
totals = view.groupby(['held_out', 'method']).size().unstack()
if not set(methods).issubset(counts.columns):
    raise ValueError('Both methods must be present in internal predictions.')
counts = counts[methods].sort_index()
totals = totals[methods].loc[counts.index]
positions = np.arange(len(counts))
fig, ax = plt.subplots(figsize=(8.2, 4.8))
bar_width = 0.36
for j, method in enumerate(methods):
    vals = counts[method].to_numpy()
    bars = ax.bar(positions + (j - 0.5) * bar_width, vals,
                  width=bar_width, label=method.replace('_', ' '))
    for bar, value in zip(bars, vals):
        ax.annotate(str(int(value)),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
ax.set_xticks(positions, counts.index)
ax.set_ylim(0, max(6.8, float(totals.max().max()) + 0.8))
ax.set_ylabel('Correct predictions (of 6 samples)')
ax.set_title('Acute vs DTP | leave-one-cell-line-out | 50 genes')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS / 'figure_orthogonal_loco_counts.png', dpi=300)
plt.close(fig)

# Figure 2: matched 20/50-gene internal comparison.
internal = metrics.loc[metrics.scope == 'Internal_LOCO']
mean_f1 = internal.groupby(['n_genes', 'method'])['macro_f1'].mean().unstack()
if not set(methods).issubset(mean_f1.columns):
    raise ValueError('Both methods must be present in metrics.')
fig, ax = plt.subplots(figsize=(7.1, 4.6))
for method in methods:
    ax.plot(mean_f1.index.to_numpy(), mean_f1[method].to_numpy(),
            marker='o', linewidth=2, label=method.replace('_', ' '))
ax.set_xticks(sorted(mean_f1.index.unique()))
ax.set_xlim(min(mean_f1.index) - 3, max(mean_f1.index) + 3)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Number of selected genes')
ax.set_ylabel('Macro F1 (mean across 4 held-out cell lines)')
ax.set_title('Acute vs DTP | internal LOCO performance')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.2)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS / 'figure_orthogonal_internal_f1.png', dpi=300)
plt.close(fig)

# Keep exact aggregate values in a reusable manuscript-notes file.
internal_summary = internal.groupby(['n_genes', 'method'])[
    ['accuracy', 'macro_f1', 'dtp_recall']].mean().round(4)
external = metrics.loc[metrics.scope == 'External_development',
                       ['n_genes', 'method', 'accuracy', 'macro_f1', 'dtp_recall']]
overlap = internal.loc[internal.n_genes == 50].drop_duplicates('held_out')[
    ['held_out', 'selected_jaccard_between_methods']]
summary = [
    '# Orthogonal Contrast — exploratory result record',
    '',
    'Binary task: osi_acute versus osi_DTP; sample-wise rank input.',
    'All selection and classifier fitting were restricted to each training fold.',
    '',
    '## Internal leave-one-cell-line-out (mean across four cell lines)',
    '',
    '```text', internal_summary.to_string(), '```', '',
    '## External development (GSE255958 PC9; previously used in development)',
    '',
    '```text', external.to_string(index=False), '```', '',
    '## Selected-gene Jaccard overlap at 50 genes',
    '',
    '```text', overlap.to_string(index=False), '```', '',
    '## Interpretation and limitations',
    '',
    '- Internal 50-gene performance differs by one correct prediction across 24 held-out samples.',
    '- No recovery of DTP recall on the external-development cohort was observed.',
    '- The external-development cohort is NOT an untouched test set.',
    '- Small sample sizes and heterogeneous exposure protocols limit generalization.',
    '- No demonstrated superiority or novelty claim is supported by these results.',
    '',
    'Figures: `figure_orthogonal_loco_counts.png`, `figure_orthogonal_internal_f1.png`.',
]
(RESULTS / 'orthogonal_experiment_summary.md').write_text(
    '\n'.join(summary), encoding='utf-8'
)
print('Saved two data-derived figures and orthogonal_experiment_summary.md')
print('\n50-gene internal classification counts:\n', counts.to_string())
print('\nInternal Macro F1:\n', mean_f1.round(3).to_string())
print('\nExternal DTP recall:\n', external[['n_genes', 'method', 'dtp_recall']].to_string(index=False))
