"""Exploratory, training-only contrast-stability regularization.

Binary task: Acute vs DTP. Fixed comparison on the SAME pooled-contrast-selected
20/50 genes. Method is a research hypothesis, NOT a validated novel algorithm.

Objective: mean logistic loss + ridge*||w||^2/2
           + lambda*Var_cell_line(w^T(mean_DTP - mean_Acute)).

Evaluation: leave-one-cell-line-out in GSE193258, and a separate previously
seen external-development study (GSE255958 PC9, without PC9 in training).

No tuning against held-out labels, no target samples used to fit preprocessing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

LABELS = ['osi_acute', 'osi_DTP']
GENE_COUNTS = [20, 50]
RIDGE = 0.1  # fixed before comparing methods
PENALTY = 0.5  # fixed; do not tune using test data


def load_data():
    first = pd.read_csv(
        RAW / 'GSE193258_RNAseq_log2TPM_abundance.tsv.gz',
        sep='\t', index_col=0,
    ).T
    names = first.index.to_series()
    y_first = names.str.split('_', n=1).str[1].str.rsplit('_', n=1).str[0]
    lines = names.str.split('_').str[0]
    keep = y_first.isin(LABELS)
    first, y_first, lines = first.loc[keep], y_first.loc[keep], lines.loc[keep]

    second = pd.read_csv(
        PROCESSED / 'GSE255958_PC9_expression.csv.gz', index_col=0
    )
    second_meta = pd.read_csv(
        PROCESSED / 'GSE255958_PC9_metadata.csv'
    ).set_index('sample_id')
    second_meta = second_meta.loc[second_meta['condition'].isin(LABELS)]
    second = second.loc[second_meta.index]
    y_second = second_meta['condition']

    common = sorted(set(first.columns) & set(second.columns))
    if len(common) < max(GENE_COUNTS):
        raise ValueError('Insufficient shared genes')
    first, second = first.loc[:, common], second.loc[:, common]
    if first.isna().any().any() or second.isna().any().any():
        raise ValueError('Missing expression data')
    if len(first) != 24 or len(second) != 6:
        raise ValueError('Unexpected treatment-group sample counts')
    if not first.index.equals(y_first.index) or not second.index.equals(y_second.index):
        raise ValueError('Metadata mismatch')

    # Independent transformation of each sample; not fitted on the test cohort.
    return (first.rank(axis=1, pct=True), y_first, lines,
            second.rank(axis=1, pct=True), y_second)


def fit_linear_model(X, y, lines, lam):
    """Convex differentiable objective; exact gradient, L-BFGS optimization."""
    yy = np.where(np.asarray(y) == 'osi_DTP', 1., -1.)
    xx = np.asarray(X, dtype=float)
    groups = np.asarray(lines)
    contrasts = []
    for group in sorted(set(groups)):
        a = xx[(groups == group) & (yy == -1.)]
        d = xx[(groups == group) & (yy == 1.)]
        if len(a) < 2 or len(d) < 2:
            raise ValueError(f'Too few samples in training group {group}')
        contrasts.append(d.mean(axis=0) - a.mean(axis=0))
    contrasts = np.stack(contrasts)
    n, p = xx.shape

    def objective(parameters):
        w, b = parameters[:p], parameters[p]
        margins = yy * (xx @ w + b)
        loss = np.logaddexp(0., -margins).mean()
        neg_prob = expit(-margins)
        dw = -xx.T @ (yy * neg_prob) / n + RIDGE * w
        db = -(yy * neg_prob).mean()
        loss += 0.5 * RIDGE * float(w @ w)

        if lam:
            group_margin = contrasts @ w
            centered = group_margin - group_margin.mean()
            loss += lam * float(np.mean(centered ** 2))
            dw += 2 * lam * (contrasts.T @ centered) / len(contrasts)
        return float(loss), np.r_[dw, db]

    result = minimize(
        objective, np.zeros(p + 1), method='L-BFGS-B', jac=True,
        options={'maxiter': 3000, 'ftol': 1e-10},
    )
    if not result.success:
        raise RuntimeError(f'Optimization did not converge: {result.message}')
    return result.x[:p], float(result.x[p]), contrasts


def run_fold(scope, held_out, X_train, y_train, lines_train, X_test, y_test):
    deltas = (X_train.loc[y_train == 'osi_DTP'].mean(axis=0)
              - X_train.loc[y_train == 'osi_acute'].mean(axis=0))
    ordered_genes = deltas.abs().sort_values(ascending=False, kind='stable').index
    metrics, predictions, signatures = [], [], []

    for n_genes in GENE_COUNTS:
        genes = ordered_genes[:n_genes].tolist()
        scaler = StandardScaler().fit(X_train[genes])
        xtr = scaler.transform(X_train[genes]); xte = scaler.transform(X_test[genes])
        for method, lam in [('Pooled_Logistic', 0.),
                            ('Contrast_Stable', PENALTY)]:
            w, b, group_contrasts = fit_linear_model(xtr, y_train, lines_train, lam)
            probabilities = expit(xte @ w + b)
            predicted = np.where(probabilities >= 0.5, 'osi_DTP', 'osi_acute')
            metrics.append({
                'scope': scope, 'held_out': held_out, 'method': method,
                'n_genes': n_genes, 'n_train': len(y_train), 'n_test': len(y_test),
                'accuracy': accuracy_score(y_test, predicted),
                'macro_f1': f1_score(y_test, predicted, labels=LABELS,
                                      average='macro', zero_division=0),
                'dtp_recall': recall_score(y_test, predicted,
                                           pos_label='osi_DTP', zero_division=0),
                'train_group_margin_sd': float(np.std(group_contrasts @ w, ddof=0)),
                'penalty_lambda': lam,
            })
            for sid, actual, pred, prob in zip(X_test.index, y_test, predicted, probabilities):
                predictions.append({
                    'scope': scope, 'held_out': held_out, 'method': method,
                    'n_genes': n_genes, 'sample_id': sid,
                    'actual': actual, 'predicted': pred, 'dtp_probability': prob,
                })
        for i, g in enumerate(genes, 1):
            signatures.append({'scope': scope, 'held_out': held_out,
                               'n_genes': n_genes, 'rank': i, 'gene': g})
    return metrics, predictions, signatures


def main():
    X, y, lines, X_external, y_external = load_data()
    print('Internal samples:', len(y), '; External development samples:', len(y_external))
    print('Per-cell-line labels:\n', pd.crosstab(lines, y))
    all_metrics, all_predictions, all_genes = [], [], []
    for held in sorted(lines.unique()):
        mask = lines != held
        out = run_fold('Internal_LOCO', held,
                       X.loc[mask], y.loc[mask], lines.loc[mask],
                       X.loc[~mask], y.loc[~mask])
        for full, part in zip((all_metrics, all_predictions, all_genes), out):
            full.extend(part)

    # External PC9 study has been inspected during development (not untouched).
    mask = lines != 'PC9'
    out = run_fold('External_development', 'GSE255958_PC9',
                   X.loc[mask], y.loc[mask], lines.loc[mask],
                   X_external, y_external)
    for full, part in zip((all_metrics, all_predictions, all_genes), out):
        full.extend(part)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(RESULTS / 'contrast_stability_metrics.csv', index=False)
    pd.DataFrame(all_predictions).to_csv(
        RESULTS / 'contrast_stability_predictions.csv', index=False)
    pd.DataFrame(all_genes).to_csv(
        RESULTS / 'contrast_stability_selected_genes.csv', index=False)

    print('\nINTERNAL LOCO (mean across 4 held-out cell lines):')
    print(metrics.loc[metrics.scope == 'Internal_LOCO'].groupby(['n_genes', 'method'])
          [['accuracy', 'macro_f1', 'dtp_recall', 'train_group_margin_sd']]
          .mean().round(3).to_string())
    print('\nEXTERNAL DEVELOPMENT (PC9 in unseen study, NOT a fresh holdout):')
    print(metrics.loc[metrics.scope == 'External_development',
          ['n_genes', 'method', 'accuracy', 'macro_f1', 'dtp_recall',
           'train_group_margin_sd']].round(3).to_string(index=False))
    print('\nFinished. Two methods use identical features and preprocessing.')
    print('Caution: contrast stability regularization is a tested hypothesis, not proven novelty.')


if __name__ == '__main__':
    main()
