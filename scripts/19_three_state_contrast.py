"""Exploratory three-state modelling with treatment-state-specific feature selection.

Compares flat three-class classification against two hierarchical approaches.
All gene selection and model fitting occur inside training folds. This is a
*development* experiment, not independent confirmatory validation.

Run from the repository root:
    python scripts/19_three_state_contrast.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

CLASSES = ["DMSO", "osi_acute", "osi_DTP"]
GENE_COUNTS = [20, 50]
PENALTY = 0.5  # Fixed exploratory value; do not tune using test labels.


def new_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42),
    )


def top_genes(scores, n):
    return scores.sort_values(ascending=False, kind="stable").head(n).index.tolist()


def load_data():
    first = pd.read_csv(
        RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
        sep="\t", index_col=0,
    ).T
    names = first.index.to_series()
    y_first = names.str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0]
    cell_lines = names.str.split("_").str[0]
    keep = y_first.isin(CLASSES)
    first, y_first, cell_lines = first.loc[keep], y_first.loc[keep], cell_lines.loc[keep]

    second = pd.read_csv(
        PROCESSED / "GSE255958_PC9_expression.csv.gz", index_col=0,
    )
    second_meta = pd.read_csv(
        PROCESSED / "GSE255958_PC9_metadata.csv"
    ).set_index("sample_id")
    second_meta = second_meta.loc[second_meta["condition"].isin(CLASSES)]
    second = second.loc[second_meta.index]
    y_second = second_meta["condition"]

    common = sorted(set(first.columns) & set(second.columns))
    if len(common) < max(GENE_COUNTS):
        raise ValueError("Insufficient shared genes")
    first, second = first.loc[:, common], second.loc[:, common]
    if first.isna().any().any() or second.isna().any().any():
        raise ValueError("Missing expression values")
    if len(first) != 36 or len(second) != 9:
        raise ValueError("Unexpected sample counts")
    for study_y in [y_first, y_second]:
        if not set(study_y.unique()) == set(CLASSES):
            raise ValueError("Unexpected classes")

    # Sample-specific transform: no information is learned from other samples.
    return (
        first.rank(axis=1, pct=True, method="average"),
        y_first,
        cell_lines,
        second.rank(axis=1, pct=True, method="average"),
        y_second,
    )


def select_gene_sets(X_train, y_train, training_lines, n_genes):
    # Drop constant genes using training data only.
    usable = X_train.columns[X_train.var(axis=0) > 0]
    train = X_train.loc[:, usable]

    # A. Standard three-class ANOVA feature filter.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f_values, _ = f_classif(train.to_numpy(), y_train.to_numpy())
    flat_scores = pd.Series(
        np.nan_to_num(f_values, nan=-1.0, posinf=1e10, neginf=-1.0),
        index=usable,
    )
    flat_genes = top_genes(flat_scores, n_genes)

    # B. Stage 1: distinguish untreated controls from all drug-exposed cells.
    n_stage1 = n_genes // 2
    n_stage2 = n_genes - n_stage1
    treated = y_train != "DMSO"
    first_stage = (train.loc[treated].mean() - train.loc[~treated].mean()).abs()
    stage1_genes = top_genes(first_stage, n_stage1)

    # C. Stage 2: focus explicitly on acute versus persister treatment states.
    acute = y_train == "osi_acute"
    dtp = y_train == "osi_DTP"
    pooled_contrast = train.loc[dtp].mean() - train.loc[acute].mean()
    pooled_genes = top_genes(pooled_contrast.abs(), n_stage2)

    # Training-cell-line-specific DTP - acute contrasts.
    line_effects = {}
    for line in sorted(training_lines.unique()):
        line_mask = training_lines == line
        acute_line = train.loc[line_mask & acute]
        dtp_line = train.loc[line_mask & dtp]
        if acute_line.empty or dtp_line.empty:
            raise ValueError(f"Missing treatment state for training cell line {line}")
        line_effects[line] = dtp_line.mean() - acute_line.mean()
    effects = pd.DataFrame(line_effects)
    consensus_scores = effects.mean(axis=1).abs() - PENALTY * effects.std(axis=1, ddof=0)
    consistent_genes = top_genes(consensus_scores, n_stage2)

    return {
        "Flat_ANOVA": (flat_genes, None),
        "Hier_Pooled": (stage1_genes, pooled_genes),
        "Hier_Consistent": (stage1_genes, consistent_genes),
    }


def fit_predict(X_train, y_train, X_test, stages):
    stage1, stage2 = stages
    if stage2 is None:
        model = new_classifier()
        model.fit(X_train[stage1], y_train)
        return model.predict(X_test[stage1])

    # Stage 1: control vs treated. Stage 2: acute vs DTP.
    treatment_train = (y_train != "DMSO").astype(int)
    model1 = new_classifier()
    model1.fit(X_train[stage1], treatment_train)

    model2 = new_classifier()
    model2.fit(X_train.loc[treatment_train == 1, stage2], y_train.loc[treatment_train == 1])

    is_treated = model1.predict(X_test[stage1]) == 1
    subtype_predictions = model2.predict(X_test[stage2])
    return np.where(is_treated, subtype_predictions, "DMSO")


def main():
    X_first, y_first, lines, X_second, y_second = load_data()
    print("Internal study:", X_first.shape, "External study:", X_second.shape)
    print("\nInternal class counts:\n", pd.crosstab(lines, y_first))
    print("\nExternal class counts:\n", y_second.value_counts())

    folds = []
    for line in sorted(lines.unique()):
        train_mask = lines != line
        folds.append((
            "Internal_LOCO", line,
            X_first.loc[train_mask], y_first.loc[train_mask], lines.loc[train_mask],
            X_first.loc[~train_mask], y_first.loc[~train_mask],
        ))
    # Stronger transfer setting: PC9 and the entire external study are unseen in training.
    no_pc9 = lines != "PC9"
    folds.append((
        "External_development", "GSE255958_PC9",
        X_first.loc[no_pc9], y_first.loc[no_pc9], lines.loc[no_pc9],
        X_second, y_second,
    ))

    metric_rows, prediction_rows, selection_rows = [], [], []
    for scope, held_out, X_train, y_train, training_lines, X_test, y_test in folds:
        print(f"\n{scope} | held out: {held_out} | train={len(X_train)} test={len(X_test)}")
        for n_genes in GENE_COUNTS:
            gene_sets = select_gene_sets(X_train, y_train, training_lines, n_genes)
            for method, stages in gene_sets.items():
                predicted = fit_predict(X_train, y_train, X_test, stages)
                dtp_recall = recall_score(y_test == "osi_DTP", predicted == "osi_DTP")
                row = {
                    "scope": scope, "held_out": held_out, "n_genes": n_genes,
                    "method": method, "n_test": len(y_test),
                    "accuracy": accuracy_score(y_test, predicted),
                    "macro_f1": f1_score(y_test, predicted, labels=CLASSES,
                                         average="macro", zero_division=0),
                    "dtp_recall": dtp_recall,
                    "dtp_mistaken_as_acute": int(((y_test == "osi_DTP") &
                                                  (predicted == "osi_acute")).sum()),
                }
                metric_rows.append(row)
                print(f"  {n_genes:>2} genes | {method:>16} "
                      f"F1={row['macro_f1']:.3f} DTP-recall={dtp_recall:.3f}")
                for sample, true, pred in zip(X_test.index, y_test, predicted):
                    prediction_rows.append({
                        "scope": scope, "held_out": held_out, "n_genes": n_genes,
                        "method": method, "sample_id": sample,
                        "actual": true, "predicted": pred,
                    })
                for stage_name, gene_list in zip(
                    ["flat"] if stages[1] is None else ["control_vs_treated", "acute_vs_DTP"],
                    [stages[0]] if stages[1] is None else [stages[0], stages[1]],
                ):
                    for rank, gene in enumerate(gene_list, 1):
                        selection_rows.append({
                            "scope": scope, "held_out": held_out, "n_genes": n_genes,
                            "method": method, "stage": stage_name,
                            "rank": rank, "gene": gene,
                        })

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(RESULTS / "three_state_contrast_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        RESULTS / "three_state_contrast_predictions.csv", index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        RESULTS / "three_state_contrast_genes.csv", index=False,
    )

    print("\nInternal LOCO: mean across 4 held-out cell lines")
    print(metric_df.loc[metric_df.scope == "Internal_LOCO"].groupby(
        ["n_genes", "method"])[["accuracy", "macro_f1", "dtp_recall"]]
          .mean().round(3).to_string())
    print("\nExternal development: unseen study AND PC9 cell line")
    print(metric_df.loc[metric_df.scope == "External_development", [
        "n_genes", "method", "accuracy", "macro_f1", "dtp_recall", "dtp_mistaken_as_acute"
    ]].round(3).to_string(index=False))
    print("\nDone. The external dataset has already been examined during development;"
          " do not describe it as an untouched final test set.")

    
    # ==================================================
    # Acute-to-DTP response consistency across studies
    # Diagnostic analysis only
    # ==================================================

    def treatment_contrast(expression, conditions):
        acute = expression.loc[
            conditions == "osi_acute"
        ].mean()

        dtp = expression.loc[
            conditions == "osi_DTP"
        ].mean()

        return dtp - acute

    # PC9 from GSE193258
    pc9_mask = lines == "PC9"

    effect_pc9_first = treatment_contrast(
        X_first.loc[pc9_mask],
        y_first.loc[pc9_mask]
    )

    # PC9 from GSE255958
    effect_pc9_second = treatment_contrast(
        X_second,
        y_second
    )

    print("\nAcute-to-DTP response consistency:")

    pearson = effect_pc9_first.corr(effect_pc9_second)
    spearman = effect_pc9_first.corr(
        effect_pc9_second,
        method="spearman"
    )

    print(f"Pearson correlation: {pearson:.3f}")
    print(f"Spearman correlation: {spearman:.3f}")

    print("\nDirection agreement for top genes:")

    for n in [20, 50, 100]:

        genes = (
            effect_pc9_first.abs()
            .nlargest(n)
            .index
        )

        first_direction = np.sign(
            effect_pc9_first.loc[genes]
        )

        second_direction = np.sign(
            effect_pc9_second.loc[genes]
        )

        agreement = (
            first_direction == second_direction
        ).mean()

        print(
            f"Top {n} genes: {agreement:.3f}"
        )

    # Compare against other cell lines in study 1
    print("\nWithin-study PC9 contrast comparison:")

    for cell_line in sorted(lines.unique()):

        if cell_line == "PC9":
            continue

        mask = lines == cell_line

        other_effect = treatment_contrast(
            X_first.loc[mask],
            y_first.loc[mask]
        )

        correlation = effect_pc9_first.corr(
            other_effect,
            method="spearman"
        )

        print(
            f"PC9 vs {cell_line}: "
            f"Spearman={correlation:.3f}"
        )
    
    # ==============================================
    # Raw log2TPM contrast consistency
    # ==============================================

    raw_first = pd.read_csv(
        RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
        sep="\t",
        index_col=0
    ).T

    raw_second = pd.read_csv(
        PROCESSED / "GSE255958_PC9_expression.csv.gz",
        index_col=0
    )

    pc9_ids = X_first.index[lines == "PC9"]

    first_pc9 = raw_first.loc[pc9_ids]
    second_pc9 = raw_second.loc[X_second.index]

    shared = sorted(
        set(first_pc9.columns) & set(second_pc9.columns)
    )

    def raw_contrast(expression, labels):
        return (
            expression.loc[labels == "osi_DTP"].mean()
            - expression.loc[labels == "osi_acute"].mean()
        )

    first_effect = raw_contrast(
        first_pc9[shared],
        y_first.loc[pc9_ids]
    )

    second_effect = raw_contrast(
        second_pc9[shared],
        y_second
    )

    print("\nOriginal log2TPM contrast comparison:")
    print("Shared genes:", len(shared))
    print(
        "Pearson:",
        round(first_effect.corr(second_effect), 3)
    )
    print(
        "Spearman:",
        round(
            first_effect.corr(
                second_effect,
                method="spearman"
            ), 3
        )
    )

    top50 = first_effect.abs().nlargest(50).index

    agreement = (
        np.sign(first_effect.loc[top50])
        == np.sign(second_effect.loc[top50])
    ).mean()

    print(
        "Top 50 direction agreement:",
        round(agreement, 3)
    )




if __name__ == "__main__":
    main()
