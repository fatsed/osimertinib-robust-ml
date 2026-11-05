from pathlib import Path
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

# 1. Project paths and settings

project_dir = Path(__file__).resolve().parents[1]

raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

labels = ["DMSO", "osi_DTP"]
gene_counts = [5, 10, 20, 50, 100]

# 2. Load dataset 1: GSE193258


first = pd.read_csv(
    raw_dir / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    sep="\t",
    index_col=0
).T

sample_names = first.index.to_series()

first_meta = pd.DataFrame({
    "study": "GSE193258",
    "cell_line": sample_names.str.split("_").str[0],
    "condition": (
        sample_names.str.split("_", n=1).str[1]
        .str.rsplit("_", n=1).str[0]
    )
}, index=first.index)

first_meta = first_meta.loc[
    first_meta["condition"].isin(labels)
]

first = first.loc[first_meta.index]


# 3. Load processed datasets

def load_study(expression_file, metadata_file):

    expression = pd.read_csv(
        processed_dir / expression_file,
        index_col=0
    )

    metadata = pd.read_csv(
        processed_dir / metadata_file
    ).set_index("sample_id")

    metadata = metadata.loc[
        metadata["condition"].isin(labels)
    ]

    expression = expression.loc[metadata.index]

    return expression, metadata


study_data = [
    (first, first_meta),

    load_study(
        "GSE255958_PC9_expression.csv.gz",
        "GSE255958_PC9_metadata.csv"
    ),

    load_study(
        "GSE249721_HCC827_original_values.csv.gz",
        "GSE249721_HCC827_metadata.csv"
    )
]


# 4. Align common genes and sample identifiers

common_genes = sorted(
    set.intersection(
        *(set(X.columns) for X, _ in study_data)
    )
)

expression_parts = []
metadata_parts = []

for X_part, meta_part in study_data:

    X_part = X_part.loc[:, common_genes].copy()
    meta_part = meta_part.copy()

    unique_ids = [
        f"{study}::{sample}"
        for study, sample in zip(
            meta_part["study"],
            meta_part.index
        )
    ]

    X_part.index = unique_ids
    meta_part.index = unique_ids

    expression_parts.append(X_part)
    metadata_parts.append(meta_part)


X = pd.concat(expression_parts)
meta = pd.concat(metadata_parts)

if not X.index.equals(meta.index):
    raise ValueError("Sample alignment error.")

if X.isna().any().any():
    raise ValueError("Missing expression values.")

if len(X) != 36:
    raise ValueError(
        f"Expected 36 samples, found {len(X)}"
    )

print("Total samples:", len(X))
print("Common genes:", len(common_genes))

print("\nSamples per study:")
print(pd.crosstab(
    meta["study"],
    meta["condition"]
))

# 5. Sample-wise percentile rank transformation

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True
)

# 6. Leave-One-Study-Out evaluation

results = []
prediction_rows = []

for test_study in sorted(meta["study"].unique()):

    print(f"\nTesting study: {test_study}")

    train_mask = meta["study"] != test_study
    test_mask = ~train_mask

    X_train = X_rank.loc[train_mask]
    X_test = X_rank.loc[test_mask]

    train_meta = meta.loc[train_mask]

    y_train = train_meta["condition"]
    y_test = meta.loc[test_mask, "condition"]

    # Calculate treatment effects per study/cell line

    group_effects = {}

    for (study, cell_line), group in train_meta.groupby(
        ["study", "cell_line"]
    ):

        group_labels = y_train.loc[group.index]

        control_ids = group_labels.index[
            group_labels == "DMSO"
        ]

        dtp_ids = group_labels.index[
            group_labels == "osi_DTP"
        ]

        if len(control_ids) == 0 or len(dtp_ids) == 0:
            raise ValueError(
                f"Missing class in {study}, {cell_line}"
            )

        group_effects[f"{study}|{cell_line}"] = (
            X_train.loc[dtp_ids].mean()
            - X_train.loc[control_ids].mean()
        )

    effect_df = pd.DataFrame(group_effects)

    
    # Method 1: Pooled Effect

    pooled_effect = (
        X_train.loc[y_train == "osi_DTP"].mean()
        - X_train.loc[y_train == "DMSO"].mean()
    )

    # Method 2: Study-Balanced Effect

    study_effects = {}

    for study in sorted(train_meta["study"].unique()):

        columns = [
            column
            for column in effect_df.columns
            if column.startswith(f"{study}|")
        ]

        study_effects[study] = (
            effect_df[columns].mean(axis=1)
        )

    study_effect_df = pd.DataFrame(study_effects)

    balanced_effect = study_effect_df.mean(axis=1)

    # Method 3: Balanced + Heterogeneity Penalty

    heterogeneity = study_effect_df.std(
        axis=1,
        ddof=0
    )

    penalized_scores = (
        balanced_effect.abs() - heterogeneity
    )

    # Evaluate different gene counts

    for n_genes in gene_counts:

        pooled_genes = (
            pooled_effect.abs()
            .nlargest(n_genes)
            .index.tolist()
        )

        balanced_genes = (
            balanced_effect.abs()
            .nlargest(n_genes)
            .index.tolist()
        )

        penalized_genes = (
            penalized_scores
            .nlargest(n_genes)
            .index.tolist()
        )

        gene_sets = {
            "Pooled Effect": pooled_genes,
            "Study Balanced": balanced_genes,
            "Balanced + Penalty": penalized_genes
        }

        for method_name, selected_genes in gene_sets.items():

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )

            model.fit(
                X_train[selected_genes],
                y_train
            )

            predictions = model.predict(
                X_test[selected_genes]
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            macro_f1 = f1_score(
                y_test,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0
            )

            dtp_recall = recall_score(
                y_test,
                predictions,
                pos_label="osi_DTP",
                zero_division=0
            )

            results.append({
                "test_study": test_study,
                "method": method_name,
                "n_genes": n_genes,
                "n_test": len(y_test),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "dtp_recall": dtp_recall
            })

            for sample_id, actual, predicted in zip(
                X_test.index,
                y_test,
                predictions
            ):

                prediction_rows.append({
                    "test_study": test_study,
                    "method": method_name,
                    "n_genes": n_genes,
                    "sample_id": sample_id,
                    "actual": actual,
                    "predicted": predicted
                })


# 7. Save results
# 

results_df = pd.DataFrame(results)
predictions_df = pd.DataFrame(prediction_rows)

results_df.to_csv(
    results_dir / "gene_count_sensitivity_metrics.csv",
    index=False
)

predictions_df.to_csv(
    results_dir / "gene_count_sensitivity_predictions.csv",
    index=False
)

# Save separate results for each gene count
for n_genes in gene_counts:

    subset = results_df[
        results_df["n_genes"] == n_genes
    ]

    subset.to_csv(
        results_dir
        / f"study_balanced_metrics_{n_genes}.csv",
        index=False
    )


# 8. Summary

summary = (
    results_df.groupby(["n_genes", "method"])[
        ["accuracy", "macro_f1", "dtp_recall"]
    ]
    .mean()
    .round(3)
)

summary.to_csv(
    results_dir / "gene_count_sensitivity_summary.csv"
)

print("\nGene count sensitivity results:")
print(summary.to_string())

print("\nGene count sensitivity analysis completed.")
