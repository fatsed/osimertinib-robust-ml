from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

# Project paths
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

labels = ["DMSO", "osi_DTP"]
n_genes = 50


# Load training study
first = pd.read_csv(
    raw_dir / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    sep="\t",
    index_col=0
).T

names = first.index.to_series()

cell_lines = names.str.split("_").str[0]

conditions = (
    names.str.split("_", n=1).str[1]
    .str.rsplit("_", n=1).str[0]
)

# Exclude HCC827 from training
train_mask = (
    conditions.isin(labels)
    & (cell_lines != "HCC827")
)

X_train = first.loc[train_mask]
y_train = conditions.loc[train_mask]
groups_train = cell_lines.loc[train_mask]

assert len(X_train) == 18
assert "HCC827" not in groups_train.unique()


# Load untouched external study
X_test = pd.read_csv(
    processed_dir / "GSE249721_HCC827_original_values.csv.gz",
    index_col=0
)

metadata = pd.read_csv(
    processed_dir / "GSE249721_HCC827_metadata.csv"
).set_index("sample_id")

X_test = X_test.loc[metadata.index]
y_test = metadata.loc[X_test.index, "condition"]

assert len(X_test) == 6
assert y_test.value_counts().reindex(
    labels, fill_value=0
).tolist() == [3, 3]


# Align genes before ranking
common_genes = sorted(
    set(X_train.columns) & set(X_test.columns)
)

X_train = X_train[common_genes]
X_test = X_test[common_genes]

assert X_train.columns.equals(X_test.columns)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Shared genes:", len(common_genes))


# Convert each sample to gene percentile ranks
train_ranks = X_train.rank(
    axis=1, method="average", pct=True
)

test_ranks = X_test.rank(
    axis=1, method="average", pct=True
)

# Calculate treatment effects per training cell line
effects = {}

for line in sorted(groups_train.unique()):
    control = train_ranks.loc[
        (groups_train == line)
        & (y_train == "DMSO")
    ]

    dtp = train_ranks.loc[
        (groups_train == line)
        & (y_train == "osi_DTP")
    ]

    effects[line] = dtp.mean() - control.mean()

effect_df = pd.DataFrame(effects)


# Select stable genes using training data only
same_direction = (
    (effect_df > 0).all(axis=1)
    | (effect_df < 0).all(axis=1)
)

stable_scores = effect_df.abs().min(axis=1)
stable_scores = stable_scores.loc[same_direction]

if len(stable_scores) < n_genes:
    raise ValueError("Insufficient stable genes.")

stable_genes = stable_scores.nlargest(n_genes).index.tolist()


# Comparison: largest pooled treatment effects
pooled_effect = (
    train_ranks.loc[y_train == "osi_DTP"].mean()
    - train_ranks.loc[y_train == "DMSO"].mean()
)

pooled_genes = (
    pooled_effect.abs().nlargest(n_genes).index.tolist()
)


# Define fixed evaluation methods
gene_sets = {
    "Rank + PCA": common_genes,
    "Pooled Effect": pooled_genes,
    "Stable Signature": stable_genes
}

metrics = []
prediction_rows = []

for method_name, selected_genes in gene_sets.items():

    if method_name == "Rank + PCA":
        model = make_pipeline(
            StandardScaler(),
            PCA(
                n_components=5,
                svd_solver="randomized",
                random_state=42
            ),
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

    model.fit(
        train_ranks[selected_genes],
        y_train
    )

    predictions = model.predict(
        test_ranks[selected_genes]
    )

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0
    )

    metrics.append({
        "method": method_name,
        "n_genes": len(selected_genes),
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    for sample_id, actual, predicted in zip(
        X_test.index, y_test, predictions
    ):
        prediction_rows.append({
            "method": method_name,
            "sample_id": sample_id,
            "actual": actual,
            "predicted": predicted
        })

    print(f"\nMethod: {method_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    print(pd.DataFrame(
        confusion_matrix(
            y_test, predictions, labels=labels
        ),
        index=labels,
        columns=labels
    ))


# Save evaluation results
metrics_df = pd.DataFrame(metrics)

metrics_df.to_csv(
    results_dir / "external_validation_metrics.csv",
    index=False
)

pd.DataFrame(prediction_rows).to_csv(
    results_dir / "external_validation_predictions.csv",
    index=False
)

pd.DataFrame({
    "gene_symbol": stable_genes,
    "stability_score": stable_scores.loc[
        stable_genes
    ].to_numpy()
}).to_csv(
    results_dir / "external_validation_stable_genes.csv",
    index=False
)

print("\nExternal validation summary:")
print(metrics_df.to_string(index=False))
print("\nEvaluation completed.")
