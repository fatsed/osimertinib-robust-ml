
from pathlib import Path

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    StandardScaler
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Paths
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# Load study 1
first = pd.read_csv(
    raw_dir / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    sep="\t",
    index_col=0
).T

sample_names = first.index.to_series()

cell_lines = sample_names.str.split("_").str[0]

conditions = (
    sample_names.str.split("_", n=1).str[1]
    .str.rsplit("_", n=1).str[0]
)


# Binary development task
labels = ["DMSO", "osi_DTP"]

# Exclude PC9 entirely from training
train_mask = (
    conditions.isin(labels)
    & (cell_lines != "PC9")
)

X_train = first.loc[train_mask]
y_train = conditions.loc[train_mask]


# Load study 2
second = pd.read_csv(
    processed_dir / "GSE255958_PC9_expression.csv.gz",
    index_col=0
)

metadata = pd.read_csv(
    processed_dir / "GSE255958_PC9_metadata.csv"
).set_index("sample_id")

test_ids = metadata.index[
    metadata["condition"].isin(labels)
]

X_test = second.loc[test_ids]
y_test = metadata.loc[test_ids, "condition"]


# Match gene features
shared_genes = sorted(
    set(X_train.columns) & set(X_test.columns)
)

X_train = X_train.loc[:, shared_genes]
X_test = X_test.loc[:, shared_genes]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Shared genes:", len(shared_genes))


# Sample-wise percentile rank transformation
def sample_percentile_ranks(X):
    data = pd.DataFrame(X)

    return data.rank(
        axis=1,
        method="average",
        pct=True
    ).to_numpy(dtype=float)


# Compare two approaches
models = {
    "Raw Expression": make_pipeline(
        VarianceThreshold(),
        StandardScaler(),
        PCA(
            n_components=5,
            svd_solver="randomized",
            random_state=42
        ),
        LogisticRegression(max_iter=2000)
    ),

    "Sample Rank": make_pipeline(
        FunctionTransformer(
            sample_percentile_ranks,
            validate=False
        ),
        VarianceThreshold(),
        StandardScaler(),
        PCA(
            n_components=5,
            svd_solver="randomized",
            random_state=42
        ),
        LogisticRegression(max_iter=2000)
    )
}

results = []
all_predictions = []


# Development evaluation
for model_name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    results.append({
        "method": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    for sample_id, actual, predicted in zip(
        X_test.index, y_test, y_pred
    ):
        all_predictions.append({
            "method": model_name,
            "sample_id": sample_id,
            "actual": actual,
            "predicted": predicted
        })

    print(f"\nMethod: {model_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    print(pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ))


# Save development results
results_df = pd.DataFrame(results)

results_df.to_csv(
    results_dir / "development_rank_comparison.csv",
    index=False
)

pd.DataFrame(all_predictions).to_csv(
    results_dir / "development_rank_predictions.csv",
    index=False
)

print("\nDevelopment comparison completed.")


# Internal leave-one-cell-line-out evaluation

from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut

# Use only control and DTP samples from study 1
internal_mask = conditions.isin(labels)

X_internal = first.loc[internal_mask, shared_genes]
y_internal = conditions.loc[internal_mask]
groups_internal = cell_lines.loc[internal_mask]

logo = LeaveOneGroupOut()
internal_results = []

print("\nInternal cross-cell-line evaluation:")
print("Samples:", len(X_internal))
print("Cell lines:", groups_internal.nunique())

for method_name, candidate in models.items():

    for train_idx, test_idx in logo.split(
        X_internal,
        y_internal,
        groups=groups_internal
    ):

        fold_model = clone(candidate)

        fold_model.fit(
            X_internal.iloc[train_idx],
            y_internal.iloc[train_idx]
        )

        predictions = fold_model.predict(
            X_internal.iloc[test_idx]
        )

        test_cell_line = groups_internal.iloc[
            test_idx
        ].iloc[0]

        accuracy = accuracy_score(
            y_internal.iloc[test_idx],
            predictions
        )

        macro_f1 = f1_score(
            y_internal.iloc[test_idx],
            predictions,
            labels=labels,
            average="macro",
            zero_division=0
        )

        internal_results.append({
            "method": method_name,
            "test_cell_line": test_cell_line,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

internal_df = pd.DataFrame(internal_results)

print("\nResults by cell line:")
print(internal_df.to_string(index=False))

print("\nMean performance:")
print(
    internal_df.groupby("method")[
        ["accuracy", "macro_f1"]
    ].mean().round(3)
)

internal_df.to_csv(
    results_dir / "internal_binary_logo_metrics.csv",
    index=False
)