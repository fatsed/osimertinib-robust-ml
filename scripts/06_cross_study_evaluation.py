from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

# Project paths
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

# Load training study
train_path = (
    raw_dir / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
)

train_df = pd.read_csv(
    train_path, sep="\t", index_col=0
).T

train_conditions = (
    train_df.index.to_series()
    .str.split("_", n=1).str[1]
    .str.rsplit("_", n=1).str[0]
)

labels = ["DMSO", "osi_acute", "osi_DTP"]
train_mask = train_conditions.isin(labels)

X_train = train_df.loc[train_mask]
y_train = train_conditions.loc[train_mask]


# Load independent test study
X_test = pd.read_csv(
    processed_dir / "GSE255958_PC9_expression.csv.gz",
    index_col=0
)

metadata = pd.read_csv(
    processed_dir / "GSE255958_PC9_metadata.csv"
).set_index("sample_id")

X_test = X_test.loc[metadata.index]
y_test = metadata["condition"]

if set(y_test.unique()) != set(labels):
    raise ValueError("Unexpected test labels.")


# Match genes in identical order
shared_genes = sorted(
    set(X_train.columns) & set(X_test.columns)
)

X_train = X_train.loc[:, shared_genes]
X_test = X_test.loc[:, shared_genes]

assert X_train.shape == (36, len(shared_genes))
assert X_test.shape == (9, len(shared_genes))

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("Shared genes:", len(shared_genes))


# Define baseline classifiers
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),
    "Linear SVM": LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=42
    ),
    "Nearest Centroid": NearestCentroid()
}

metrics_rows = []
prediction_rows = []


# Train on study 1 and test on study 2
for model_name, classifier in models.items():
    model = make_pipeline(
        VarianceThreshold(),
        StandardScaler(),
        PCA(n_components=5, random_state=42),
        classifier
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    metrics_rows.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    for sample_id, actual, predicted in zip(
        X_test.index, y_test, y_pred
    ):
        prediction_rows.append({
            "model": model_name,
            "sample_id": sample_id,
            "actual": actual,
            "predicted": predicted
        })

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    cm = confusion_matrix(
        y_test, y_pred, labels=labels
    )

    print("Confusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ))


# Save results
metrics_df = pd.DataFrame(metrics_rows)
predictions_df = pd.DataFrame(prediction_rows)

metrics_df.to_csv(
    results_dir / "cross_study_metrics.csv",
    index=False
)

predictions_df.to_csv(
    results_dir / "cross_study_predictions.csv",
    index=False
)

print("\nSummary:")
print(metrics_df.to_string(index=False))
print("\nCross-study evaluation completed.")



# Cross-study evaluation with unseen cell line

from sklearn.base import clone

# Exclude PC9 from training
train_cell_lines = (
    X_train.index.to_series().str.split("_").str[0]
)

mask_no_pc9 = train_cell_lines != "PC9"

X_train_unseen = X_train.loc[mask_no_pc9]
y_train_unseen = y_train.loc[mask_no_pc9]

print("\nEvaluation without PC9 in training:")
print("Training shape:", X_train_unseen.shape)
print("Testing shape:", X_test.shape)

unseen_results = []

for model_name, classifier in models.items():

    model = make_pipeline(
        VarianceThreshold(),
        StandardScaler(),
        PCA(n_components=5, random_state=42),
        clone(classifier)
    )

    model.fit(X_train_unseen, y_train_unseen)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    unseen_results.append({
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    })

    print(f"\n{model_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")

    cm = confusion_matrix(
        y_test, predictions, labels=labels
    )

    print(pd.DataFrame(
        cm, index=labels, columns=labels
    ))

unseen_df = pd.DataFrame(unseen_results)

unseen_df.to_csv(
    results_dir / "cross_study_unseen_cell_line_metrics.csv",
    index=False
)

print("\nUnseen-cell-line evaluation completed.")

# Cross-study DTP response consistency in PC9

# Identify PC9 samples in study 1
first_cell_lines = (
    train_df.index.to_series().str.split("_").str[0]
)

pc9_control_1 = train_df.loc[
    (first_cell_lines == "PC9")
    & (train_conditions == "DMSO"),
    shared_genes
]

pc9_dtp_1 = train_df.loc[
    (first_cell_lines == "PC9")
    & (train_conditions == "osi_DTP"),
    shared_genes
]

# Identify corresponding samples in study 2
pc9_control_2 = X_test.loc[
    y_test == "DMSO", shared_genes
]

pc9_dtp_2 = X_test.loc[
    y_test == "osi_DTP", shared_genes
]

# Verify sample counts
assert len(pc9_control_1) == 3
assert len(pc9_dtp_1) == 3
assert len(pc9_control_2) == 3
assert len(pc9_dtp_2) == 3

# Compute DTP - Control expression changes
effect_1 = pc9_dtp_1.mean() - pc9_control_1.mean()
effect_2 = pc9_dtp_2.mean() - pc9_control_2.mean()

# Compare treatment-response patterns
pearson = effect_1.corr(effect_2, method="pearson")
spearman = effect_1.corr(effect_2, method="spearman")

print("\nPC9 DTP response consistency:")
print("Shared genes:", len(shared_genes))
print("Pearson correlation:", round(pearson, 3))
print("Spearman correlation:", round(spearman, 3))

# Save gene-level effects
effect_comparison = pd.DataFrame({
    "GSE193258_DTP_change": effect_1,
    "GSE255958_DTP_change": effect_2
})

effect_comparison.to_csv(
    results_dir / "pc9_dtp_response_consistency.csv",
    index_label="gene_symbol"
)

print("DTP response comparison saved.")
