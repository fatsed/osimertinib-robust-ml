from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

# Project paths
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

classes = ["DMSO", "osi_acute", "osi_DTP"]
gene_counts = [10, 20, 50]

# Load expression data
X = pd.read_csv(
    raw_dir / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    sep="\t",
    index_col=0
).T

names = X.index.to_series()

cell_lines = names.str.split("_").str[0]

conditions = (
    names.str.split("_", n=1).str[1]
    .str.rsplit("_", n=1).str[0]
)

# Keep the three treatment states
selected = conditions.isin(classes)

X = X.loc[selected]
y = conditions.loc[selected]
cell_lines = cell_lines.loc[selected]

# Sample-wise rank transformation
X = X.rank(axis=1, method="average", pct=True)

print("Input shape:", X.shape)
print("\nSamples per cell line and condition:")
print(pd.crosstab(cell_lines, y))

results = []
predictions_rows = []


# Leave-One-Cell-Line-Out
for test_cell_line in sorted(cell_lines.unique()):

    train_mask = cell_lines != test_cell_line
    test_mask = ~train_mask

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print(f"\nHeld-out cell line: {test_cell_line}")

    for n_genes in gene_counts:

        # Feature selection is fitted on training data
        model = make_pipeline(
            VarianceThreshold(),
            SelectKBest(
                score_func=f_classif,
                k=n_genes
            ),
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted)

        macro_f1 = f1_score(
            y_test,
            predicted,
            labels=classes,
            average="macro",
            zero_division=0
        )

        print(
            f"{n_genes} genes | "
            f"Accuracy={accuracy:.3f} | "
            f"Macro F1={macro_f1:.3f}"
        )

        results.append({
            "test_cell_line": test_cell_line,
            "n_genes": n_genes,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        for sample, actual, prediction in zip(
            X_test.index, y_test, predicted
        ):
            predictions_rows.append({
                "test_cell_line": test_cell_line,
                "n_genes": n_genes,
                "sample_id": sample,
                "actual": actual,
                "predicted": prediction
            })

        if n_genes == 50:
            matrix = confusion_matrix(
                y_test,
                predicted,
                labels=classes
            )

            print("Confusion matrix (50 genes):")
            print(
                pd.DataFrame(
                    matrix,
                    index=classes,
                    columns=classes
                )
            )


# Save results
metrics = pd.DataFrame(results)
predictions = pd.DataFrame(predictions_rows)

metrics.to_csv(
    results_dir / "three_state_loco_metrics.csv",
    index=False
)

predictions.to_csv(
    results_dir / "three_state_loco_predictions.csv",
    index=False
)

print("\nMean performance by gene count:")
print(
    metrics.groupby("n_genes")[
        ["accuracy", "macro_f1"]
    ].mean().round(3)
)

print("\nThree-state LOCO evaluation completed.")
