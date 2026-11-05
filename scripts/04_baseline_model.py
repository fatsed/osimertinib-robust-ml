import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestCentroid

# Load dataset
project_dir = Path(__file__).resolve().parents[1]

data_path = (
    project_dir
    / "data"
    / "raw"
    / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
)

df = pd.read_csv(data_path, sep="\t", index_col=0)
X = df.T

# Extract sample information
sample_names = X.index.to_series()

cell_lines = sample_names.str.split("_").str[0]

conditions = (
    sample_names.str.split("_", n=1).str[1]
    .str.rsplit("_", n=1).str[0]
)

# Select three experimental conditions
selected_conditions = ["DMSO", "osi_acute", "osi_DTP"]

mask = conditions.isin(selected_conditions)

X = X.loc[mask]
y = conditions.loc[mask]
groups = cell_lines.loc[mask]

# Display selected dataset
print("Input shape:", X.shape)

print("\nSamples per cell line and condition:")
print(pd.crosstab(groups, y))


# Leave-One-Cell-Line-Out cross-validation

logo = LeaveOneGroupOut()

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups=groups),
    start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    test_cell_line = groups.iloc[test_idx].unique()[0]

    print(f"\nFold {fold}")
    print("Test cell line:", test_cell_line)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Training classes:", y_train.value_counts().to_dict())
    print("Testing classes:", y_test.value_counts().to_dict())

    
    # Build the baseline model
    model = make_pipeline(
        VarianceThreshold(),
        StandardScaler(),
        PCA(n_components=5, random_state=42),
        LogisticRegression(max_iter=2000)
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict unseen samples
    y_pred = model.predict(X_test)

    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy, 3))
    print("Macro F1:", round(macro_f1, 3))

    
    # Confusion matrix
    labels = ["DMSO", "osi_acute", "osi_DTP"]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    # Identify misclassified samples
    predictions = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred
        },
        index=X_test.index
    )

    errors = predictions[
        predictions["Actual"] != predictions["Predicted"]
    ]

    print("\nMisclassified samples:")
    if errors.empty:
        print("No misclassified samples.")
    else:
        print(errors)

    
    # Save predictions for reproducibility
    results_dir = project_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        results_dir
        / f"baseline_predictions_{test_cell_line}.csv"
    )

    predictions.to_csv(
        output_path,
        index_label="sample_id"
    )

    print("Predictions saved:", output_path)

    
    # Compare additional baseline models
    comparison_models = {
        "Linear SVM": LinearSVC(
            C=1.0,
            max_iter=10000,
            random_state=42
        ),
        "Nearest Centroid": NearestCentroid()
    }

    print("\nAdditional baseline comparison:")

    for model_name, classifier in comparison_models.items():
        comparison_pipeline = make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            PCA(n_components=5, random_state=42),
            classifier
        )

        comparison_pipeline.fit(X_train, y_train)

        comparison_pred = comparison_pipeline.predict(X_test)

        comparison_accuracy = accuracy_score(
            y_test, comparison_pred
        )

        comparison_f1 = f1_score(
            y_test,
            comparison_pred,
            average="macro",
            zero_division=0
        )

        print(
            f"{model_name}: "
            f"Accuracy={comparison_accuracy:.3f}, "
            f"Macro F1={comparison_f1:.3f}"
        )




