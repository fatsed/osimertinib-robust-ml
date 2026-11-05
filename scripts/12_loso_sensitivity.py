from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Paths and settings
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

labels = ["DMSO", "osi_DTP"]
n_genes = 50


# Load first study
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


# Load processed studies
def load_study(study, expression_file, metadata_file):
    X = pd.read_csv(
        processed_dir / expression_file,
        index_col=0
    )

    meta = pd.read_csv(
        processed_dir / metadata_file
    ).set_index("sample_id")

    meta["condition"] = meta["condition"].replace({
        "control": "DMSO"
    })

    meta = meta.loc[meta["condition"].isin(labels)]
    X = X.loc[meta.index]

    return X, meta


study_data = [
    (first, first_meta),

    load_study(
        "GSE255958",
        "GSE255958_PC9_expression.csv.gz",
        "GSE255958_PC9_metadata.csv"
    ),

    load_study(
        "GSE249721",
        "GSE249721_HCC827_original_values.csv.gz",
        "GSE249721_HCC827_metadata.csv"
    ),

    load_study(
        "GSE285632",
        "GSE285632_expression.csv.gz",
        "GSE285632_metadata.csv"
    )
]


# Sensitivity analysis: exclude study 4
study_data = [
    (expression, metadata)
    for expression, metadata in study_data
    if metadata["study"].iloc[0] != "GSE285632"
]

print("\nSensitivity analysis: three studies only")


# Align common genes and unique sample IDs
common_genes = sorted(
    set.intersection(
        *(set(X.columns) for X, _ in study_data)
    )
)

all_X = []
all_meta = []

for X, meta in study_data:
    X = X.loc[:, common_genes].copy()
    meta = meta.copy()

    unique_ids = [
        f"{study}::{sample}"
        for study, sample in zip(
            meta["study"], meta.index
        )
    ]

    X.index = unique_ids
    meta.index = unique_ids

    all_X.append(X)
    all_meta.append(meta)

X = pd.concat(all_X)
meta = pd.concat(all_meta)

if X.isna().any().any():
    raise ValueError("Missing expression values.")

print("Total samples:", len(X))
print("Common genes:", len(common_genes))
print("\nSamples per study:")
print(pd.crosstab(meta["study"], meta["condition"]))


# Leave-One-Study-Out evaluation
results = []
prediction_rows = []

for test_study in sorted(meta["study"].unique()):

    train_mask = meta["study"] != test_study
    test_mask = ~train_mask

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = meta.loc[train_mask, "condition"]
    y_test = meta.loc[test_mask, "condition"]

    # Rank genes separately within every sample
    train_ranks = X_train.rank(
        axis=1, method="average", pct=True
    )

    test_ranks = X_test.rank(
        axis=1, method="average", pct=True
    )

    # Group by study and cell line
    train_meta = meta.loc[train_mask]
    groups = (
        train_meta["study"]
        + "|"
        + train_meta["cell_line"]
    )

    effects = {}

    for group in sorted(groups.unique()):
        group_mask = groups == group

        control = train_ranks.loc[
            group_mask & (y_train == "DMSO")
        ]

        dtp = train_ranks.loc[
            group_mask & (y_train == "osi_DTP")
        ]

        if control.empty or dtp.empty:
            raise ValueError(
                f"Missing condition in group: {group}"
            )

        effects[group] = dtp.mean() - control.mean()

    effect_df = pd.DataFrame(effects)

    # Stable gene selection: training only
    same_direction = (
        (effect_df > 0).all(axis=1)
        | (effect_df < 0).all(axis=1)
    )

    stable_scores = (
        effect_df.abs().min(axis=1)
        .loc[same_direction]
    )

    if len(stable_scores) < n_genes:
        raise ValueError(
            f"Only {len(stable_scores)} stable genes "
            f"when testing {test_study}"
        )

    stable_genes = (
        stable_scores.nlargest(n_genes).index.tolist()
    )

    # Pooled-effect selection: training only
    pooled_effect = (
        train_ranks.loc[y_train == "osi_DTP"].mean()
        - train_ranks.loc[y_train == "DMSO"].mean()
    )

    pooled_genes = (
        pooled_effect.abs().nlargest(n_genes).index.tolist()
    )

    gene_sets = {
        "Rank + PCA": common_genes,
        "Pooled Effect": pooled_genes,
        "Stable Signature": stable_genes
    }

    for method_name, selected in gene_sets.items():

        steps = [StandardScaler()]

        if method_name == "Rank + PCA":
            steps.append(
                PCA(n_components=5, random_state=42)
            )

        steps.append(
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

        model = make_pipeline(*steps)

        model.fit(
            train_ranks[selected],
            y_train
        )

        predictions = model.predict(
            test_ranks[selected]
        )

        results.append({
            "test_study": test_study,
            "method": method_name,
            "n_test": len(y_test),
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test, predictions,
                labels=labels,
                average="macro",
                zero_division=0
            ),
            "dtp_recall": recall_score(
                y_test, predictions,
                pos_label="osi_DTP",
                zero_division=0
            )
        })

        for sample_id, actual, predicted in zip(
            X_test.index, y_test, predictions
        ):
            prediction_rows.append({
                "test_study": test_study,
                "method": method_name,
                "sample_id": sample_id,
                "actual": actual,
                "predicted": predicted
            })


# Save and summarize
results_df = pd.DataFrame(results)

results_df.to_csv(
    results_dir / "loso_three_studies_metrics.csv",
    index=False
)

pd.DataFrame(prediction_rows).to_csv(
    results_dir / "loso_three_studies_predictions.csv",
    index=False
)

print("\nLeave-One-Study-Out results:")
print(results_df.to_string(index=False))

print("\nMean performance across studies:")
print(
    results_df.groupby("method")[
        ["accuracy", "macro_f1", "dtp_recall"]
    ].mean().round(3)
)

print("\nLOSO evaluation completed.")
