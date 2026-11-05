
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# Paths and settings
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)

labels = ["DMSO", "osi_DTP"]
n_genes = 50
n_permutations = 100


# Load first study
first = pd.read_csv(
    raw_dir / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    sep="\t",
    index_col=0
).T

names = first.index.to_series()

first_meta = pd.DataFrame({
    "study": "GSE193258",
    "cell_line": names.str.split("_").str[0],
    "condition": (
        names.str.split("_", n=1).str[1]
        .str.rsplit("_", n=1).str[0]
    )
}, index=first.index)

first_meta = first_meta.loc[
    first_meta["condition"].isin(labels)
]

first = first.loc[first_meta.index]


# Load other eligible studies
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

    return expression.loc[metadata.index], metadata


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


# Align genes and sample IDs
common_genes = sorted(
    set.intersection(
        *(set(x.columns) for x, _ in study_data)
    )
)

expression_parts = []
metadata_parts = []

for expression, metadata in study_data:
    expression = expression[common_genes].copy()
    metadata = metadata.copy()

    ids = [
        f"{study}::{sample}"
        for study, sample in zip(
            metadata["study"], metadata.index
        )
    ]

    expression.index = ids
    metadata.index = ids

    expression_parts.append(expression)
    metadata_parts.append(metadata)

X = pd.concat(expression_parts)
meta = pd.concat(metadata_parts)

assert len(X) == 36
assert X.index.equals(meta.index)
assert not X.isna().any().any()

# Sample-wise transformation needs no fitted parameters
X_rank = X.rank(
    axis=1, method="average", pct=True
)

print("Samples:", len(X_rank))
print("Genes:", len(common_genes))
print("Studies:", meta["study"].nunique())


# Evaluate pooled effect with optional permutation
def evaluate(seed=None):
    rng = np.random.default_rng(seed)
    fold_scores = []

    for test_study in sorted(meta["study"].unique()):
        train_mask = meta["study"] != test_study

        X_train = X_rank.loc[train_mask]
        X_test = X_rank.loc[~train_mask]

        y_train = meta.loc[
            train_mask, "condition"
        ].copy()

        y_test = meta.loc[
            ~train_mask, "condition"
        ]

        # Permute training labels within study/cell line
        if seed is not None:
            train_meta = meta.loc[train_mask]

            grouped = train_meta.groupby(
                ["study", "cell_line"]
            )

            for _, group_ids in grouped.groups.items():
                group_ids = list(group_ids)

                y_train.loc[group_ids] = rng.permutation(
                    y_train.loc[group_ids].to_numpy()
                )

        # Select genes using training labels only
        effect = (
            X_train.loc[y_train == "osi_DTP"].mean()
            - X_train.loc[y_train == "DMSO"].mean()
        )

        selected_genes = (
            effect.abs().nlargest(n_genes).index
        )

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

        score = f1_score(
            y_test,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0
        )

        fold_scores.append(score)

    return float(np.mean(fold_scores))


# Original-label performance
observed_score = evaluate()

print("\nOriginal labels:")
print("Mean Macro F1:", round(observed_score, 3))


# Repeated label permutations
null_scores = []

for seed in range(n_permutations):
    score = evaluate(seed=seed)
    null_scores.append(score)

null_scores = np.array(null_scores)

p_value = (
    1 + np.sum(null_scores >= observed_score - 1e-12)
) / (n_permutations + 1)


# Save results
pd.DataFrame({
    "permutation": range(n_permutations),
    "macro_f1": null_scores
}).to_csv(
    results_dir / "permutation_scores.csv",
    index=False
)

summary = pd.DataFrame([{
    "observed_macro_f1": observed_score,
    "null_mean": null_scores.mean(),
    "null_std": null_scores.std(),
    "null_max": null_scores.max(),
    "empirical_p_value": p_value,
    "n_permutations": n_permutations
}])

summary.to_csv(
    results_dir / "permutation_summary.csv",
    index=False
)

print("\nPermutation test results:")
print(summary.to_string(index=False))
print("\nPermutation test completed.")
