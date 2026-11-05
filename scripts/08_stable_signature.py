from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Paths
project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)


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

train_mask = (
    conditions.isin(["DMSO", "osi_DTP"])
    & (cell_lines != "PC9")
)

X_train = first.loc[train_mask]
y_train = conditions.loc[train_mask]
train_groups = cell_lines.loc[train_mask]


# Load development study
second = pd.read_csv(
    processed_dir / "GSE255958_PC9_expression.csv.gz",
    index_col=0
)

metadata = pd.read_csv(
    processed_dir / "GSE255958_PC9_metadata.csv"
).set_index("sample_id")

test_ids = metadata.index[
    metadata["condition"].isin(["DMSO", "osi_DTP"])
]

X_test = second.loc[test_ids]
y_test = metadata.loc[test_ids, "condition"]


# Match gene features
genes = sorted(set(X_train.columns) & set(X_test.columns))

X_train = X_train.loc[:, genes]
X_test = X_test.loc[:, genes]

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Calculate percentile ranks within each sample
train_ranks = X_train.rank(
    axis=1, method="average", pct=True
)

test_ranks = X_test.rank(
    axis=1, method="average", pct=True
)


# Calculate DTP effects in each training cell line
effects = {}

for cell_line in sorted(train_groups.unique()):
    mask = train_groups == cell_line

    control = train_ranks.loc[
        mask & (y_train == "DMSO")
    ]

    dtp = train_ranks.loc[
        mask & (y_train == "osi_DTP")
    ]

    effects[cell_line] = (
        dtp.mean(axis=0) - control.mean(axis=0)
    )

effect_df = pd.DataFrame(effects)


# Select stable genes using training data only
same_direction = (
    (effect_df > 0).all(axis=1)
    | (effect_df < 0).all(axis=1)
)

stability_score = effect_df.abs().min(axis=1)
eligible_scores = stability_score.loc[same_direction]

selected_genes = (
    eligible_scores.nlargest(50).index.tolist()
)

if len(selected_genes) < 50:
    raise ValueError("Fewer than 50 stable genes found.")

print("\nStable gene candidates:", same_direction.sum())
print("Selected genes:", len(selected_genes))
print("First 10 selected genes:", selected_genes[:10])


# Train classifier on selected genes
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


# Evaluate on development study
predictions = model.predict(
    test_ranks[selected_genes]
)

accuracy = accuracy_score(y_test, predictions)

macro_f1 = f1_score(
    y_test,
    predictions,
    labels=["DMSO", "osi_DTP"],
    average="macro",
    zero_division=0
)

print("\nStable signature results:")
print("Accuracy:", round(accuracy, 3))
print("Macro F1:", round(macro_f1, 3))

labels = ["DMSO", "osi_DTP"]

cm = confusion_matrix(
    y_test, predictions, labels=labels
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
))


# Save results
pd.DataFrame({
    "gene_symbol": selected_genes,
    "stability_score": eligible_scores.loc[
        selected_genes
    ].to_numpy()
}).to_csv(
    results_dir / "stable_signature_genes.csv",
    index=False
)

pd.DataFrame({
    "sample_id": X_test.index,
    "actual": y_test.to_numpy(),
    "predicted": predictions
}).to_csv(
    results_dir / "stable_signature_predictions.csv",
    index=False
)

print("\nStable signature experiment completed.")

# Ablation study: internal cross-cell-line validation
from sklearn.model_selection import LeaveOneGroupOut

all_mask = conditions.isin(["DMSO", "osi_DTP"])

X_all = first.loc[all_mask, genes]
y_all = conditions.loc[all_mask]
groups_all = cell_lines.loc[all_mask]

logo = LeaveOneGroupOut()
ablation_results = []

for train_idx, test_idx in logo.split(
    X_all, y_all, groups=groups_all
):
    X_fold_train = X_all.iloc[train_idx].rank(
        axis=1, method="average", pct=True
    )

    X_fold_test = X_all.iloc[test_idx].rank(
        axis=1, method="average", pct=True
    )

    y_fold_train = y_all.iloc[train_idx]
    y_fold_test = y_all.iloc[test_idx]
    groups_train = groups_all.iloc[train_idx]

    test_line = groups_all.iloc[test_idx].iloc[0]

    # Calculate treatment effects in training cell lines
    fold_effects = {}

    for line in sorted(groups_train.unique()):
        control = X_fold_train.loc[
            (groups_train == line)
            & (y_fold_train == "DMSO")
        ]

        dtp = X_fold_train.loc[
            (groups_train == line)
            & (y_fold_train == "osi_DTP")
        ]

        fold_effects[line] = (
            dtp.mean(axis=0) - control.mean(axis=0)
        )

    effects_df = pd.DataFrame(fold_effects)

    # Method 1: Stable genes
    same_direction = (
        (effects_df > 0).all(axis=1)
        | (effects_df < 0).all(axis=1)
    )

    stable_scores = (
        effects_df.abs().min(axis=1)
        .loc[same_direction]
    )

    if len(stable_scores) < 50:
        raise ValueError("Not enough stable genes.")

    stable_genes = stable_scores.nlargest(50).index.tolist()

    # Method 2: Largest pooled treatment effects
    pooled_effect = (
        X_fold_train.loc[y_fold_train == "osi_DTP"].mean()
        - X_fold_train.loc[y_fold_train == "DMSO"].mean()
    )

    pooled_genes = (
        pooled_effect.abs().nlargest(50).index.tolist()
    )

    # Method 3: Random genes
    rng = np.random.default_rng(42)

    random_genes = rng.choice(
        X_fold_train.columns.to_numpy(),
        size=50,
        replace=False
    ).tolist()

    gene_sets = {
        "Stable Signature": stable_genes,
        "Pooled Effect": pooled_genes,
        "Random Genes": random_genes
    }

    for method_name, selected in gene_sets.items():

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

        classifier.fit(
            X_fold_train[selected],
            y_fold_train
        )

        y_pred = classifier.predict(
            X_fold_test[selected]
        )

        ablation_results.append({
            "method": method_name,
            "test_cell_line": test_line,
            "accuracy": accuracy_score(
                y_fold_test, y_pred
            ),
            "macro_f1": f1_score(
                y_fold_test, y_pred,
                labels=["DMSO", "osi_DTP"],
                average="macro",
                zero_division=0
            )
        })

ablation_df = pd.DataFrame(ablation_results)

print("\nAblation results:")
print(ablation_df.to_string(index=False))

print("\nMean performance by method:")
print(
    ablation_df.groupby("method")[
        ["accuracy", "macro_f1"]
    ].mean().round(3)
)

ablation_df.to_csv(
    results_dir / "stable_signature_ablation.csv",
    index=False
)

print("\nAblation study completed.")

# Robustness check: repeated random gene selection

random_results = []

for train_idx, test_idx in logo.split(
    X_all, y_all, groups=groups_all
):
    X_fold_train = X_all.iloc[train_idx].rank(
        axis=1, method="average", pct=True
    )

    X_fold_test = X_all.iloc[test_idx].rank(
        axis=1, method="average", pct=True
    )

    y_fold_train = y_all.iloc[train_idx]
    y_fold_test = y_all.iloc[test_idx]

    test_line = groups_all.iloc[test_idx].iloc[0]

    for seed in range(30):
        rng = np.random.default_rng(seed)

        random_genes = rng.choice(
            X_fold_train.columns.to_numpy(),
            size=50,
            replace=False
        ).tolist()

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )

        classifier.fit(
            X_fold_train[random_genes],
            y_fold_train
        )

        predictions = classifier.predict(
            X_fold_test[random_genes]
        )

        random_results.append({
            "seed": seed,
            "test_cell_line": test_line,
            "accuracy": accuracy_score(
                y_fold_test, predictions
            ),
            "macro_f1": f1_score(
                y_fold_test, predictions,
                labels=["DMSO", "osi_DTP"],
                average="macro",
                zero_division=0
            )
        })

random_df = pd.DataFrame(random_results)

random_df.to_csv(
    results_dir / "random_gene_robustness.csv",
    index=False
)

seed_summary = random_df.groupby("seed")[
    ["accuracy", "macro_f1"]
].mean()

print("\nRepeated random selection (30 seeds):")
print("Mean:")
print(seed_summary.mean().round(3))

print("\nStandard deviation:")
print(seed_summary.std().round(3))

print("\nAccuracy range:")
print(
    seed_summary["accuracy"].min(),
    seed_summary["accuracy"].max()
)

print("\nRandom robustness check completed.")
