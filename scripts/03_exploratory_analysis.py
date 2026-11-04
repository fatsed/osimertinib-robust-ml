import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Dataset path
project_dir = Path(__file__).resolve().parents[1]

data_path = (
    project_dir
    / "data"
    / "raw"
    / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
)

# Load gene expression data
df = pd.read_csv(
    data_path,
    sep="\t",
    compression="gzip",
    index_col=0
)

# Inspect data structure
print("Original shape:", df.shape)
print("\nFirst 5 rows:")
print(df.iloc[:5, :5])

# Transpose the matrix
X = df.T
print("\nTransposed shape:", X.shape)


# Gene expression distribution
results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

expression_values = X.to_numpy().ravel()

plt.figure(figsize=(9, 5))

plt.hist(expression_values, bins=80)

plt.title("Distribution of Gene Expression Values")
plt.xlabel("Gene Expression (log2 TPM)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    results_dir / "01_expression_distribution.png",
    dpi=300
)

plt.show()


# Check the minimum expression value
min_value = expression_values.min()
min_count = (expression_values == min_value).sum()
min_percentage = (min_count / expression_values.size) * 100

print("\nExpression lower-bound analysis:")
print("Minimum value:", min_value)
print("Count at minimum:", min_count)
print("Percentage at minimum:", round(min_percentage, 2), "%")


# Identify genes with zero TPM in all samples

zero_expression_genes = (X == min_value).all(axis=0)

print("\nZero-expression gene analysis:")
print("Genes with zero TPM in all samples:", zero_expression_genes.sum())
print("Genes with nonzero TPM in at least one sample:", (~zero_expression_genes).sum())


# PCA analysis

X_filtered = X.loc[:, ~zero_expression_genes]

pca = PCA(
    n_components=2,
    svd_solver="randomized",
    random_state=42
)

X_pca = pca.fit_transform(X_filtered)

print("\nPCA analysis:")
print("Input shape:", X_filtered.shape)
print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)


# PCA visualization by cell line

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=X_filtered.index
)

pca_df["cell_line"] = [
    name.split("_")[0]
    for name in pca_df.index
]

plt.figure(figsize=(9, 6))

for cell_line, group in pca_df.groupby("cell_line"):
    plt.scatter(
        group["PC1"],
        group["PC2"],
        label=cell_line,
        s=65,
        alpha=0.8
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)"
)

plt.title("PCA of Gene Expression by Cell Line")
plt.legend(title="Cell Line")
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    results_dir / "02_pca_cell_lines.png",
    dpi=300
)

plt.show()


# PCA visualization by treatment condition

pca_df["condition"] = [
    name.split("_", 1)[1].rsplit("_", 1)[0]
    for name in pca_df.index
]

plt.figure(figsize=(9, 6))

for condition, group in pca_df.groupby("condition"):
    plt.scatter(
        group["PC1"],
        group["PC2"],
        label=condition,
        s=65,
        alpha=0.8
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)"
)

plt.title("PCA of Gene Expression by Treatment Condition")
plt.legend(title="Treatment Condition")
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    results_dir / "03_pca_conditions.png",
    dpi=300
)

plt.show()

# PCA analysis within PC9 cell line

pc9_data = pca_df[pca_df["cell_line"] == "PC9"]

plt.figure(figsize=(8, 6))

for condition, group in pc9_data.groupby("condition"):
    plt.scatter(
        group["PC1"],
        group["PC2"],
        label=condition,
        s=80,
        alpha=0.8
    )

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Treatment Conditions in PC9")
plt.legend(title="Treatment")
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    results_dir / "04_pca_pc9_conditions.png",
    dpi=300
)

plt.show()


# Compare DTP response across cell lines
effect_vectors = {}

for cell_line in sorted(pca_df["cell_line"].unique()):
    samples = pca_df[pca_df["cell_line"] == cell_line]

    control_ids = samples.index[
        samples["condition"] == "DMSO"
    ]

    dtp_ids = samples.index[
        samples["condition"] == "osi_DTP"
    ]

    control_mean = X.loc[control_ids].mean(axis=0)
    dtp_mean = X.loc[dtp_ids].mean(axis=0)

    effect_vectors[cell_line] = dtp_mean - control_mean

effect_df = pd.DataFrame(effect_vectors)

similarity = effect_df.corr()

print("\nDTP response similarity across cell lines:")
print(similarity.round(3))
