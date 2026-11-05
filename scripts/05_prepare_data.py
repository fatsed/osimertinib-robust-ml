import gzip
import re
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
project_dir = Path(__file__).resolve().parents[1]

first_dataset_path = (
    project_dir / "data" / "raw"
    / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
)

second_dataset_path = (
    project_dir / "data" / "raw" / "GSE255958_RAW.tar"
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)


# Define selected samples using GEO metadata
sample_conditions = {
    **{f"GSM{i}": "DMSO" for i in range(8083374, 8083377)},
    **{f"GSM{i}": "osi_acute" for i in range(8083377, 8083380)},
    **{f"GSM{i}": "osi_DTP" for i in range(8083380, 8083383)},
}


# Read gene identifiers from dataset 1
first_genes = set(
    pd.read_csv(
        first_dataset_path,
        sep="\t",
        usecols=[0]
    ).iloc[:, 0].astype(str)
)


# Convert dataset 2 gene identifiers
def extract_gene_symbol(gene_id):
    match = re.fullmatch(r"(.+)_\1", str(gene_id))

    if match is None:
        raise ValueError(f"Unexpected gene ID: {gene_id}")

    return match.group(1)


# Read selected samples from dataset 2
expression_data = {}
metadata_rows = []

with tarfile.open(second_dataset_path, "r") as archive:
    members = [
        member for member in archive.getmembers()
        if member.isfile()
        and member.name.endswith(".genes.results.gz")
    ]

    for sample_id, condition in sample_conditions.items():
        matches = [
            member for member in members
            if member.name.startswith(f"{sample_id}_")
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected one file for {sample_id}, "
                f"found {len(matches)}"
            )

        member = matches[0]
        sample_stream = archive.extractfile(member)

        if sample_stream is None:
            raise ValueError(f"Cannot read {member.name}")

        with sample_stream:
            with gzip.GzipFile(fileobj=sample_stream) as gz:
                sample_df = pd.read_csv(
                    gz,
                    sep="\t",
                    usecols=["gene_id", "TPM"]
                )

        symbols = sample_df["gene_id"].map(
            extract_gene_symbol
        )

        if symbols.duplicated().any():
            raise ValueError(
                f"Duplicate gene symbols in {sample_id}"
            )

        tpm = pd.to_numeric(
            sample_df["TPM"],
            errors="raise"
        ).to_numpy(dtype=float)

        if not np.isfinite(tpm).all() or (tpm < 0).any():
            raise ValueError(f"Invalid TPM in {sample_id}")

        expression_data[sample_id] = pd.Series(
            np.log2(tpm + 0.01),
            index=symbols.to_numpy(),
            dtype=float
        )

        metadata_rows.append({
            "sample_id": sample_id,
            "study": "GSE255958",
            "cell_line": "PC9",
            "condition": condition,
            "source_file": member.name
        })


# Identify genes shared by both studies
common_genes = set(first_genes)

for sample in expression_data.values():
    common_genes.intersection_update(sample.index)

common_genes = sorted(common_genes)

if not common_genes:
    raise ValueError("No common genes found.")


# Construct sample-by-gene expression matrix
expression_matrix = pd.DataFrame({
    sample_id: expression.reindex(common_genes)
    for sample_id, expression in expression_data.items()
}).T

expression_matrix.index.name = "sample_id"

if expression_matrix.isna().any().any():
    raise ValueError("Missing values in expression matrix.")

metadata = pd.DataFrame(metadata_rows)


# Save processed data
expression_matrix.to_csv(
    output_dir / "GSE255958_PC9_expression.csv.gz",
    compression="gzip"
)

metadata.to_csv(
    output_dir / "GSE255958_PC9_metadata.csv",
    index=False
)


# Report results
print("\nGSE255958 preprocessing completed")
print("Matrix shape:", expression_matrix.shape)
print("Shared genes:", len(common_genes))
print("Missing values:", expression_matrix.isna().sum().sum())

print("\nSamples per condition:")
print(metadata["condition"].value_counts())

print("\nSaved processed files successfully.")


# Prepare GSE249721 HCC827 expression matrix

third_dataset_path = (
    project_dir / "data" / "raw"
    / "GSE249721_all_samples_processed_data.tsv.gz"
)

third_df = pd.read_csv(
    third_dataset_path,
    sep="\t",
    compression="gzip",
    index_col=0
)

# Select six HCC827 samples
control_columns = [
    f"HCC827Untreated{i}.genes.results"
    for i in range(1, 4)
]

dtp_columns = [
    f"HCC827Osimertinib11d{i}.genes.results"
    for i in range(1, 4)
]

selected_columns = control_columns + dtp_columns

missing_columns = [
    col for col in selected_columns
    if col not in third_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing HCC827 samples: {missing_columns}"
    )

# Extract gene symbols
gene_symbols = pd.Series(
    third_df.index.astype(str)
).str.extract(
    r"^ENSG\d+(?:\.\d+)?_(.+)$",
    expand=False
)

# Remove ambiguous or duplicated gene symbols
keep_mask = (
    gene_symbols.notna()
    & ~gene_symbols.duplicated(keep=False)
)

removed_genes = (~keep_mask).sum()

third_expression = third_df.iloc[
    keep_mask.to_numpy()
][selected_columns].copy()

third_expression.index = gene_symbols[
    keep_mask
].to_numpy()

# Convert to sample-by-gene matrix
third_expression = third_expression.T

third_expression.index = [
    name.removesuffix(".genes.results")
    for name in third_expression.index
]

third_expression.index.name = "sample_id"

third_expression = third_expression.apply(
    pd.to_numeric,
    errors="raise"
)

if not np.isfinite(
    third_expression.to_numpy()
).all():
    raise ValueError("Invalid expression values.")

# Create metadata
third_metadata = pd.DataFrame({
    "sample_id": third_expression.index,
    "study": "GSE249721",
    "cell_line": "HCC827",
    "condition": ["DMSO"] * 3 + ["osi_DTP"] * 3,
    "treatment_days": [0] * 3 + [11] * 3
})

# Save original-scale values
third_expression.to_csv(
    output_dir / "GSE249721_HCC827_original_values.csv.gz",
    compression="gzip"
)

third_metadata.to_csv(
    output_dir / "GSE249721_HCC827_metadata.csv",
    index=False
)

# Report
shared_genes = set(first_genes) & set(
    third_expression.columns
)

print("\nGSE249721 preparation completed")
print("Matrix shape:", third_expression.shape)
print("Shared genes:", len(shared_genes))
print("Removed gene rows:", removed_genes)
print(
    "Missing values:",
    third_expression.isna().sum().sum()
)

print("\nSamples per condition:")
print(third_metadata["condition"].value_counts())

print("\nOriginal-scale data saved successfully.")

# Prepare GSE285632: Parental vs DTP

fourth_path = (
    project_dir
    / "data"
    / "raw"
    / "GSE285632_All.DEG.Expression.xlsx"
)

fourth_df = pd.read_excel(
    fourth_path,
    sheet_name="mRNA",
    header=1
)

fourth_df.columns = fourth_df.columns.str.strip()

sample_columns = [
    "P1", "P2", "P3",
    "D1", "D2", "D3"
]

# Validate gene identifiers
if fourth_df["gene_id"].isna().any():
    raise ValueError("Missing gene identifiers.")

if fourth_df["gene_id"].duplicated().any():
    raise ValueError("Duplicated gene identifiers.")

# Build sample-by-gene matrix
fourth_expression = (
    fourth_df.set_index("gene_id")[sample_columns]
    .apply(pd.to_numeric, errors="raise")
    .T
)

fourth_expression.index.name = "sample_id"

if not np.isfinite(fourth_expression.to_numpy()).all():
    raise ValueError("Invalid expression values.")

# Preserve original values without extra transformations
fourth_metadata = pd.DataFrame({
    "sample_id": sample_columns,
    "study": "GSE285632",
    "cell_line": "HCC827",
    "original_condition": ["Parental"] * 3 + ["DTP"] * 3,
    "condition": ["control"] * 3 + ["osi_DTP"] * 3
})

fourth_expression.to_csv(
    output_dir / "GSE285632_expression.csv.gz",
    compression="gzip"
)

fourth_metadata.to_csv(
    output_dir / "GSE285632_metadata.csv",
    index=False
)

shared_fourth = (
    set(first_genes) & set(fourth_expression.columns)
)

print("\nGSE285632 preparation completed")
print("Matrix shape:", fourth_expression.shape)
print("Shared genes:", len(shared_fourth))
print("Missing values:", fourth_expression.isna().sum().sum())

print("\nConditions:")
print(fourth_metadata["condition"].value_counts())

print("\nProcessed files saved successfully.")
