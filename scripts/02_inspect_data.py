
import csv
import gzip
import math
from collections import Counter
from pathlib import Path
import tarfile
import pandas as pd

# File paths
project_dir = Path(__file__).resolve().parents[1]

data_path = (
    project_dir
    / "data"
    / "raw"
    / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
)

if not data_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

# Initialize counters
gene_rows = 0
missing_values = 0
invalid_values = 0
incorrect_rows = 0

gene_counts = Counter()

# Read and inspect dataset
with gzip.open(data_path, "rt", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")
    header = next(reader, None)

    if header is None or len(header) < 2:
        raise ValueError("Invalid dataset header.")

    sample_names = header[1:]

    for row in reader:
        if not row:
            continue

        gene_rows += 1
        gene_counts[row[0].strip()] += 1

        if len(row) != len(header):
            incorrect_rows += 1
            continue

        for value in row[1:]:
            value = value.strip()

            if value.lower() in ("", "na", "nan", "null"):
                missing_values += 1
                continue

            try:
                if not math.isfinite(float(value)):
                    invalid_values += 1
            except ValueError:
                invalid_values += 1


# Analyze sample metadata
cell_lines = Counter(
    name.split("_")[0]
    for name in sample_names
)

conditions = Counter(
    name.split("_", 1)[1].rsplit("_", 1)[0]
    for name in sample_names
)

# Check duplicates
duplicate_genes = sum(
    count > 1 for count in gene_counts.values()
)

duplicate_samples = (
    len(sample_names) - len(set(sample_names))
)

print("\nDataset overview:")
print("Number of gene rows:", gene_rows)
print("Number of samples:", len(sample_names))
print("First 5 sample names:", sample_names[:5])

print("\nCell lines:")
for name, count in sorted(cell_lines.items()):
    print(name, ":", count)

print("\nExperimental conditions:")
for name, count in sorted(conditions.items()):
    print(name, ":", count)

print("\nData quality:")
print("Missing values:", missing_values)
print("Invalid numeric values:", invalid_values)
print("Rows with incorrect columns:", incorrect_rows)

print("\nDuplicate check:")
print("Unique gene IDs:", len(gene_counts))
print("Duplicated gene IDs:", duplicate_genes)
print("Duplicated sample names:", duplicate_samples)

# Dataset 2: GSE255958

second_archive_path = (
    project_dir / "data" / "raw" / "GSE255958_RAW.tar"
)

if not second_archive_path.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {second_archive_path}"
    )

with tarfile.open(second_archive_path, "r") as archive:
    pc9_members = sorted(
        [
            member
            for member in archive.getmembers()
            if member.isfile()
            and "_PC9_" in member.name
            and member.name.endswith(".genes.results.gz")
        ],
        key=lambda member: member.name
    )

    print("\nDataset 2: GSE255958")
    print("Total files:", len(archive.getmembers()))
    print("PC9 samples:", len(pc9_members))

    print("\nPC9 filenames:")
    for member in pc9_members:
        print(member.name)

    if not pc9_members:
        raise ValueError("No PC9 expression files found.")

    # Inspect the first PC9 sample
    sample_member = pc9_members[0]
    sample_stream = archive.extractfile(sample_member)

    if sample_stream is None:
        raise ValueError("Cannot read sample file.")

    with sample_stream:
        with gzip.GzipFile(fileobj=sample_stream) as gz:
            sample_df = pd.read_csv(
                gz,
                sep="\t",
                nrows=5
            )

    print("\nSample:", sample_member.name)
    print("\nColumns:")
    print(sample_df.columns.tolist())

    print("\nFirst five gene records:")
    print(sample_df.to_string(index=False))


# Compare gene identifiers between two datasets

with tarfile.open(second_archive_path, "r") as archive:
    sample_name = (
        "GSM8083374_PC9_D_1.genes.results.gz"
    )
    sample_stream = archive.extractfile(sample_name)
    with gzip.GzipFile(fileobj=sample_stream) as gz:
        genes_df = pd.read_csv(
            gz,
            sep="\t",
            usecols=["gene_id", "TPM"]
        )

# Inspect possible gene symbol conversion
original_ids = genes_df["gene_id"].astype(str)
candidate_symbols = original_ids.str.split(
    "_", n=1
).str[0]

first_dataset_genes = set(gene_counts.keys())

direct_overlap = len(
    set(original_ids) & first_dataset_genes
)

candidate_overlap = len(
    set(candidate_symbols) & first_dataset_genes
)

print("\nGene identifier comparison:")
print("Dataset 2 gene rows:", len(genes_df))
print("Direct overlap:", direct_overlap)
print("Candidate symbol overlap:", candidate_overlap)
print("Candidate duplicate genes:", candidate_symbols.duplicated().sum())

print("\nExample gene identifiers:")
print(pd.DataFrame({
    "Original": original_ids.head(5).values,
    "Candidate": candidate_symbols.head(5).values
}))


# Validate gene identifier conversion
import re

def extract_gene_symbol(gene_id):
    match = re.fullmatch(r"(.+)_\1", gene_id)
    if match:
        return match.group(1)
    return None

validated_symbols = original_ids.map(extract_gene_symbol)

audit_df = pd.DataFrame({
    "gene_id": original_ids,
    "symbol": validated_symbols
})

print("\nGene mapping validation:")
print(
    "Validated identifiers:",
    validated_symbols.notna().sum()
)
print(
    "Unrecognized identifiers:",
    validated_symbols.isna().sum()
)
print(
    "Validated gene overlap:",
    len(set(validated_symbols.dropna()) & first_dataset_genes)
)

duplicates = audit_df[
    audit_df["symbol"].notna()
    & audit_df["symbol"].duplicated(keep=False)
]

print("\nDuplicated gene symbols:")
print(duplicates.to_string(index=False))
print("\nUnrecognized examples:")
print(
    audit_df[audit_df["symbol"].isna()]
    .head(10)
    .to_string(index=False)
)
