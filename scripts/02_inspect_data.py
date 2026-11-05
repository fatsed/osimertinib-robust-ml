
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

# Dataset 3: GSE249721

third_dataset_path = (
    project_dir
    / "data"
    / "raw"
    / "GSE249721_all_samples_processed_data.tsv.gz"
)

if not third_dataset_path.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {third_dataset_path}"
    )

print("\nDataset 3: GSE249721")

with gzip.open(third_dataset_path, "rt") as file:
    for i in range(4):
        line = file.readline()

        if not line:
            break

        values = line.rstrip("\n").split("\t")

        print(f"\nRow {i + 1}:")
        print("Number of columns:", len(values))
        print("First 12 values:", values[:12])


# Dataset 3: HCC827 sample and expression audit

third_df = pd.read_csv(
    third_dataset_path,
    sep="\t"
)

hcc827_columns = [
    col for col in third_df.columns
    if "HCC827" in col.upper()
]

print("\nHCC827 sample columns:")
for col in hcc827_columns:
    print(col)

print("\nNumber of HCC827 samples:", len(hcc827_columns))

# Extract symbols from Ensembl ID + gene symbol
third_ids = third_df.iloc[:, 0].astype(str)

third_symbols = third_ids.str.extract(
    r"^ENSG\d+(?:\.\d+)?_(.+)$",
    expand=False
)

print("\nGene identifier check:")
print("Total gene rows:", len(third_ids))
print("Valid identifiers:", third_symbols.notna().sum())
print("Duplicate symbols:", third_symbols.dropna().duplicated().sum())

print(
    "Shared genes with dataset 1:",
    len(set(third_symbols.dropna()) & first_dataset_genes)
)

# Examine expression scale
hcc827_values = third_df[hcc827_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

print("\nExpression value check:")
print("Missing values:", hcc827_values.isna().sum().sum())
print("Minimum:", hcc827_values.min().min())
print("Maximum:", hcc827_values.max().max())
print("Median:", hcc827_values.stack().median())
print("Negative values:", (hcc827_values < 0).sum().sum())

# Dataset 4: GSE285632

fourth_dataset_path = (
    project_dir
    / "data"
    / "raw"
    / "GSE285632_All.DEG.Expression.xlsx"
)

if not fourth_dataset_path.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {fourth_dataset_path}"
    )

print("\nDataset 4: GSE285632")

with pd.ExcelFile(fourth_dataset_path) as excel:
    print("\nSheet names:", excel.sheet_names)

    for sheet_name in excel.sheet_names:
        preview = pd.read_excel(
            excel,
            sheet_name=sheet_name,
            nrows=5
        )

        print(f"\nSheet: {sheet_name}")
        print("Number of columns:", len(preview.columns))
        print("First 15 columns:")
        print(preview.columns.tolist()[:15])

        print("\nFirst five rows:")
        print(
            preview.iloc[:, :8].to_string(index=False)
        )

# Dataset 4: Detailed mRNA inspection

mrna_df = pd.read_excel(
    fourth_dataset_path,
    sheet_name="mRNA",
    header=1
)

mrna_df.columns = (
    mrna_df.columns.astype(str).str.strip()
)

print("\nGSE285632 mRNA inspection:")
print("Matrix shape:", mrna_df.shape)
print("Columns:", mrna_df.columns.tolist())

# Check gene identifiers
gene_ids = mrna_df["gene_id"].dropna().astype(str)

print("\nGene information:")
print("Total gene rows:", len(gene_ids))
print("Duplicate gene IDs:", gene_ids.duplicated().sum())

print(
    "Shared genes with dataset 1:",
    len(set(gene_ids) & first_dataset_genes)
)

# Check expression values
sample_columns = ["P1", "P2", "P3", "D1", "D2", "D3"]

expression_values = mrna_df[
    sample_columns
].apply(pd.to_numeric, errors="coerce")

print("\nExpression information:")
print("Missing values:", expression_values.isna().sum().sum())
print("Minimum:", expression_values.min().min())
print("Maximum:", expression_values.max().max())
print("Median:", expression_values.stack().median())

print("\nFirst five rows:")
print(
    mrna_df[["gene_id"] + sample_columns]
    .head()
    .to_string(index=False)
)
