
import csv
import gzip
import math
from collections import Counter
from pathlib import Path

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
