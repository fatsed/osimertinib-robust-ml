from pathlib import Path
from urllib.request import Request, urlopen
import gzip
import tarfile
import zipfile
import shutil


# Project paths
project_dir = Path(__file__).resolve().parents[1]

raw_dir = project_dir / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)


# Dataset registry
datasets = {
    "GSE193258": "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    "GSE255958": "GSE255958_RAW.tar",
    "GSE249721": "GSE249721_all_samples_processed_data.tsv.gz",
    "GSE285632": "GSE285632_All.DEG.Expression.xlsx",
}


# Validate downloaded file format
def is_valid_file(path):
    suffix = Path(
        path.name.removesuffix(".part")
    ).suffix.lower()

    try:
        if suffix == ".tar":
            return tarfile.is_tarfile(path)

        if suffix == ".xlsx":
            return zipfile.is_zipfile(path)

        if suffix == ".gz":
            with gzip.open(path, "rb") as file:
                return bool(file.read(100))

    except (OSError, EOFError, ValueError):
        return False

    return False


# Build direct GEO download URL
def build_geo_url(accession, filename):
    series_group = f"{accession[:-3]}nnn"

    return (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{series_group}/{accession}/suppl/{filename}"
    )


# Download a GEO supplementary file
def download_dataset(accession, filename):
    output = raw_dir / filename

    if output.exists() and is_valid_file(output):
        print(f"{accession}: Already downloaded.")
        return

    url = build_geo_url(accession, filename)

    temporary_file = raw_dir / f"{filename}.part"

    print(f"\nDownloading {accession}...")

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    try:
        with urlopen(request, timeout=120) as response:
            with open(temporary_file, "wb") as file:
                shutil.copyfileobj(response, file)

        if not is_valid_file(temporary_file):
            raise ValueError(
                f"Invalid downloaded file: {filename}"
            )

        temporary_file.replace(output)

    except Exception:
        temporary_file.unlink(missing_ok=True)
        raise

    size_mb = output.stat().st_size / (1024 ** 2)

    print(f"Downloaded: {filename}")
    print(f"Size: {size_mb:.2f} MB")


# Download all registered datasets
def main():
    for accession, filename in datasets.items():
        download_dataset(accession, filename)

    print("\nDataset download check completed.")


if __name__ == "__main__":
    main()
