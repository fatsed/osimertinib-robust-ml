from pathlib import Path
from urllib.request import urlretrieve

url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE193258&file=GSE193258_RNAseq_log2TPM_abundance.tsv.gz&format=file"

project_dir = Path(__file__).resolve().parents[1]
output = project_dir / "data" / "raw" / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"

output.parent.mkdir(parents=True, exist_ok=True)

if output.exists():
    print("Dataset already exists.")
else:
    print("Downloading dataset...")
    urlretrieve(url, output)
    print("Download completed:", output)
