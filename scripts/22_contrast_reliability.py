"""Independent-replicate Acute-to-DTP contrast reliability audit.

Run: python scripts/22_contrast_reliability.py

No model is trained. In each of 9 splits per cell line, one Acute and one DTP
sample are held out, and the contrast is estimated from the *other two*
replicates in each condition. Selected genes come exclusively from the
training replicates of that split. These are exploratory diagnostics.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOP_N = 50


def read_data():
    first = pd.read_csv(
        RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
        sep="\t", index_col=0,
    ).T
    names = first.index.to_series()
    first_meta = pd.DataFrame({
        "cell_line": names.str.split("_").str[0],
        "condition": names.str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0],
    }, index=first.index)

    second = pd.read_csv(
        PROCESSED / "GSE255958_PC9_expression.csv.gz", index_col=0
    )
    second_meta = pd.read_csv(
        PROCESSED / "GSE255958_PC9_metadata.csv"
    ).set_index("sample_id")
    if not second.index.equals(second_meta.index):
        if set(second.index) != set(second_meta.index):
            raise ValueError("GSE255958 expression/metadata sample mismatch")
        second_meta = second_meta.loc[second.index]

    common = sorted(set(first.columns) & set(second.columns))
    if len(common) < 1000:
        raise ValueError("Too few common gene identifiers")

    # Rank each sample independently. No pooled fit or test-data statistics.
    first = first.loc[:, common].rank(axis=1, method="average", pct=True)
    second = second.loc[:, common].rank(axis=1, method="average", pct=True)
    return first, first_meta, second, second_meta, common


def effect(x, acute_ids, dtp_ids):
    return x.loc[dtp_ids].mean(axis=0) - x.loc[acute_ids].mean(axis=0)


def split_reliability(study, cell_line, expression, metadata):
    mask = metadata["cell_line"].eq(cell_line)
    acute = list(metadata.index[mask & metadata["condition"].eq("osi_acute")])
    dtp = list(metadata.index[mask & metadata["condition"].eq("osi_DTP")])
    if len(acute) != 3 or len(dtp) != 3:
        raise ValueError(f"Expected 3+3 samples: {study}, {cell_line}; found {len(acute)}+{len(dtp)}")
    rows = []
    for held_acute, held_dtp in product(acute, dtp):
        train_acute = [sid for sid in acute if sid != held_acute]
        train_dtp = [sid for sid in dtp if sid != held_dtp]
        train_delta = effect(expression, train_acute, train_dtp)
        hold_delta = expression.loc[held_dtp] - expression.loc[held_acute]
        top = train_delta.abs().nlargest(TOP_N).index
        rows.append({
            "study": study,
            "cell_line": cell_line,
            "held_out_acute": held_acute,
            "held_out_dtp": held_dtp,
            "n_train": 4,
            "n_holdout": 2,
            "spearman_train_vs_holdout": float(train_delta.corr(hold_delta, method="spearman")),
            "top50_sign_agreement": float((np.sign(train_delta.loc[top]) == np.sign(hold_delta.loc[top])).mean()),
        })
    return rows


def main():
    first, first_meta, second, second_meta, common = read_data()
    print(f"Shared genes: {len(common)}")
    all_rows = []
    for line in sorted(first_meta["cell_line"].unique()):
        all_rows.extend(split_reliability("GSE193258", line, first, first_meta))
    all_rows.extend(split_reliability("GSE255958", "PC9", second, second_meta))

    splits = pd.DataFrame(all_rows)
    summary = (splits.groupby(["study", "cell_line"], as_index=False)
               .agg(median_spearman=("spearman_train_vs_holdout", "median"),
                    min_spearman=("spearman_train_vs_holdout", "min"),
                    median_top50_agreement=("top50_sign_agreement", "median"),
                    min_top50_agreement=("top50_sign_agreement", "min")))

    # Cross-study comparison uses ALL replicates and is descriptive only.
    # It must not be used for selecting features and claiming a fresh test.
    first_ids = first_meta.index[first_meta["cell_line"].eq("PC9")]
    first_pc9 = first_meta.loc[first_ids]
    delta_a = effect(
        first, first_pc9.index[first_pc9["condition"].eq("osi_acute")],
        first_pc9.index[first_pc9["condition"].eq("osi_DTP")],
    )
    delta_b = effect(
        second, second_meta.index[second_meta["condition"].eq("osi_acute")],
        second_meta.index[second_meta["condition"].eq("osi_DTP")],
    )
    top = delta_a.abs().nlargest(TOP_N).index
    cross = pd.DataFrame([{
        "comparison": "PC9 GSE193258 vs GSE255958",
        "spearman_all_genes": float(delta_a.corr(delta_b, method="spearman")),
        "top50_direction_agreement": float((np.sign(delta_a.loc[top]) == np.sign(delta_b.loc[top])).mean()),
    }])

    splits.to_csv(RESULTS / "contrast_reliability_splits.csv", index=False)
    summary.to_csv(RESULTS / "contrast_reliability_summary.csv", index=False)
    cross.to_csv(RESULTS / "contrast_reliability_cross_study.csv", index=False)
    print("\nDISJOINT REPLICATE CONTRAST RELIABILITY (2+2 train, 1+1 held out):")
    print(summary.round(3).to_string(index=False))
    print("\nCROSS-STUDY PC9 (descriptive, previously seen during development):")
    print(cross.round(3).to_string(index=False))
    print("\nSaved three CSV reports. No model was trained.")


if __name__ == "__main__":
    main()
