"""Pre-model data quality audit for osimertinib-robust-ml.

Run from the repository root:
    python scripts/21_dataset_quality_audit.py

This script does not train or tune a model. All cross-study contrast comparisons
are descriptive diagnostics, not an independent test or feature-selection step.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "GSE193258": RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
    "GSE255958": PROCESSED / "GSE255958_PC9_expression.csv.gz",
    "GSE249721": PROCESSED / "GSE249721_HCC827_original_values.csv.gz",
    "GSE285632": PROCESSED / "GSE285632_expression.csv.gz",
}


def load_expression(path, sep=","):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    data = pd.read_csv(path, sep=sep, index_col=0)
    if data.index.has_duplicates:
        raise ValueError(f"Duplicated sample IDs: {path.name}")
    if data.columns.has_duplicates:
        raise ValueError(f"Duplicated gene IDs: {path.name}")
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError(f"NaN / infinity in expression matrix: {path.name}")
    return data


def load_metadata(path, expression):
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    meta = pd.read_csv(path)
    if "sample_id" not in meta or meta["sample_id"].duplicated().any():
        raise ValueError(f"Missing or duplicated sample IDs in {path.name}")
    meta = meta.set_index("sample_id")
    if set(meta.index) != set(expression.index):
        raise ValueError(
            f"Expression/metadata mismatch in {path.name}: "
            f"expression only={list(set(expression.index)-set(meta.index))[:4]}, "
            f"metadata only={list(set(meta.index)-set(expression.index))[:4]}"
        )
    meta = meta.loc[expression.index]
    if not {"study", "cell_line", "condition"}.issubset(meta.columns):
        raise ValueError(f"Missing required metadata fields in {path.name}")
    return meta


# Dataset 1 contains five conditions, of which three are used here.
first_all = load_expression(REQUIRED["GSE193258"], sep="\t").T
if first_all.index.has_duplicates or first_all.columns.has_duplicates:
    raise ValueError("Duplicate sample or gene IDs in GSE193258")
sample_names = first_all.index.to_series()
first_meta = pd.DataFrame({
    "study": "GSE193258",
    "cell_line": sample_names.str.split("_").str[0],
    "condition": (
        sample_names.str.split("_", n=1).str[1]
        .str.rsplit("_", n=1).str[0]
    ),
}, index=first_all.index)
first_mask = first_meta["condition"].isin(["DMSO", "osi_acute", "osi_DTP"])
first = first_all.loc[first_mask]
first_meta = first_meta.loc[first_mask]

studies = {
    "GSE193258": (first, first_meta),
}
for study, expr_file, meta_file in [
    ("GSE255958", "GSE255958_PC9_expression.csv.gz", "GSE255958_PC9_metadata.csv"),
    ("GSE249721", "GSE249721_HCC827_original_values.csv.gz", "GSE249721_HCC827_metadata.csv"),
    ("GSE285632", "GSE285632_expression.csv.gz", "GSE285632_metadata.csv"),
]:
    expression = load_expression(PROCESSED / expr_file)
    metadata = load_metadata(PROCESSED / meta_file, expression)
    studies[study] = (expression, metadata)

# Explicit source checks based on the selected GEO sample IDs.
expected_255958 = {f"GSM{i}" for i in range(8083374, 8083383)}
if set(studies["GSE255958"][0].index) != expected_255958:
    raise ValueError("GSE255958 GSM identities differ from the planned 9 samples")

# Keep the original metadata labels; do not silently equate untreated and DMSO.
summary_rows = []
replicate_rows = []
floor = np.log2(0.01)

for study, (expression, metadata) in studies.items():
    values = expression.to_numpy()
    floor_fraction = (
        float(np.isclose(values, floor, atol=1e-6).mean())
        if study in {"GSE193258", "GSE255958"} else np.nan
    )
    study_warnings = []
    if study == "GSE249721":
        study_warnings.append("Untreated mapped to DMSO; processed scale not verified")
    if study == "GSE285632":
        study_warnings.append("P/D columns to GEO GSM mapping not verified; EXCLUDED")
    if study in {"GSE193258", "GSE255958"}:
        study_warnings.append("Different osimertinib exposure durations across studies")

    summary_rows.append({
        "study": study,
        "n_samples": len(expression),
        "n_genes": expression.shape[1],
        "n_cell_lines": metadata["cell_line"].nunique(),
        "conditions": "; ".join(f"{x}:{n}" for x, n in metadata["condition"].value_counts().sort_index().items()),
        "missing_values": int(np.isnan(values).sum()),
        "nonfinite_values": int((~np.isfinite(values)).sum()),
        "constant_genes": int((expression.nunique(axis=0) <= 1).sum()),
        "min_expression": float(values.min()),
        "max_expression": float(values.max()),
        "fraction_at_log2_floor": floor_fraction,
        "primary_analysis": study != "GSE285632",
        "known_limitations": "; ".join(study_warnings),
    })

    for (cell_line, condition), rows in metadata.groupby(["cell_line", "condition"]):
        ids = list(rows.index)
        if len(ids) < 2:
            replicate_rows.append({
                "study": study, "cell_line": cell_line, "condition": condition,
                "n_replicates": len(ids), "median_spearman": np.nan,
                "min_spearman": np.nan, "flag": "LESS_THAN_TWO_SAMPLES",
            })
            continue
        # Within-group sample-to-sample Spearman across genes.
        correlations = expression.loc[ids].T.corr(method="spearman")
        pair_values = [float(correlations.loc[a, b]) for a, b in combinations(ids, 2)]
        finite_pairs = [v for v in pair_values if np.isfinite(v)]
        minimum = float(min(finite_pairs)) if finite_pairs else np.nan
        replicate_rows.append({
            "study": study, "cell_line": cell_line, "condition": condition,
            "n_replicates": len(ids),
            "median_spearman": float(np.median(finite_pairs)) if finite_pairs else np.nan,
            "min_spearman": minimum,
            # A screening threshold only, not a biological validity criterion.
            "flag": "REVIEW" if not np.isfinite(minimum) or minimum < 0.90 else "OK_SCREEN",
        })

summary = pd.DataFrame(summary_rows)
replicates = pd.DataFrame(replicate_rows)

# Contrast consistency across matched CELL LINES only. No class labels are pooled.
def ranked_effect(study, line, positive, reference):
    X, M = studies[study]
    mask = M["cell_line"].eq(line)
    cases = X.loc[mask & M["condition"].eq(positive)]
    controls = X.loc[mask & M["condition"].eq(reference)]
    if cases.empty or controls.empty:
        raise ValueError(f"Missing contrast {study}/{line}: {reference} -> {positive}")
    # Rank each sample independently to avoid relying on incompatible absolute scales.
    cases_rank = cases.rank(axis=1, pct=True, method="average")
    controls_rank = controls.rank(axis=1, pct=True, method="average")
    return cases_rank.mean(axis=0) - controls_rank.mean(axis=0)


comparison_plans = [
    ("PC9 DTP vs acute", "GSE193258", "PC9", "osi_DTP", "osi_acute", "GSE255958", "PC9", "osi_DTP", "osi_acute"),
    ("PC9 DTP vs control", "GSE193258", "PC9", "osi_DTP", "DMSO", "GSE255958", "PC9", "osi_DTP", "DMSO"),
    ("HCC827 DTP vs control", "GSE193258", "HCC827", "osi_DTP", "DMSO", "GSE249721", "HCC827", "osi_DTP", "DMSO"),
]
contrast_rows = []
for label, a, line_a, pos_a, ref_a, b, line_b, pos_b, ref_b in comparison_plans:
    effect_a = ranked_effect(a, line_a, pos_a, ref_a)
    effect_b = ranked_effect(b, line_b, pos_b, ref_b)
    common = effect_a.index.intersection(effect_b.index)
    va, vb = effect_a.loc[common], effect_b.loc[common]
    largest = va.abs().nlargest(min(50, len(va))).index
    agreement = float((np.sign(va.loc[largest]) == np.sign(vb.loc[largest])).mean())
    contrast_rows.append({
        "comparison": label, "study_A": a, "study_B": b,
        "n_common_genes": len(common),
        "pearson": float(va.corr(vb)),
        "spearman": float(va.corr(vb, method="spearman")),
        "top50_direction_agreement": agreement,
        "note": "Exploratory; exposure and/or controls differ across studies",
    })

contrasts = pd.DataFrame(contrast_rows)

# Review eligibility information from metadata audit; never infer sample-to-GSM mapping.
audit_path = RESULTS / "metadata_audit.csv"
if audit_path.exists():
    audit = pd.read_csv(audit_path)
    if "primary_analysis_eligible" not in audit:
        raise ValueError("metadata_audit.csv lacks primary_analysis_eligible")
    is_primary = audit["primary_analysis_eligible"].astype(str).str.lower().eq("true")
    if len(audit) != 42 or int(is_primary.sum()) != 36:
        raise ValueError("Metadata audit expected 42 binary samples with 36 eligible")
    if (is_primary & audit["study"].eq("GSE285632")).any():
        raise ValueError("Unverified GSE285632 unexpectedly marked eligible")
else:
    print("WARNING: results/metadata_audit.csv not found; run script 13 first")

primary_genes = set.intersection(*(set(studies[s][0].columns) for s in ["GSE193258", "GSE255958", "GSE249721"]))
print("\nPrimary studies - shared genes:", len(primary_genes))
print("\nSTUDY QC SUMMARY")
print(summary.to_string(index=False))
print("\nBIOLOGICAL REPLICATE QC")
print(replicates.to_string(index=False))
print("\nMATCHED-CELL-LINE RESPONSE CONSISTENCY (EXPLORATORY)")
print(contrasts.to_string(index=False))

summary.to_csv(RESULTS / "qc_study_summary.csv", index=False)
replicates.to_csv(RESULTS / "qc_replicate_correlations.csv", index=False)
contrasts.to_csv(RESULTS / "qc_cross_study_contrasts.csv", index=False)

print("\nSaved 3 QC reports in results/ . No model trained or altered.")
