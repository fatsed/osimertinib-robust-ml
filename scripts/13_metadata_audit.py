from pathlib import Path
import pandas as pd

# 1. Project paths

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"
results_dir = project_dir / "results"

results_dir.mkdir(parents=True, exist_ok=True)

audit_rows = []

# 2. Add sample metadata


def add_sample(
    study,
    sample_id,
    gsm,
    cell_line,
    original_condition,
    analysis_label,
    control_type,
    treatment_days,
    control_exposure_hours,
    expression_scale,
    verification_status
):
    audit_rows.append({
        "study": study,
        "sample_id": sample_id,
        "gsm": gsm,
        "cell_line": cell_line,
        "original_condition": original_condition,
        "analysis_label": analysis_label,
        "control_type": control_type,
        "drug": "osimertinib",
        "osimertinib_exposure_days": treatment_days,
        "control_exposure_hours": control_exposure_hours,
        "expression_scale": expression_scale,
        "verification_status": verification_status
    })

# 3. Dataset 1: GSE193258


first_gsm_starts = {
    "H1975": {
        "DMSO": 5777996,
        "osi_DTP": 5778005
    },
    "HCC2935": {
        "DMSO": 5778011,
        "osi_DTP": 5778020
    },
    "HCC827": {
        "DMSO": 5778026,
        "osi_DTP": 5778035
    },
    "PC9": {
        "DMSO": 5778041,
        "osi_DTP": 5778050
    }
}

first_path = (
    raw_dir
    / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz"
)

first_samples = pd.read_csv(
    first_path,
    sep="\t",
    nrows=0
).columns[1:]

for sample_id in first_samples:

    if "_DMSO_" in sample_id:
        condition = "DMSO"
        label = "Control"
        control = "DMSO"
        treatment_days = 0
        control_hours = 24

    elif "_osi_DTP_" in sample_id:
        condition = "osi_DTP"
        label = "DTP"
        control = ""
        treatment_days = 21
        control_hours = None

    else:
        continue

    cell_line = sample_id.split("_")[0]
    replicate = int(sample_id.rsplit("_", 1)[1])

    gsm_number = (
        first_gsm_starts[cell_line][condition]
        + replicate - 1
    )

    gsm_id = f"GSM{gsm_number}"

    add_sample(
        study="GSE193258",
        sample_id=sample_id,
        gsm=gsm_id,
        cell_line=cell_line,
        original_condition=condition,
        analysis_label=label,
        control_type=control,
        treatment_days=treatment_days,
        control_exposure_hours=control_hours,
        expression_scale="log2(TPM+0.01)",
        verification_status="GEO sample title verified"
    )


# 4. Dataset 2: GSE255958

second_meta = pd.read_csv(
    processed_dir / "GSE255958_PC9_metadata.csv"
)

for _, row in second_meta.iterrows():

    condition = row["condition"]

    if condition not in ["DMSO", "osi_DTP"]:
        continue

    is_control = condition == "DMSO"

    add_sample(
        study="GSE255958",
        sample_id=row["sample_id"],
        gsm=row["sample_id"],
        cell_line=row["cell_line"],
        original_condition=(
            "ctrl_48h" if is_control else "Osim_9d"
        ),
        analysis_label=(
            "Control" if is_control else "DTP"
        ),
        control_type="DMSO" if is_control else "",
        treatment_days=0 if is_control else 9,
        control_exposure_hours=48 if is_control else None,
        expression_scale="log2(TPM+0.01)",
        verification_status="GEO sample ID verified"
    )

# 5. Dataset 3: GSE249721

third_meta = pd.read_csv(
    processed_dir / "GSE249721_HCC827_metadata.csv"
)

third_gsm = {
    **{
        f"HCC827Osimertinib11d{i}": f"GSM{7961981+i}"
        for i in range(1, 4)
    },
    **{
        f"HCC827Untreated{i}": f"GSM{7961984+i}"
        for i in range(1, 4)
    }
}

for _, row in third_meta.iterrows():

    sample_id = row["sample_id"]
    is_control = "Untreated" in sample_id

    if sample_id not in third_gsm:
        raise ValueError(
            f"Unrecognized GSE249721 sample: {sample_id}"
        )

    add_sample(
        study="GSE249721",
        sample_id=sample_id,
        gsm=third_gsm[sample_id],
        cell_line="HCC827",
        original_condition=(
            "Untreated" if is_control
            else "Osimertinib11d"
        ),
        analysis_label=(
            "Control" if is_control else "DTP"
        ),
        control_type=(
            "DMSO (series design; sample named Untreated)"
            if is_control else ""
        ),
        treatment_days=0 if is_control else 11,
        control_exposure_hours=None,
        expression_scale="processed scale not verified",
        verification_status="GEO sample ID verified"
    )

# 6. Dataset 4: GSE285632

fourth_meta = pd.read_csv(
    processed_dir / "GSE285632_metadata.csv"
)

for _, row in fourth_meta.iterrows():

    original_condition = row["original_condition"]
    is_control = original_condition == "Parental"

    if original_condition not in ["Parental", "DTP"]:
        raise ValueError(
            f"Unexpected condition: {original_condition}"
        )

    add_sample(
        study="GSE285632",
        sample_id=row["sample_id"],
        gsm="",
        cell_line=row["cell_line"],
        original_condition=original_condition,
        analysis_label=(
            "Control" if is_control else "DTP"
        ),
        control_type="Parental" if is_control else "",
        treatment_days=None,
        control_exposure_hours=None,
        expression_scale="Excel scale not verified",
        verification_status="P/D to GSM mapping NOT verified"
    )

# 7. Validate audit table

audit_df = pd.DataFrame(audit_rows)

# Check duplicate sample identifiers
if audit_df.duplicated(
    subset=["study", "sample_id"]
).any():
    raise ValueError("Duplicate sample identifiers.")

# Check total number of samples
if len(audit_df) != 42:
    raise ValueError(
        f"Expected 42 samples, found {len(audit_df)}"
    )

# Check analysis labels
expected_labels = {"Control", "DTP"}

if set(audit_df["analysis_label"]) != expected_labels:
    raise ValueError("Unexpected analysis labels.")

# Check class distribution
counts = audit_df["analysis_label"].value_counts()

if counts.to_dict() != {"Control": 21, "DTP": 21}:
    raise ValueError(
        f"Unexpected class distribution: {counts.to_dict()}"
    )

# Check sample balance in each study
study_counts = pd.crosstab(
    audit_df["study"],
    audit_df["analysis_label"]
)

expected_study_counts = {
    "GSE193258": 12,
    "GSE255958": 3,
    "GSE249721": 3,
    "GSE285632": 3
}

for study, expected in expected_study_counts.items():
    if (
        study_counts.loc[study, "Control"] != expected
        or study_counts.loc[study, "DTP"] != expected
    ):
        raise ValueError(
            f"Unexpected class counts in {study}"
        )

# Primary analysis eligibility

audit_df["primary_analysis_eligible"] = (
    audit_df["study"] != "GSE285632"
)

audit_df["exclusion_reason"] = ""

audit_df.loc[
    audit_df["study"] == "GSE285632",
    "exclusion_reason"
] = "P/D sample-to-GSM mapping not verified"

print("\nPrimary analysis eligibility:")
print(
    audit_df["primary_analysis_eligible"]
    .value_counts()
)

assert audit_df["primary_analysis_eligible"].sum() == 36

# 8. Save audit table

audit_path = results_dir / "metadata_audit.csv"

audit_df.to_csv(
    audit_path,
    index=False
)

# 9. Print audit summary


print("\nMetadata audit completed.")
print("Total samples:", len(audit_df))

print("\nSamples by study and analysis label:")
print(study_counts)

print("\nOriginal conditions:")
print(pd.crosstab(
    audit_df["study"],
    audit_df["original_condition"]
))

print("\nVerification status:")
print(
    audit_df.groupby(
        ["study", "verification_status"]
    ).size().to_string()
)

print("\nMissing GSM identifiers:")

missing_gsm = (
    audit_df["gsm"].isna()
    | audit_df["gsm"].eq("")
)

print(
    audit_df.assign(missing_gsm=missing_gsm)
    .groupby("study")["missing_gsm"]
    .sum()
)

print("\nSaved:", audit_path)
