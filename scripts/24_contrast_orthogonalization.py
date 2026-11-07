from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

CLASSES = ("DMSO", "osi_acute", "osi_DTP")
GENE_COUNTS = (20, 50)  # Fixed before evaluation; no holdout-based tuning.


def load_data():
    first = pd.read_csv(
        RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
        sep="\t", index_col=0,
    ).T
    names = first.index.to_series()
    y = names.str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0]
    lines = names.str.split("_").str[0]
    keep = y.isin(CLASSES)
    first, y, lines = first.loc[keep], y.loc[keep], lines.loc[keep]

    second = pd.read_csv(
        PROCESSED / "GSE255958_PC9_expression.csv.gz", index_col=0,
    )
    meta = pd.read_csv(
        PROCESSED / "GSE255958_PC9_metadata.csv",
    ).set_index("sample_id")
    meta = meta.loc[meta["condition"].isin(CLASSES)]
    second = second.loc[meta.index]
    y_second = meta["condition"]

    common = sorted(set(first.columns) & set(second.columns))
    if len(common) < max(GENE_COUNTS):
        raise ValueError("Not enough shared genes")
    first, second = first.loc[:, common], second.loc[:, common]
    if first.columns.has_duplicates or second.columns.has_duplicates:
        raise ValueError("Duplicate gene symbols")
    if first.index.has_duplicates or second.index.has_duplicates:
        raise ValueError("Duplicate sample names")
    if not np.isfinite(first.to_numpy(dtype=float)).all() or not np.isfinite(
        second.to_numpy(dtype=float)
    ).all():
        raise ValueError("Nonfinite expression values")
    if not first.index.equals(y.index) or not second.index.equals(y_second.index):
        raise ValueError("Misaligned sample metadata")
    if first.shape != (36, len(common)) or second.shape != (9, len(common)):
        raise ValueError("Unexpected sample count")
    if set(y.unique()) != set(CLASSES) or set(y_second.unique()) != set(CLASSES):
        raise ValueError("Missing treatment state")

    # This transform uses one sample at a time; no fit on test data.
    first = first.rank(axis=1, pct=True, method="average")
    second = second.rank(axis=1, pct=True, method="average")
    return first, y, lines, second, y_second


def compute_contrasts(X_train, y_train, line_train):
    """Equal-weight per-cell-line treatment effects, train only.

    g = mean_line[Acute - Control]: general drug response.
    p = mean_line[DTP - Acute]: putative persister-specific contrast.
    residual = p - projection of p onto g.
    Orthogonality is in gene-effect space; it does NOT correct batch effects.
    """
    usable = X_train.columns[X_train.var(axis=0) > 0]
    X = X_train.loc[:, usable]
    acute_effects, dtp_effects = [], []
    for line in sorted(line_train.unique()):
        local = X.loc[line_train == line]
        labels = y_train.loc[local.index]
        if not all((labels == label).sum() >= 2 for label in CLASSES):
            raise ValueError(f"Not enough training samples for {line}")
        control_mean = local.loc[labels == "DMSO"].mean(axis=0)
        acute_mean = local.loc[labels == "osi_acute"].mean(axis=0)
        dtp_mean = local.loc[labels == "osi_DTP"].mean(axis=0)
        acute_effects.append((acute_mean - control_mean).to_numpy())
        dtp_effects.append((dtp_mean - acute_mean).to_numpy())

    g = np.mean(np.stack(acute_effects), axis=0)
    p = np.mean(np.stack(dtp_effects), axis=0)
    denom = float(np.dot(g, g))
    if denom < 1e-15:
        raise ValueError("Training Acute-Control effect direction is negligible")
    coefficient = float(np.dot(p, g) / denom)
    residual = p - coefficient * g
    cos = float(np.dot(p, g) / (np.linalg.norm(p) * np.linalg.norm(g)))
    fraction = float(np.linalg.norm(residual) / np.linalg.norm(p))

    return (
        pd.Series(p, index=usable),
        pd.Series(residual, index=usable),
        {"acute_dtp_cosine": cos, "projection_coefficient": coefficient,
         "residual_to_original_norm": fraction,
         "acute_control_norm": float(np.linalg.norm(g)),
         "dtp_acute_norm": float(np.linalg.norm(p))},
    )


def run_fold(scope, held_out, X_train, y_train, line_train, X_test, y_test):
    pooled, orthogonal, diag = compute_contrasts(X_train, y_train, line_train)
    feature_scores = {
        "Pooled_Contrast": pooled.abs(),
        "Orthogonal_Contrast": orthogonal.abs(),
    }
    rows, predictions, genes = [], [], []
    for n_genes in GENE_COUNTS:
        selected = {
            method: scores.sort_values(ascending=False, kind="stable")
                          .head(n_genes).index.tolist()
            for method, scores in feature_scores.items()
        }
        jaccard = len(set(selected["Pooled_Contrast"]) &
                      set(selected["Orthogonal_Contrast"])) / len(
                          set(selected["Pooled_Contrast"]) |
                          set(selected["Orthogonal_Contrast"])
                      )
        for method, selection in selected.items():
            # Controls contribute to contrast estimation only.
            is_treated = y_train.isin(("osi_acute", "osi_DTP"))
            X_binary = X_train.loc[is_treated, selection]
            y_binary = y_train.loc[is_treated]
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, random_state=42),
            )
            model.fit(X_binary, y_binary)
            pred = model.predict(X_test.loc[:, selection])
            dtp_column = list(model.classes_).index("osi_DTP")
            probability = model.predict_proba(X_test.loc[:, selection])[:, dtp_column]
            # This is the binary task; discard external/LOCO controls at scoring.
            label_mask = y_test.isin(("osi_acute", "osi_DTP"))
            if not label_mask.all():
                raise ValueError("Testing data should contain treated states only")
            rows.append({
                "scope": scope, "held_out": held_out,
                "method": method, "n_genes": n_genes,
                "n_train_binary": len(y_binary), "n_train_control": int((y_train == "DMSO").sum()),
                "n_test": len(y_test),
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, labels=["osi_acute", "osi_DTP"],
                                     average="macro", zero_division=0),
                "dtp_recall": recall_score(y_test, pred, pos_label="osi_DTP", zero_division=0),
                "selected_jaccard_between_methods": jaccard,
                **diag,
            })
            for sid, truth, predicted, proba in zip(X_test.index, y_test, pred, probability):
                predictions.append({
                    "scope": scope, "held_out": held_out, "n_genes": n_genes,
                    "method": method, "sample_id": sid,
                    "actual": truth, "predicted": predicted, "dtp_probability": float(proba),
                })
            for i, gene in enumerate(selection, start=1):
                genes.append({
                    "scope": scope, "held_out": held_out, "n_genes": n_genes,
                    "method": method, "rank": i, "gene": gene,
                    "pooled_effect": float(pooled[gene]),
                    "orthogonal_effect": float(orthogonal[gene]),
                })
    return rows, predictions, genes


def main():
    first, y, lines, second, y_second = load_data()
    print("Shared genes:", first.shape[1])
    print("Internal three-state samples:", len(first))
    print("External three-state samples:", len(second))
    print("\nInternal class composition:\n", pd.crosstab(lines, y))

    metrics, predictions, genes = [], [], []
    for held_out in sorted(lines.unique()):
        train_mask = lines != held_out
        test_mask = ~train_mask
        treated_test = test_mask & y.isin(("osi_acute", "osi_DTP"))
        result = run_fold(
            "Internal_LOCO", held_out,
            first.loc[train_mask], y.loc[train_mask], lines.loc[train_mask],
            first.loc[treated_test], y.loc[treated_test],
        )
        for target, portion in zip((metrics, predictions, genes), result):
            target.extend(portion)

    # External-development study already informed earlier choices; NOT untouched.
    train_mask = lines != "PC9"
    second_treated = y_second.isin(("osi_acute", "osi_DTP"))
    result = run_fold(
        "External_development", "GSE255958_PC9",
        first.loc[train_mask], y.loc[train_mask], lines.loc[train_mask],
        second.loc[second_treated], y_second.loc[second_treated],
    )
    for target, portion in zip((metrics, predictions, genes), result):
        target.extend(portion)

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(RESULTS / "orthogonal_contrast_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        RESULTS / "orthogonal_contrast_predictions.csv", index=False,
    )
    pd.DataFrame(genes).to_csv(
        RESULTS / "orthogonal_contrast_genes.csv", index=False,
    )

    print("\nINTERNAL LOCO (unweighted average of four cell lines):")
    print(metric_df.loc[metric_df.scope == "Internal_LOCO"].groupby(
        ["n_genes", "method"]
    )[["accuracy", "macro_f1", "dtp_recall"]].mean().round(3).to_string())

    print("\nEXTERNAL DEVELOPMENT (previously seen, NOT independent final validation):")
    print(metric_df.loc[metric_df.scope == "External_development", [
        "n_genes", "method", "accuracy", "macro_f1", "dtp_recall",
    ]].round(3).to_string(index=False))

    print("\nDIAGNOSTICS (feature-selection effect space, train-only):")
    print(metric_df.drop_duplicates(["scope", "held_out"])[[
        "scope", "held_out", "acute_dtp_cosine", "residual_to_original_norm",
    ]].round(3).to_string(index=False))
    print("\nExploratory results only. Orthogonalization changes gene ranking, not expression matrices.")


if __name__ == "__main__":
    main()
