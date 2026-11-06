from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, log_loss
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

CLASS_NAMES = ["DMSO", "osi_DTP"]
GENE_COUNTS = [5, 10, 20, 50]
C = 1.0
DRO_ITERATIONS = 12
DRO_ETA = 0.5


def load_first_study():
    expression = pd.read_csv(
        RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
        sep="\t", index_col=0
    ).T
    names = expression.index.to_series()
    metadata = pd.DataFrame({
        "study": "GSE193258",
        "cell_line": names.str.split("_").str[0],
        "condition": (
            names.str.split("_", n=1).str[1]
            .str.rsplit("_", n=1).str[0]
        ),
    }, index=expression.index)
    metadata = metadata.loc[metadata["condition"].isin(CLASS_NAMES)]
    return expression.loc[metadata.index], metadata


def load_processed(expression_file, metadata_file):
    expression = pd.read_csv(PROCESSED / expression_file, index_col=0)
    metadata = pd.read_csv(PROCESSED / metadata_file).set_index("sample_id")
    metadata = metadata.loc[metadata["condition"].isin(CLASS_NAMES)]
    if not metadata.index.is_unique or not expression.index.is_unique:
        raise ValueError("Repeated sample IDs")
    return expression.loc[metadata.index], metadata


def load_data():
    parts = [
        load_first_study(),
        load_processed("GSE255958_PC9_expression.csv.gz", "GSE255958_PC9_metadata.csv"),
        load_processed("GSE249721_HCC827_original_values.csv.gz", "GSE249721_HCC827_metadata.csv"),
    ]
    genes = sorted(set.intersection(*(set(x.columns) for x, _ in parts)))
    if len(genes) < max(GENE_COUNTS):
        raise ValueError("Too few common genes")
    xs, metas = [], []
    for expression, metadata in parts:
        expression = expression.loc[:, genes].copy()
        metadata = metadata.copy()
        ids = [f"{study}::{sample}" for study, sample in zip(metadata["study"], metadata.index)]
        expression.index = ids
        metadata.index = ids
        xs.append(expression)
        metas.append(metadata)
    x, meta = pd.concat(xs), pd.concat(metas)
    if not x.index.equals(meta.index) or x.isna().any().any():
        raise ValueError("Missing expression values or metadata misalignment")
    if len(x) != 36 or meta["study"].nunique() != 3:
        raise ValueError("Unexpected study/sample count")
    # Each sample is rank transformed independently; no fitted test parameters.
    return x.rank(axis=1, method="average", pct=True), meta


def feature_scores(x_train, meta_train):
    y = meta_train["condition"]
    pooled = (
        x_train.loc[y == "osi_DTP"].mean()
        - x_train.loc[y == "DMSO"].mean()
    )
    by_study = {}
    for study, group in meta_train.groupby("study"):
        per_cell_line = []
        for _, cell_group in group.groupby("cell_line"):
            yi = cell_group["condition"]
            if set(yi.unique()) != set(CLASS_NAMES):
                raise ValueError(f"Missing condition in training group {study}")
            per_cell_line.append(
                x_train.loc[yi.index[yi == "osi_DTP"]].mean()
                - x_train.loc[yi.index[yi == "DMSO"]].mean()
            )
        by_study[study] = pd.concat(per_cell_line, axis=1).mean(axis=1)
    study_balanced = pd.DataFrame(by_study).mean(axis=1)
    return {"Pooled Effect": pooled.abs(), "Study Balanced": study_balanced.abs()}


def compute_groups(meta_train):
    groups = (meta_train["study"].astype(str) + "|" + meta_train["cell_line"].astype(str)).to_numpy()
    group_names, group_index = np.unique(groups, return_inverse=True)
    counts = np.bincount(group_index)
    return group_names, group_index, counts


def fit_classifier(x_train, y_train, group_index, counts, training_method):
    """Fit ERM, equally weighted groups, or exponential Group DRO.

    All methods use the SAME scaler, features, LR penalty and regularization.
    Group DRO updates weights based on within-training-group logistic loss.
    """
    scaler = StandardScaler().fit(x_train)
    z = scaler.transform(x_train)
    group_count = len(counts)
    q = np.full(group_count, 1.0 / group_count)

    def train_with_weights(sample_weights):
        clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", random_state=42)
        clf.fit(z, y_train, sample_weight=sample_weights)
        return clf

    if training_method == "ERM":
        model = train_with_weights(np.ones(len(y_train)))
    elif training_method == "Group Balanced":
        weights = (q[group_index] / counts[group_index]) * len(y_train)
        model = train_with_weights(weights)
    elif training_method == "Group DRO":
        y_binary = (np.asarray(y_train) == "osi_DTP").astype(float)
        for _ in range(DRO_ITERATIONS):
            weights = (q[group_index] / counts[group_index]) * len(y_train)
            model = train_with_weights(weights)
            p = model.predict_proba(z)[:, list(model.classes_).index("osi_DTP")]
            p = np.clip(p, 1e-9, 1 - 1e-9)
            losses = -(y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p))
            group_losses = np.bincount(group_index, weights=losses) / counts
            log_q = np.log(q) + DRO_ETA * group_losses
            log_q -= log_q.max()
            q = np.exp(log_q)
            q /= q.sum()
        # Refit with the final group weights, not the previous iteration weights.
        model = train_with_weights((q[group_index] / counts[group_index]) * len(y_train))
    else:
        raise ValueError(f"Unknown training method: {training_method}")
    return scaler, model, q


def main():
    x, meta = load_data()
    print("Samples:", len(x), "Genes:", x.shape[1])
    print(pd.crosstab(meta["study"], meta["condition"]))
    metrics, predictions = [], []

    for test_study in sorted(meta["study"].unique()):
        is_train = meta["study"] != test_study
        x_train, x_test = x.loc[is_train], x.loc[~is_train]
        meta_train, meta_test = meta.loc[is_train], meta.loc[~is_train]
        y_train = meta_train["condition"].to_numpy()
        y_test = meta_test["condition"].to_numpy()
        group_names, group_index, counts = compute_groups(meta_train)
        scores = feature_scores(x_train, meta_train)

        print(f"\nHeld-out study: {test_study} | training groups: {list(group_names)}")
        for n_genes in GENE_COUNTS:
            for selection_name, score in scores.items():
                genes = score.nlargest(n_genes).index.tolist()
                train_array = x_train[genes].to_numpy()
                test_array = x_test[genes].to_numpy()
                for training_method in ["ERM", "Group Balanced", "Group DRO"]:
                    scaler, model, group_weights = fit_classifier(
                        train_array, y_train, group_index, counts, training_method
                    )
                    z_test = scaler.transform(test_array)
                    predicted = model.predict(z_test)
                    prob_dtp = model.predict_proba(z_test)[:, list(model.classes_).index("osi_DTP")]
                    metrics.append({
                        "test_study": test_study,
                        "n_test": len(y_test),
                        "n_genes": n_genes,
                        "feature_method": selection_name,
                        "training_method": training_method,
                        "accuracy": accuracy_score(y_test, predicted),
                        "macro_f1": f1_score(y_test, predicted, labels=CLASS_NAMES, average="macro", zero_division=0),
                        "dtp_recall": recall_score(y_test, predicted, pos_label="osi_DTP", zero_division=0),
                        "min_training_group_weight": group_weights.min(),
                        "max_training_group_weight": group_weights.max(),
                    })
                    for sid, true, pred, prob in zip(x_test.index, y_test, predicted, prob_dtp):
                        predictions.append({
                            "test_study": test_study, "n_genes": n_genes,
                            "feature_method": selection_name, "training_method": training_method,
                            "sample_id": sid, "actual": true, "predicted": pred,
                            "dtp_probability": prob,
                        })

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(RESULTS / "group_robust_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(RESULTS / "group_robust_predictions.csv", index=False)
    print("\nMean across held-out studies (unweighted study mean):")
    print(metric_df.groupby(["n_genes", "feature_method", "training_method"])[
        ["accuracy", "macro_f1", "dtp_recall"]
    ].mean().round(3).to_string())
    print("\nSaved results/group_robust_metrics.csv and results/group_robust_predictions.csv")
    print("Exploratory comparison only: no claim of novel Group DRO or independent final validation.")


if __name__ == "__main__":
    main()
