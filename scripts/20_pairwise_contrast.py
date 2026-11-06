from itertools import combinations
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

STATES = ["osi_acute", "osi_DTP"]
N_GENES = 20
N_CANDIDATES = 60
N_PAIRS = 20
HETEROGENEITY_PENALTY = 0.5  # Exploratory; do not tune against held-out labels.


def load_data():
    first = pd.read_csv(
        RAW / "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
        sep="\t", index_col=0,
    ).T
    sample_names = first.index.to_series()
    y_first = sample_names.str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0]
    lines = sample_names.str.split("_").str[0]
    mask = y_first.isin(STATES)
    first, y_first, lines = first.loc[mask], y_first.loc[mask], lines.loc[mask]

    second = pd.read_csv(
        PROCESSED / "GSE255958_PC9_expression.csv.gz", index_col=0
    )
    second_meta = pd.read_csv(
        PROCESSED / "GSE255958_PC9_metadata.csv"
    ).set_index("sample_id")
    second_meta = second_meta.loc[second_meta["condition"].isin(STATES)]
    second = second.loc[second_meta.index]
    y_second = second_meta["condition"]

    shared = sorted(set(first.columns) & set(second.columns))
    if len(shared) < N_CANDIDATES:
        raise ValueError("Insufficient shared genes")
    first = first.loc[:, shared]
    second = second.loc[:, shared]
    if first.isna().any().any() or second.isna().any().any():
        raise ValueError("Expression data contains missing values")
    if len(first) != 24 or len(second) != 6:
        raise ValueError("Unexpected treated sample counts")
    if not first.index.equals(y_first.index) or not second.index.equals(y_second.index):
        raise ValueError("Samples and metadata are misaligned")

    # Same within-sample rank operation; no fitted transform on held-out samples.
    return (first.rank(axis=1, method="average", pct=True), y_first, lines,
            second.rank(axis=1, method="average", pct=True), y_second)


def model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42),
    )


def make_pair_features(data, pairs):
    # Pair order is invariant to strictly monotone transforms within each sample.
    genes_a = [pair[0] for pair in pairs]
    genes_b = [pair[1] for pair in pairs]
    left = data.loc[:, genes_a].to_numpy()
    right = data.loc[:, genes_b].to_numpy()
    features = (left > right).astype(np.float64)
    return pd.DataFrame(
        features, index=data.index,
        columns=[f"{a}|{b}" for a, b in pairs],
    )


def choose_features(X_train, y_train, line_train):
    # All selections below use training-fold data and labels only.
    contrast = (
        X_train.loc[y_train == "osi_DTP"].mean(axis=0)
        - X_train.loc[y_train == "osi_acute"].mean(axis=0)
    )
    ranked_genes = contrast.abs().sort_values(ascending=False, kind="stable")
    top_genes = ranked_genes.head(N_GENES).index.tolist()
    candidate_genes = ranked_genes.head(N_CANDIDATES).index.tolist()
    candidate_pairs = list(combinations(candidate_genes, 2))
    pair_features = make_pair_features(X_train, candidate_pairs)

    pooled = (
        pair_features.loc[y_train == "osi_DTP"].mean(axis=0)
        - pair_features.loc[y_train == "osi_acute"].mean(axis=0)
    )
    pooled_pairs = pooled.abs().sort_values(ascending=False, kind="stable")

    effects = []
    for line in sorted(line_train.unique()):
        line_ids = line_train.index[line_train == line]
        yy = y_train.loc[line_ids]
        acute_ids = yy.index[yy == "osi_acute"]
        dtp_ids = yy.index[yy == "osi_DTP"]
        if len(acute_ids) != 3 or len(dtp_ids) != 3:
            raise ValueError(f"Unexpected classes in training cell line {line}")
        effects.append(
            pair_features.loc[dtp_ids].mean(axis=0)
            - pair_features.loc[acute_ids].mean(axis=0)
        )
    effects = pd.DataFrame(effects)
    consensus_score = (effects.mean(axis=0).abs()
                       - HETEROGENEITY_PENALTY * effects.std(axis=0, ddof=0))
    consensus_pairs = consensus_score.sort_values(ascending=False, kind="stable")

    def recover(series):
        chosen = series.head(N_PAIRS).index
        position = pair_features.columns.get_indexer(chosen)
        return [candidate_pairs[int(i)] for i in position]

    return {
        "Gene_Rank": top_genes,
        "Pair_Pooled": recover(pooled_pairs),
        "Pair_Consensus": recover(consensus_pairs),
    }


def run_fold(scope, held_out, X_train, y_train, line_train, X_test, y_test):
    selected = choose_features(X_train, y_train, line_train)
    rows, preds, signatures = [], [], []

    for method, features in selected.items():
        if method == "Gene_Rank":
            train_matrix, test_matrix = X_train.loc[:, features], X_test.loc[:, features]
            feature_count = len(features)
            unique_genes = len(set(features))
        else:
            train_matrix = make_pair_features(X_train, features)
            test_matrix = make_pair_features(X_test, features)
            feature_count = len(features)
            unique_genes = len(set(g for pair in features for g in pair))

        clf = model()
        clf.fit(train_matrix, y_train)
        predicted = clf.predict(test_matrix)
        accuracy = accuracy_score(y_test, predicted)
        macro_f1 = f1_score(y_test, predicted, labels=STATES,
                            average="macro", zero_division=0)
        dtp_recall = recall_score(y_test, predicted,
                                  pos_label="osi_DTP", zero_division=0)
        rows.append({
            "scope": scope, "held_out": held_out, "method": method,
            "n_train": len(y_train), "n_test": len(y_test),
            "n_features": feature_count, "n_unique_genes": unique_genes,
            "accuracy": accuracy, "macro_f1": macro_f1, "dtp_recall": dtp_recall,
        })
        for sample, truth, prediction in zip(X_test.index, y_test, predicted):
            preds.append({
                "scope": scope, "held_out": held_out, "method": method,
                "sample_id": sample, "actual": truth, "predicted": prediction,
            })
        for rank, feature in enumerate(features, start=1):
            if method == "Gene_Rank":
                a, b = feature, ""
            else:
                a, b = feature
            signatures.append({
                "scope": scope, "held_out": held_out, "method": method,
                "rank": rank, "gene_a": a, "gene_b": b,
            })
    return rows, preds, signatures


def main():
    X_first, y_first, lines, X_second, y_second = load_data()
    print("Internal treated samples:", X_first.shape)
    print("External treated samples:", X_second.shape)
    print("States per training cell line:\n", pd.crosstab(lines, y_first))

    metrics, predictions, signatures = [], [], []
    for held_out in sorted(lines.unique()):
        is_train = lines != held_out
        out = run_fold(
            "Internal_LOCO", held_out,
            X_first.loc[is_train], y_first.loc[is_train], lines.loc[is_train],
            X_first.loc[~is_train], y_first.loc[~is_train],
        )
        for target, part in zip((metrics, predictions, signatures), out):
            target.extend(part)

    # External study and PC9 cell line are held out of fitting and feature selection.
    # HOWEVER: GSE255958 has previously been examined during development.
    is_train = lines != "PC9"
    out = run_fold(
        "External_development", "GSE255958_PC9",
        X_first.loc[is_train], y_first.loc[is_train], lines.loc[is_train],
        X_second, y_second,
    )
    for target, part in zip((metrics, predictions, signatures), out):
        target.extend(part)

    result = pd.DataFrame(metrics)
    result.to_csv(RESULTS / "pairwise_contrast_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(
        RESULTS / "pairwise_contrast_predictions.csv", index=False,
    )
    pd.DataFrame(signatures).to_csv(
        RESULTS / "pairwise_contrast_features.csv", index=False,
    )

    print("\nInternal LOCO (unweighted average across four cell lines):")
    print(result.loc[result.scope == "Internal_LOCO"].groupby("method")
          [["accuracy", "macro_f1", "dtp_recall", "n_unique_genes"]]
          .mean().round(3).to_string())
    print("\nExternal development (unseen study and PC9 in training):")
    print(result.loc[result.scope == "External_development",
          ["method", "accuracy", "macro_f1", "dtp_recall", "n_unique_genes"]]
          .round(3).to_string(index=False))
    print("\nOnly exploratory results. External study is NOT an untouched test set.")


if __name__ == "__main__":
    main()
