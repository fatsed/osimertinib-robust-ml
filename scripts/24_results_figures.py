from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Subdued journal-style tones, consistent across all generated figures.
INK = "#243747"
BLUE = "#49677D"
SLATE = "#899AA6"
PALE = "#BFCBD4"
LAVENDER = "#8883A4"
GRID = "#DFE4E8"
PALETTE = [BLUE, SLATE, LAVENDER, PALE]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.0,
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK,
    "svg.fonttype": "none",  # keep vector text editable
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

REQUIRED_FILES = {
    "01_study_composition": ["metadata_audit.csv"],
    "02_three_state_baselines": ["cross_study_metrics.csv"],
    "03_three_state_LOCO": ["three_state_loco_metrics.csv"],
    "04_control_DTP_LOSO": ["loso_three_studies_metrics.csv"],
    "05_acute_DTP_LOCO_F1": ["contrast_stability_metrics.csv"],
    "06_training_contrast_dispersion": ["contrast_stability_metrics.csv"],
    "07_replicate_reliability": ["contrast_reliability_summary.csv"],
    "08_cross_study_contrast": ["qc_cross_study_contrasts.csv"],
    "09_gene_count_sensitivity": ["gene_count_sensitivity_summary.csv"],
    "10_permutation_test": ["permutation_scores.csv", "permutation_summary.csv"],
}


def frame(results_dir: Path, filename: str, required: list[str]) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding experiment first.")
    table = pd.read_csv(path)
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{filename} missing expected columns: {missing}")
    if table.empty:
        raise ValueError(f"{filename} is empty")
    return table


def valid_unit_interval(table: pd.DataFrame, fields: list[str], source: str) -> None:
    for name in fields:
        values = pd.to_numeric(table[name], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"{source}: {name} must contain finite values in [0,1]")


def decorate(ax: plt.Axes, *, y_range: tuple[float, float] | None = (0, 1)) -> None:
    if y_range is not None:
        ax.set_ylim(*y_range)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, alpha=0.85, linewidth=0.55)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def fig01(root: Path, out: Path) -> str:
    df = frame(root, "metadata_audit.csv", ["study", "analysis_label", "primary_analysis_eligible"])
    eligibility = df["primary_analysis_eligible"].astype(str).str.lower() == "true"
    df = df.loc[eligibility].copy()
    if "GSE285632" in set(df["study"]):
        raise ValueError("Excluded GSE285632 marked as primary eligible in metadata_audit.csv")
    if set(df["analysis_label"]) != {"Control", "DTP"}:
        raise ValueError("Unexpected labels in metadata_audit.csv")
    tally = pd.crosstab(df["study"], df["analysis_label"])
    tally = tally.reindex(columns=["Control", "DTP"], fill_value=0)
    tally = tally.reindex([s for s in ["GSE193258", "GSE255958", "GSE249721"] if s in tally.index])
    y = np.arange(len(tally))
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    left = np.zeros(len(tally))
    for cls, color in zip(["Control", "DTP"], [SLATE, BLUE]):
        vals = tally[cls].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=color, edgecolor="white", linewidth=0.6, label=cls, height=0.58)
        for i, (val, base) in enumerate(zip(vals, left)):
            if val:
                ax.text(base + val / 2, i, str(int(val)), ha="center", va="center", color="white", fontsize=9)
        left += vals
    ax.set_yticks(y, tally.index.to_list())
    ax.invert_yaxis()
    ax.set_xlabel("Selected samples")
    ax.set_title("Verified primary cohorts: control and DTP samples")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, alpha=0.6, linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, out, "01_study_composition")
    return "Primary dataset composition only; GSE285632 intentionally excluded."


def fig02(root: Path, out: Path) -> str:
    df = frame(root, "cross_study_metrics.csv", ["model", "accuracy", "macro_f1"])
    valid_unit_interval(df, ["accuracy", "macro_f1"], "cross_study_metrics.csv")
    x = np.arange(len(df)); width = 0.34
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    for dx, field, label, color in [(-width/2, "accuracy", "Accuracy", BLUE), (width/2, "macro_f1", "Macro-F1", SLATE)]:
        ax.bar(x + dx, df[field], width=width, label=label, color=color)
    ax.set_xticks(x, df["model"].tolist(), rotation=10)
    ax.set_ylabel("Score")
    ax.set_title("Three-state transfer: GSE193258 to GSE255958 (n = 9)")
    ax.legend(frameon=False, ncol=2)
    decorate(ax)
    save(fig, out, "02_three_state_baselines")
    return "Earlier three-state cross-study development comparison, only nine test samples."


def fig03(root: Path, out: Path) -> str:
    df = frame(root, "three_state_loco_metrics.csv", ["test_cell_line", "n_genes", "macro_f1"])
    valid_unit_interval(df, ["macro_f1"], "three_state_loco_metrics.csv")
    lines = sorted(df["test_cell_line"].unique())
    fig, ax = plt.subplots(figsize=(7.3, 3.6))
    for i, line in enumerate(lines):
        sub = df.loc[df["test_cell_line"].eq(line)].sort_values("n_genes")
        ax.plot(sub["n_genes"], sub["macro_f1"], marker=["o","s","D","^"][i%4],
                linewidth=1.65, color=PALETTE[i%4], label=line)
    ax.set_xticks(sorted(df["n_genes"].unique()))
    ax.set_xlabel("Selected genes (training-fold ANOVA)")
    ax.set_ylabel("Held-out macro-F1")
    ax.set_title("Three-state classification across held-out cell lines")
    ax.legend(frameon=False, ncol=4, loc="lower right")
    decorate(ax)
    save(fig, out, "03_three_state_LOCO")
    return "Internal LOCO experiment; each held-out cell line has nine samples."


def fig04(root: Path, out: Path) -> str:
    df = frame(root, "loso_three_studies_metrics.csv", ["test_study", "method", "n_test", "macro_f1"])
    if "GSE285632" in set(df["test_study"]):
        raise ValueError("Do not use four-study LOSO output as verified main analysis")
    valid_unit_interval(df, ["macro_f1"], "loso_three_studies_metrics.csv")
    studies = [s for s in ["GSE193258", "GSE255958", "GSE249721"] if s in set(df["test_study"])]
    methods = [m for m in ["Rank + PCA", "Pooled Effect", "Stable Signature"] if m in set(df["method"])]
    x = np.arange(len(studies)); width = .78 / max(len(methods),1)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, method in enumerate(methods):
        scores = df.loc[df["method"].eq(method)].set_index("test_study").reindex(studies)["macro_f1"]
        if scores.isna().any():
            raise ValueError(f"Missing LOSO test study for {method}")
        ax.bar(x - .39 + width*(i+.5), scores.to_numpy(), width=width*.94,
               color=PALETTE[i], label=method)
    n_by_study = df.drop_duplicates("test_study").set_index("test_study")["n_test"]
    ax.set_xticks(x, [f"{s}\n(n={int(n_by_study[s])})" for s in studies])
    ax.set_ylabel("Held-out macro-F1")
    ax.set_title("Control vs DTP: leave-one-study-out comparisons")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=8)
    decorate(ax)
    save(fig, out, "04_control_DTP_LOSO")
    return "Development-stage LOSO; high scores do not resolve treatment-duration and study confounding."


def _contrast_subset(root: Path) -> pd.DataFrame:
    df = frame(root, "contrast_stability_metrics.csv", ["scope", "held_out", "n_genes", "method", "macro_f1", "train_group_margin_sd"])
    df = df.loc[df["scope"].eq("Internal_LOCO") & df["n_genes"].eq(20)].copy()
    if df.empty:
        raise ValueError("No Internal_LOCO rows for 20 selected genes")
    if not df["train_group_margin_sd"].ge(0).all():
        raise ValueError("train_group_margin_sd contains negative values")
    valid_unit_interval(df, ["macro_f1"], "contrast_stability_metrics.csv")
    return df


def _method_bars(ax: plt.Axes, df: pd.DataFrame, field: str) -> None:
    groups = sorted(df["held_out"].unique()); methods = ["Pooled_Logistic", "Contrast_Stable"]
    x = np.arange(len(groups)); width = .35
    for i, (method, color) in enumerate(zip(methods, [SLATE, BLUE])):
        vals = df.loc[df["method"].eq(method)].set_index("held_out").reindex(groups)[field]
        if vals.isna().any():
            raise ValueError(f"Missing {method} values for one held-out cell line")
        ax.bar(x + (i-.5)*width, vals, width=width*.94, color=color,
               label=method.replace("_", " "))
    ax.set_xticks(x, groups)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)


def fig05(root: Path, out: Path) -> str:
    df = _contrast_subset(root)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _method_bars(ax, df, "macro_f1")
    ax.set_ylabel("Held-out macro-F1")
    ax.set_title("Acute vs DTP: pooled vs contrast-stable logistic (20 genes)")
    decorate(ax)
    save(fig, out, "05_acute_DTP_LOCO_F1")
    return "Within-study LOCO; external development cohort intentionally not included."


def fig06(root: Path, out: Path) -> str:
    df = _contrast_subset(root)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _method_bars(ax, df, "train_group_margin_sd")
    ax.set_ylabel("Training-group margin SD (model score units)")
    ax.set_title("Contrast-margin dispersion across training cell lines (20 genes)")
    decorate(ax, y_range=None)
    save(fig, out, "06_training_contrast_dispersion")
    return "Training-set diagnostic, NOT independent generalization performance."


def fig07(root: Path, out: Path) -> str:
    df = frame(root, "contrast_reliability_summary.csv", ["study", "cell_line", "median_spearman", "median_top50_agreement"])
    valid_unit_interval(df, ["median_top50_agreement"], "contrast_reliability_summary.csv")
    # Spearman can be [-1,1]; check separately
    if not pd.to_numeric(df["median_spearman"], errors="coerce").between(-1,1).all():
        raise ValueError("Spearman correlations outside [-1,1]")
    names = [f"{s}\n{c}" for s,c in zip(df["study"], df["cell_line"])]
    y = np.arange(len(df));
    fig, ax = plt.subplots(figsize=(8.2, 3.75))
    ax.scatter(df["median_spearman"], y - .13, marker="o", s=43, color=BLUE,
               label="Median Spearman (all genes)")
    ax.scatter(df["median_top50_agreement"], y + .13, marker="s", s=38, color=LAVENDER,
               label="Median top-50 direction agreement")
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlim(-.05, 1.05)
    ax.set_xlabel("Correlation / proportion")
    ax.set_title("Acute–DTP replicate-level contrast reliability")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, out, "07_replicate_reliability")
    return "Independent replicate splits (2+2 train, 1+1 held out); not a classification accuracy."


def fig08(root: Path, out: Path) -> str:
    df = frame(root, "qc_cross_study_contrasts.csv", ["comparison", "spearman", "top50_direction_agreement"])
    valid_unit_interval(df, ["top50_direction_agreement"], "qc_cross_study_contrasts.csv")
    if not pd.to_numeric(df["spearman"], errors="coerce").between(-1,1).all():
        raise ValueError("Cross-study Spearman correlations outside [-1,1]")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.1, 3.45))
    ax.barh(y, df["spearman"], height=.55, color=BLUE)
    ax.set_yticks(y, df["comparison"])
    ax.invert_yaxis()
    ax.set_xlim(min(0, float(df["spearman"].min())-.05), max(0.8, float(df["spearman"].max())+.08))
    ax.set_xlabel("Spearman correlation of matched-cell-line treatment contrasts")
    ax.set_title("Exploratory cross-study expression-contrast consistency")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    save(fig, out, "08_cross_study_contrast")
    return "Descriptive only; between-study exposure and control protocols may differ."


def fig09(root: Path, out: Path) -> str:
    df = frame(root, "gene_count_sensitivity_summary.csv", ["n_genes", "method", "macro_f1"])
    valid_unit_interval(df, ["macro_f1"], "gene_count_sensitivity_summary.csv")
    methods = ["Pooled Effect", "Study Balanced", "Balanced + Penalty"]
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    for i, method in enumerate(methods):
        sub = df.loc[df["method"].eq(method)].sort_values("n_genes")
        if sub.empty: continue
        ax.plot(sub["n_genes"], sub["macro_f1"], marker=["o", "s", "^"][i], color=PALETTE[i],
                linewidth=1.55, label=method)
    ax.set_xscale("log"); ax.set_xticks(sorted(df["n_genes"].unique()))
    ax.set_xticklabels([str(int(x)) for x in sorted(df["n_genes"].unique())])
    ax.set_xlabel("Number of selected genes")
    ax.set_ylabel("Average LOSO macro-F1")
    ax.set_title("Sensitivity to the number of selected genes")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    decorate(ax)
    save(fig, out, "09_gene_count_sensitivity")
    return "Three eligible studies only; mean over held-out-study folds."


def fig10(root: Path, out: Path) -> str:
    scores = frame(root, "permutation_scores.csv", ["macro_f1"])
    summary = frame(root, "permutation_summary.csv", ["observed_macro_f1", "empirical_p_value", "n_permutations"])
    valid_unit_interval(scores, ["macro_f1"], "permutation_scores.csv")
    observed = float(summary.iloc[0]["observed_macro_f1"])
    if not 0 <= observed <= 1: raise ValueError("Observed macro-F1 outside [0,1]")
    n = int(summary.iloc[0]["n_permutations"])
    if n != len(scores): raise ValueError(f"Permutation count mismatch: summary says {n}, found {len(scores)}")
    p = float(summary.iloc[0]["empirical_p_value"])
    fig, ax = plt.subplots(figsize=(7,3.5))
    ax.hist(scores["macro_f1"], bins=min(20, max(5,int(np.sqrt(n)))), color=SLATE, edgecolor="white", alpha=.9)
    ax.axvline(observed, color=INK, lw=1.7, ls="--", label=f"Observed F1 = {observed:.3f}")
    ax.set_xlabel("Macro-F1 under within-group label permutation")
    ax.set_ylabel("Number of permutations")
    ax.set_title(f"Control vs DTP: permutation diagnostic (B={n}, empirical p={p:.4f})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.55)
    ax.set_axisbelow(True)
    save(fig, out, "10_permutation_test")
    return "Exploratory permutation diagnostic; does not remove underlying study confounding."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping missing input CSVs")
    args = parser.parse_args()
    root = args.results.resolve()
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    funcs = [fig01,fig02,fig03,fig04,fig05,fig06,fig07,fig08,fig09,fig10]
    manifest = []
    errors = []
    for func in funcs:
        name = func.__name__.replace("fig", "")
        try:
            description = func(root, out)
            manifest.append({"figure":func.__name__,"status":"created","notes":description})
            print(f"[OK] {func.__name__}: {description}")
        except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError) as exc:
            errors.append(f"{func.__name__}: {exc}")
            manifest.append({"figure":func.__name__,"status":"skipped","notes":str(exc)})
            print(f"[SKIP] {func.__name__}: {exc}")
    pd.DataFrame(manifest).to_csv(out / "figure_manifest.csv", index=False)
    print(f"\nFigures saved to {out}")
    if errors and args.strict:
        raise RuntimeError("Missing / invalid data for: " + "; ".join(errors))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
