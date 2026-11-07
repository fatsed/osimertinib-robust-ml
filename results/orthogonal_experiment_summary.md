# Orthogonal Contrast — exploratory result record

Binary task: osi_acute versus osi_DTP; sample-wise rank input.
All selection and classifier fitting were restricted to each training fold.

## Internal leave-one-cell-line-out (mean across four cell lines)

```text
                             accuracy  macro_f1  dtp_recall
n_genes method                                             
20      Orthogonal_Contrast    0.6667    0.5729      0.5833
        Pooled_Contrast        0.7500    0.6667      0.7500
50      Orthogonal_Contrast    0.7917    0.7396      0.7500
        Pooled_Contrast        0.7500    0.6667      0.7500
```

## External development (GSE255958 PC9; previously used in development)

```text
 n_genes              method  accuracy  macro_f1  dtp_recall
      20     Pooled_Contrast       0.5  0.333333         0.0
      20 Orthogonal_Contrast       0.5  0.333333         0.0
      50     Pooled_Contrast       0.5  0.333333         0.0
      50 Orthogonal_Contrast       0.5  0.333333         0.0
```

## Selected-gene Jaccard overlap at 50 genes

```text
held_out  selected_jaccard_between_methods
   H1975                          0.754386
 HCC2935                          0.587302
  HCC827                          0.612903
     PC9                          0.612903
```

## Interpretation and limitations

- Internal 50-gene performance differs by one correct prediction across 24 held-out samples.
- No recovery of DTP recall on the external-development cohort was observed.
- The external-development cohort is NOT an untouched test set.
- Small sample sizes and heterogeneous exposure protocols limit generalization.
- No demonstrated superiority or novelty claim is supported by these results.

Figures: `figure_orthogonal_loco_counts.png`, `figure_orthogonal_internal_f1.png`.