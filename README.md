# SHAP Stability Across Training Samples

**DS5622 Explainable AI — Track 1: Experimental Investigation of Existing XAI Methods**

**Project title:** How Stable Are SHAP Explanations? An Experimental Investigation Across Training Samples

**Team members**
- Chilaka Sri Krishna Sai — 142502010 (M.Tech Data Science)
- Kedika Varshitha Reddy — 142301016 (B.Tech Data Science)

## Research question

**How stable are global SHAP feature-importance rankings for a Random Forest classifier across different training samples and sample sizes?**

**Hypothesis:** Rankings generally become more consistent as the training sample size increases; features with similar importance may change rank.

The project tests explanation *stability*, rather than simply producing SHAP plots.

## Dataset and method

- **Dataset:** [UCI Default of Credit Card Clients (Kaggle mirror)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) — 30,000 customers; 23 predictors; binary next-month default target.
- **Predictive model:** `sklearn.ensemble.RandomForestClassifier` (100 trees, minimum leaf size 2).
- **Explanation method:** SHAP `TreeExplainer`, positive/default class (`class=1`).
- **Global feature importance:** mean absolute SHAP value, computed on the *same fixed 200 test observations* for every model.
- **Data split:** 80% training / 20% held-out test, stratified, split seed 2026.
- **Training sample sizes:** 20%, 40%, 60%, 80%, and 100% of the training split.
- **Subsample seeds:** 11, 23, 37, 53, 71.

For 20%–80%, five seeded stratified subsets are fitted at each size. The five 100% entries reuse the same deterministic baseline model as a **reference/control**: they are *not* five independent training samples. The design uses **21 actual fitted models** (20 subset models + 1 baseline), 25 labeled entries, and 50 within-size pairwise comparisons (10 per size).

## Metrics and evaluation

- **Spearman rank correlation:** agreement between the full 23-feature rankings.
- **Top-5 overlap:** fraction of shared features in two models' five most important features.
- **Top-3 overlap (sensitivity analysis):** checks whether the conclusion changes when only three features are considered.
- **ROC-AUC:** predictive performance on the common held-out test set.
- **SHAP additivity verification:** checks whether base value plus sum of class-1 SHAP values reproduces the predicted default probability on 20 test cases.

All pairwise comparison results are **descriptive**; the 10 pairs within one sample size share the same five models and therefore are not independent observations.

## Results

Results below are from the completed Kaggle experiment on the real dataset (see `shap_stability_20_40_60_80_100/results_summary.md`).

| Training fraction | Mean Spearman | Mean Top-3 overlap | Mean Top-5 overlap | Mean ROC-AUC |
|:---:|---:|---:|---:|---:|
| 20% | 0.8772 | 100% | 68% | 0.7648 |
| 40% | 0.9476 | 100% | 86% | 0.7699 |
| 60% | 0.9624 | 100% | 88% | 0.7717 |
| 80% | 0.9743 | 100% | 88% | 0.7723 |
| 100% (identical control) | 1.0000 | 100% | 100% | 0.7745 |

The full-data baseline ROC-AUC is **0.7745**. Within the independently sampled 20%–80% experiments, average SHAP-ranking agreement increases with training size. The 100% perfect agreement is expected by construction and is not independent support for the hypothesis.

**Additivity verification:** 20 cases checked; maximum absolute reconstruction error approximately `5.04e-13` (pass threshold `1e-5`).

## Figures

The result directory contains four PNG figures:

1. `figure_1_baseline_shap.png` — top 10 global SHAP importances for the baseline.
2. `figure_2_size_vs_spearman.png` — training sample size versus mean rank correlation, with observed ranges (truncated vertical axis labeled on the plot).
3. `figure_3_topk_sensitivity.png` — Top-3 versus Top-5 feature overlap.
4. `figure_4_auc_vs_stability.png` — ROC-AUC compared with explanation stability.

## Repository structure

```text
SHAP-Stability-XAI-Project/
├── README.md
├── xai-project.ipynb
├── requirements.txt                     # recommended dependency file
└── shap_stability_20_40_60_80_100/
    ├── baseline_feature_importance.csv
    ├── model_roc_auc.csv
    ├── pairwise_stability.csv
    ├── stability_summary.csv
    ├── experiment_config.json
    ├── shap_additivity_verification.csv
    ├── shap_additivity_summary.json
    ├── results_summary.md
    ├── figure_1_baseline_shap.png
    ├── figure_2_size_vs_spearman.png
    ├── figure_3_topk_sensitivity.png
    └── figure_4_auc_vs_stability.png
```

## Reproduce the experiment in Kaggle

1. Open [Kaggle Notebooks](https://www.kaggle.com/code) and import `xai-project.ipynb`.
2. Attach the [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) dataset through **Add Input**, or enable Internet access so the notebook can retrieve it with `kagglehub`.
3. Select **CPU** (GPU is not required for this scikit-learn Random Forest + SHAP implementation).
4. Execute **Run All**, starting at the top. The notebook performs the split, baseline fitting, seeded subsampling, explanations, pairwise comparisons, four graphs, verification, and exports.
5. Inspect output files in `/kaggle/working/shap_stability_20_40_60_80_100/`.

**Reproducibility:** the notebook saves seeds, data configuration, hyperparameters, feature names, and package versions to `experiment_config.json`. To rerun on another environment, check `requirements.txt` and the configuration JSON; minor package or platform differences may affect exact numerical results.

### Software versions used for recorded results

- Python `3.13.18`
- NumPy `2.1.3`
- pandas `2.3.3`
- scikit-learn `1.6.1`
- SciPy `1.16.3`
- SHAP `0.54.1`
- Matplotlib `3.10.0`
- KaggleHub is used for dataset retrieval; its version can be checked in the notebook environment.

## Interpretation and limitations

These results suggest that, for this dataset and fixed Random Forest configuration, larger training subsets give more consistent **global** SHAP rankings. The highest-ranked three features were stable across the tested within-size comparisons, while Top-5 overlap was lower, particularly with smaller training subsets.

Limitations include only five seeded subsets per size, shared models among pairwise comparisons, a single dataset and model type, correlated credit variables, and a fixed set of 200 explained test cases. Different SHAP evaluation samples, model settings, or datasets might produce different findings. The observations do not establish causal explanations or guarantee stability of individual customer explanations.

## Use of LLM / AI tools

ChatGPT was used to assist with notebook/code drafting, debugging guidance, analysis organization, and report/README preparation. The work was verified by **executing the notebook on the actual dataset**, inspecting exported metrics and plots, checking the deterministic 100% control, and testing SHAP additivity against model-predicted probabilities for 20 cases. The team remains responsible for reviewing the outputs, interpreting results, and explaining the method during the viva.

## References

1. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. https://arxiv.org/abs/1705.07874
2. Yeh, I.-C., & Lien, C.-h. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480.
3. [SHAP documentation](https://shap.readthedocs.io/)

---
**Course submission:** the public GitHub repository contains code and results; only one final report submission is needed per team, as directed by the instructor.
