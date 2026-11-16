# Experimental results (automatically generated)
The results below come from the dataset actually loaded by the notebook. Values are descriptive.

## Reference baseline
ROC-AUC = 0.7745

## Stability by training sample size
- 20%: Spearman mean=0.8772 (range 0.8182 to 0.9595); top-3=1.000, top-5=0.680.
- 40%: Spearman mean=0.9476 (range 0.9101 to 0.9733); top-3=1.000, top-5=0.860.
- 60%: Spearman mean=0.9624 (range 0.9427 to 0.9822); top-3=1.000, top-5=0.880.
- 80%: Spearman mean=0.9743 (range 0.9664 to 0.9872); top-3=1.000, top-5=0.880.
- 100%: Spearman mean=1.0000 (range 1.0000 to 1.0000); top-3=1.000, top-5=1.000.

Least-stable pair: 20pct_seed23 / 20pct_seed53, Spearman=0.8182.

## Required interpretation
Discuss trend, spread, unexpected cases, model AUC, correlated predictors, small repetition count, model/dataset specificity, and SHAP assumptions.
The 50 within-size pairwise comparisons are not independent observations; 100% controls are identical by design.
Only one fixed 200-observation (or configured-size) explanation sample was used.
20 distinct subset models plus one baseline were trained; 100% control rows reuse baseline results.
LLM/code assistance must be disclosed and outputs verified by execution and manual checks.
