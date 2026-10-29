# Probability and Statistics for ML Model Evaluation

This project compares a baseline Decision Tree with a **Random Forest** classifier on the Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn.

## Run locally

```bash
pip install numpy scipy scikit-learn
python evaluate_ml_models.py
```

The script uses a stratified train/test split, evaluates both models on the same held-out test set, estimates a paired bootstrap confidence interval for the accuracy difference, and runs an exact two-sided McNemar test.

## Files

- `evaluate_ml_models.py` — reproducible experiment
- `ML_Model_Statistical_Evaluation_Report.pdf` — report with methodology, observations, decision, limitations, and statistical concepts

## Fixed-seed result

- Decision Tree accuracy: **92.31%** (132/143 correct)
- Random Forest accuracy: **95.80%** (137/143 correct)
- Observed difference: **+3.50 percentage points**
- Paired bootstrap 95% CI: approximately **−0.70 to +7.69 percentage points**
- Exact two-sided McNemar p-value: **0.1797**

Random Forest has higher observed accuracy in this run, but the confidence interval includes zero and the p-value is above 0.05. Therefore, the result is not statistically significant at the 5% level. These results correspond to the fixed settings and package behavior; the test set is small and only one split is evaluated.
