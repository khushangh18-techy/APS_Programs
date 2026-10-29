"""
Applying Probability and Statistics to Evaluate a Machine Learning Model

Dataset: Breast Cancer Wisconsin Diagnostic dataset bundled with scikit-learn.
Baseline: Decision Tree
Alternative model: Random Forest ensemble
Evaluation: Same stratified held-out test set, paired bootstrap CI, exact McNemar test.

Run:
    pip install numpy scipy scikit-learn
    python evaluate_ml_models.py
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.25
BOOTSTRAP_REPLICATES = 10_000
ALPHA = 0.05


def paired_bootstrap_ci(correct_baseline, correct_alternative, n_bootstrap=10_000,
                        confidence=0.95, random_state=42):
    """Percentile bootstrap CI for paired accuracy difference (alternative - baseline)."""
    rng = np.random.default_rng(random_state)
    n = len(correct_baseline)
    paired_differences = correct_alternative.astype(float) - correct_baseline.astype(float)
    boot_differences = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_differences[i] = paired_differences[indices].mean()
    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(boot_differences, [tail, 1.0 - tail]))


def exact_mcnemar_test(y_true, pred_baseline, pred_alternative):
    """Exact two-sided McNemar test based on discordant paired predictions."""
    baseline_correct = pred_baseline == y_true
    alternative_correct = pred_alternative == y_true
    b = int(np.sum(baseline_correct & ~alternative_correct))
    c = int(np.sum(~baseline_correct & alternative_correct))
    if b + c == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(b, c), n=b + c, p=0.5,
                                  alternative="two-sided").pvalue)
    return b, c, p_value


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
    alternative = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=1
    )

    baseline.fit(X_train, y_train)
    alternative.fit(X_train, y_train)

    pred_baseline = baseline.predict(X_test)
    pred_alternative = alternative.predict(X_test)

    acc_baseline = accuracy_score(y_test, pred_baseline)
    acc_alternative = accuracy_score(y_test, pred_alternative)
    difference = acc_alternative - acc_baseline

    correct_baseline = pred_baseline == y_test
    correct_alternative = pred_alternative == y_test
    ci_low, ci_high = paired_bootstrap_ci(
        correct_baseline, correct_alternative,
        n_bootstrap=BOOTSTRAP_REPLICATES, confidence=0.95,
        random_state=RANDOM_STATE
    )
    b, c, p_value = exact_mcnemar_test(y_test, pred_baseline, pred_alternative)

    print("=" * 72)
    print("PROBABILITY AND STATISTICS FOR ML MODEL EVALUATION")
    print("=" * 72)
    print(f"Dataset: {data.DESCR.splitlines()[0]}")
    print(f"Total samples: {len(y)}; features: {X.shape[1]}")
    print(f"Train samples: {len(y_train)}; held-out test samples: {len(y_test)}")
    print(f"Random state: {RANDOM_STATE}; test size: {TEST_SIZE:.0%}")
    print()
    print(f"Baseline (Decision Tree) accuracy: {acc_baseline:.4f} ({acc_baseline:.2%})")
    print(f"Alternative (Random Forest) accuracy: {acc_alternative:.4f} ({acc_alternative:.2%})")
    print(f"Observed accuracy difference: {difference:+.4f} ({difference:+.2%})")
    print(f"95% paired bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print()
    print("Exact McNemar test (paired predictions)")
    print(f"Baseline correct / alternative wrong (b): {b}")
    print(f"Baseline wrong / alternative correct (c): {c}")
    print(f"Two-sided p-value: {p_value:.4f}")
    print(f"Significance level alpha: {ALPHA:.2f}")

    print("\\nConfusion matrix — baseline (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, pred_baseline))
    print("\\nConfusion matrix — alternative (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, pred_alternative))
    print("\\nClassification report — baseline:")
    print(classification_report(y_test, pred_baseline, target_names=data.target_names, zero_division=0))
    print("Classification report — alternative:")
    print(classification_report(y_test, pred_alternative, target_names=data.target_names, zero_division=0))

    print("\\nSTATISTICAL DECISION")
    if p_value < ALPHA and ci_low > 0:
        print("Evidence supports an improvement at the 5% significance level.")
    else:
        print("Random Forest has higher observed accuracy, but evidence is not strong")
        print("enough to establish a reliable improvement at the 5% significance level.")
        print("The bootstrap interval includes zero and/or McNemar's p-value is not below alpha.")
        print("Do not claim statistical significance.")

    print("\\nThe test set is a sample, so its result has sampling uncertainty.")
    print("The bootstrap CI quantifies uncertainty in the paired accuracy difference;")
    print("McNemar's test compares discordant paired predictions.")


if __name__ == "__main__":
    main()
