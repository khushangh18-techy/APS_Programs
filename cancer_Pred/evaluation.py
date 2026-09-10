import pandas as pd

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)


def get_probabilities(model, X_test):

    probabilities = model.predict_proba(X_test)
    
    results = pd.DataFrame({
        "P_malignant": probabilities[:, 0],
        "P_benign": probabilities[:, 1]
    })

    return probabilities, results


def evaluate_default_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("Accuracy:",
          round(accuracy_score(y_test, y_pred), 3))

    print("Precision:",
          round(precision_score(y_test, y_pred), 3))

    print("Recall:",
          round(recall_score(y_test, y_pred), 3))

    print("F1-score:",
          round(f1_score(y_test, y_pred), 3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def evaluate_thresholds(probabilities, y_test):

    thresholds = [0.30, 0.50, 0.70]

    metrics_data = []

    for threshold in thresholds:

        # Probability of malignant = column 1
        predicted_malignant = (
            probabilities[:, 1] >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predicted_malignant
        )

        TN, FP, FN, TP = cm.ravel()

        accuracy = accuracy_score(
            y_test,
            predicted_malignant
        )

        precision = precision_score(
            y_test,
            predicted_malignant
        )

        recall = recall_score(
            y_test,
            predicted_malignant
        )

        f1 = f1_score(
            y_test,
            predicted_malignant
        )

        metrics_data.append({
            "Threshold": threshold,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "TP": TP,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

    threshold_metrics = pd.DataFrame(metrics_data)

    print("\n========== THRESHOLD EVALUATION ==========")

    print(
        threshold_metrics.round(3).to_string(index=False)
    )

    return threshold_metrics
