from data_loader import load_data
from data_analysis import analyze_data
from visualization import plot_class_distribution
from model_training import (split_data,create_pipeline,train_model)
from evaluation import (get_probabilities,evaluate_default_model,evaluate_thresholds)

def main():
    X, y, data = load_data()

    class_distribution = analyze_data(
        X,
        y,
        data
    )
    plot_class_distribution(
        class_distribution
    )
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )
    pipeline = create_pipeline()

    model = train_model(
        pipeline,
        X_train,
        y_train
    )
    print("\nModel trained successfully.")

    probabilities, results = get_probabilities(
        model,
        X_test
    )
    print(results.head(10).round(3))

    evaluate_default_model(
        model,
        X_test,
        y_test
    )
    threshold_metrics = evaluate_thresholds(
        probabilities,
        y_test
    )

if __name__ == "__main__":
    main()
