import pandas as pd


def analyze_data(X, y, data):

    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    print("Class names:", data.target_names)

    class_counts = y.value_counts().sort_index()
    class_distribution = pd.DataFrame({"Class": data.target_names, "Count": class_counts.values, "Proportion": class_counts.values / len(y)})
    print(class_distribution)
