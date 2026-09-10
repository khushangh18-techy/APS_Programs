from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training size:", len(y_train))
    print("Test size:", len(y_test))
    print("\nTraining proportions:")
    print(y_train.value_counts(normalize=True).sort_index())
    print("\nTest proportions:")
    print(y_test.value_counts(normalize=True).sort_index())

    return X_train, X_test, y_train, y_test


def create_pipeline():

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    return model


def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model
