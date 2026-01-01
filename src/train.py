from sklearn.neighbors import KNeighborsClassifier
from preprocess import preprocess_data

def train_knn(n_neighbors=5):
    # Load preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Initialize KNN
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="minkowski",  # Euclidean distance
        p=2
    )

    # Train KNN
    model.fit(X_train, y_train)

    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_knn()
    print("KNN model trained successfully!")
