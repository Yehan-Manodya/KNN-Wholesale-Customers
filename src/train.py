# src/train.py

from sklearn.neighbors import KNeighborsClassifier
from preprocess import preprocess_data

def train_knn(n_neighbors=5):
    

    # Get preprocessed data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Initialize KNN model
    knn_model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric="minkowski",  # Euclidean distance
        p=2
    )

    
    knn_model.fit(X_train, y_train)

    return knn_model, X_test, y_test


# Run training directly
if __name__ == "__main__":
    model, X_test, y_test = train_knn()
    print("KNN model trained successfully!")
    print("Test data shape:", X_test.shape)
