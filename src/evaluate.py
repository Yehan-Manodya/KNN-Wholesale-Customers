# src/evaluate.py

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from train import train_knn
from preprocess import preprocess_data


def evaluate_model():
    # Get data
    X_train, X_test, y_train, y_test = preprocess_data()

    # Train model
    model, _, _ = train_knn()

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average="weighted")
    recall = recall_score(y_test, y_test_pred, average="weighted")
    f1 = f1_score(y_test, y_test_pred, average="weighted")

    print("Model Evaluation Metrics:")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Test Accuracy  : {test_accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    evaluate_model()
