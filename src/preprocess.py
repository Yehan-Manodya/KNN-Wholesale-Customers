# src/preprocess.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    

    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build correct path to dataset
    data_path = os.path.join(
        BASE_DIR, "data", "Wholesale customers data.csv"
    )

    # Load dataset
    df = pd.read_csv(data_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Features and target
    X = df.drop("Channel", axis=1)
    y = df["Channel"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# Test run
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Preprocessing completed successfully!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
