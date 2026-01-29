import pandas as pd
import json
from pathlib import Path


def encode_features():
    """
    Executes one hot encoding and binary encoding on the dataset.
    """
    # Setting up paths
    BASE_DIR = Path("/Users/chad/Desktop/Documenti/Uni/4_Anno/MLOps")
    DIR = BASE_DIR / "data/processed"

    df = pd.read_csv(DIR / "data_cleaned.csv")

    # -----------------------------
    # 1. Binary Columns → 0/1
    # -----------------------------
    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "SeniorCitizen",
        "Churn"  
    ]

    # Mapping Yes/No → 1/0
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Female": 1, "Male": 0})

    # -----------------------------
    # 2. Multi-categorical columns
    # -----------------------------
    multi_cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    df = df.replace("No internet service", "No")
    # One-hot encoding 
    df = pd.get_dummies(df, columns=multi_cat_cols)

    # -----------------------------
    # 3. Save column schema
    # -----------------------------
    feature_columns = [col for col in df.columns if col != "Churn"]

    #Creating file if not exists
    with open(BASE_DIR / "data/processed/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=4)
    df.to_csv(DIR / 'data_encoded.csv', index=False)
    
    print("-------------------------")
    print("Data Encoding Completed")
    print("-------------------------")

if __name__ == "__main__":
    encode_features()