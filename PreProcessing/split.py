import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

def split_data():

    #setting up paths
    DIR = Path("data/processed/")
    TRAIN_DIR = Path(DIR / "train/")
    TEST_DIR = Path(DIR / "test/")

    # Create directories if they don't exist
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    #Load the dataset
    df = pd.read_csv(DIR / "data_encoded.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    try:
        if(len(X) == len(y)):
            X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
    except ValueError as ve:
        print("Error during train-test split:", ve)
        return
    
    # CSV and Pickle Dump
    X_train.to_csv(DIR / "train/X_train.csv", index=False)
    X_test.to_csv(DIR / "test/X_test.csv", index=False)
    y_train.to_csv(DIR / "train/y_train.csv", index=False)
    y_test.to_csv(DIR / "test/y_test.csv", index=False)
    pickle.dump(X_train.columns.tolist(), open(DIR / "train/feature_columns.pkl", "wb"))
    pickle.dump(y_train.name, open(DIR / "train/target_column.pkl", "wb"))
    pickle.dump(X_test.columns.tolist(), open(DIR / "test/feature_columns_test.pkl", "wb"))
    pickle.dump(y_test.name, open(DIR / "test/target_column_test.pkl", "wb"))

    print("-------------------------")
    print("Data Splitting Completed")
    print("-------------------------")

if __name__ == "__main__":
    split_data()
