import pandas as pd
from pathlib import Path
import numpy as np

def clean_data():
    """Cleans data and restructure the dataset for analysis"""

    BASE_DIR = Path("/Users/chad/Desktop/Documenti/Uni/4_Anno/MLOps")
    INPUT_DIR = BASE_DIR / "data/raw"
    OUTPUT_DIR = BASE_DIR / "data/processed"
    #Some debugging prints
    print("-------------------------")
    print("Data Loading Completed")

    df = pd.read_csv(INPUT_DIR / 'data.csv')
    df = df.replace(" ", np.nan)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.drop(columns=["customerID"])
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})
    df.to_csv(OUTPUT_DIR / 'data_cleaned.csv', index=False)

    print("-------------------------")
    print("Data Cleaning Completed")
    print("-------------------------")

if __name__ == "__main__":
    clean_data()