# This files aim to scale data to get a faster 
# and better performances in gradient training classifiers.

from sklearn.preprocessing import StandardScaler
import pandas as pd
from pathlib import Path

def scale_data():
    categories = ["MonthlyCharges",
                  "TotalCharges",
                  "tenure"]
    
    DIR = Path("data/processed/")
    df = pd.read_csv(DIR / "data_encoded.csv")
    scaler = StandardScaler()

    for cat in categories:
        scaler.fit(df[[cat]])
        df[cat] = scaler.transform(df[[cat]])
    
    df.to_csv(DIR / "data_scaled.csv")

    print("-------------------------")
    print("Data Scaling Completed")
    print("-------------------------")

if __name__ == "__main__":
    scale_data()