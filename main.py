from PreProcessing import clean_data, encode_features, scale_data, split_data
from Training import random_forest_training, xgboost_training

if __name__ == "__main__":
    clean_data()
    encode_features()
    scale_data()
    X_train, y_train, X_test, y_test = split_data()
    print("Starting Random Forest Model Training...")
    random_forest_training(X_train, y_train, X_test, y_test)
    print("Model Training with Completed.")
    print("-------------------------")
    print("Starting XGBoost Model Training...")
    xgboost_training(X_train, y_train, X_test, y_test)
    print("XGBoost Model Training Completed.")