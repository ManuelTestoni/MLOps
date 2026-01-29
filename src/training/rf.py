import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pickle
from pathlib import Path
import json

def random_forest_training():

    #Create connection for MLFlow
    print("-------------------------")
    print("Connecting to MLFlow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("Connected")

    mlflow.set_experiment("Churn_Prediction")

    BASE_DIR = Path("/Users/chad/Desktop/Documenti/Uni/4_Anno/MLOps/src")
    OUT_DIR = BASE_DIR / "models/rf"
    METRICS_DIR = BASE_DIR / "data/metrics/rf"

    # Create directories if they don't exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BASE_DIR / "data/processed/data_encoded.csv")
    X = df.iloc[:, 0:34]
    X_train = pd.read_csv(BASE_DIR / "data/processed/train/X_train.csv")
    X_test = pd.read_csv(BASE_DIR / "data/processed/test/X_test.csv")
    y_train = pd.read_csv(BASE_DIR / "data/processed/train/y_train.csv")
    y_test = pd.read_csv(BASE_DIR / "data/processed/test/y_test.csv")

    #Hyper Parameters

    params = {
        "n_estimators": 1000,
        "criterion": 'entropy',
        "min_samples_split": 10,
        "max_depth": 14,
        "random_state": 42
    }

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train.values.ravel())
    print("Model Training Completed")
    print("-------------------------")
    y_pred = rf.predict(X_test)
    

    #Calculating Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support, = precision_recall_fscore_support(y_test, y_pred, average='macro')
    _, _, _, support_per_class = precision_recall_fscore_support(y_test, y_pred, average=None)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "support_0": int(support_per_class[0]),
        "support_1": int(support_per_class[1])
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", "Random_Forest_Classifier")
        signature = infer_signature(X_train, rf.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=rf, 
            name="random_forest_model", 
            signature=signature, 
            input_example=X_train, 
            registered_model_name="ChurnModel_Staging")
    
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)
    

    with open(METRICS_DIR / "rf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("-------------------------")
    print(f"Metrics saved to {METRICS_DIR / 'metrics.json'}")

    print("-------------------------")
    #Debug or info prints
    #print("Classification Report:")
    #print(classification_report(y_test, y_pred))    
    #print("-------------------------")
    #print("Extracting Feature Importances...")
    #feature = pd.DataFrame(rf.feature_importances_, index = X.columns)
    #print("Feature Importances:")
    #print(feature)

    pickle.dump(rf, open(OUT_DIR / "random_forest_model.pkl", "wb"))

if __name__ == "__main__":
    random_forest_training()