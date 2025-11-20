import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pickle
from pathlib import Path
import json

def random_forest_training():

    OUT_DIR = Path("models/rf/")
    METRICS_DIR = Path("data/metrics/rf/")

    # Create directories if they don't exist
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/processed/data_encoded.csv")
    X = df.iloc[:, 0:34]
    X_train = pd.read_csv("data/processed/train/X_train.csv")
    X_test = pd.read_csv("data/processed/test/X_test.csv")
    y_train = pd.read_csv("data/processed/train/y_train.csv")
    y_test = pd.read_csv("data/processed/test/y_test.csv")

    #Hyper Parameters
    rf = RandomForestClassifier(n_estimators=1000, criterion= 'entropy', min_samples_split = 10, max_depth= 14, random_state = 42)
    rf.fit(X_train, y_train)
    print("Model Training Completed")
    print("-------------------------")
    print("Model Evaluation:")
    y_pred = rf.predict(X_test)
    print(rf.score(X_test, y_test))

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