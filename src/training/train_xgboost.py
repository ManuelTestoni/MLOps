import mlflow
from mlflow.models import infer_signature
from xgboost import XGBClassifier, plot_importance
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from skopt.space import Real, Integer
import pandas as pd
import pickle
from pathlib import Path
import json


def xgboost_training():

    #Create connection for MLFlow
    print("-------------------------")
    print("Connecting to MLFlow...")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("Connected")

    BASE_DIR = Path("/Users/chad/Desktop/Documenti/Uni/4_Anno/MLOps")
    OUT_DIR = BASE_DIR / "models/XGBoost"
    METRICS_DIR = BASE_DIR / "data/metrics/XGBoost"

    X_train = pd.read_csv(BASE_DIR / "data/processed/train/X_train.csv")
    X_test = pd.read_csv(BASE_DIR / "data/processed/test/X_test.csv")
    y_train = pd.read_csv(BASE_DIR / "data/processed/train/y_train.csv")
    y_test = pd.read_csv(BASE_DIR / "data/processed/test/y_test.csv")

    clf = XGBClassifier(random_state=42)

    #Hyperparamters tuning
    search_space = {
        'max_depth': Integer(2, 8),
        'learning_rate': Real(0.001, 1.0, prior= 'log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5,1.0),
        'colsample_bylevel': Real(0.5,1.0),
        'colsample_bynode': Real(0.5,1.0),
        'reg_alpha': Real(0.0,10.0),
        'reg_lambda': Real(0.0,10.0),
        'gamma': Real(0.0,10.0)
    }

    opt = BayesSearchCV(estimator=clf, search_spaces = search_space, cv=10, n_iter=100, scoring='roc_auc', random_state=8)

    opt.fit(X_train, y_train.values.ravel())
    y_pred = opt.predict(X_test)

    #Calculating Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1,_ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    _,_,_, support_per_class = precision_recall_fscore_support(y_test, y_pred, average=None)

    metrics = {
         "accuracy": round(accuracy, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "f1_macro": round(f1, 4),
        "support_0": int(support_per_class[0]),
        "support_1": int(support_per_class[1])
    }

    with mlflow.start_run():
        mlflow.log_params(opt.best_params_)

        mlflow.log_metrics(metrics)
        mlflow.set_tag("model", "XGBoost_Classifier")
        signature = infer_signature(X_train, opt.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=opt, 
            name="XGBoost_model", 
            signature=signature, 
            input_example=X_train, 
            registered_model_name="ChurnModel_Staging")
    
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    predictions = loaded_model.predict(X_test)

    with open(METRICS_DIR / "xgboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("-------------------------")
    print(f"Metrics saved to {METRICS_DIR / 'xgboost.json'}")

    opt.predict_proba(X_test)
    plot_importance(opt.best_estimator_)    

    pickle.dump(opt, open(OUT_DIR / "xgboost_model.pkl", "wb"))
    print("-------------------------")
    print(f"Model saved to {OUT_DIR / 'xgboost_metrics.pkl'}")


if __name__ == "__main__":
    xgboost_training()