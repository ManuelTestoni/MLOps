from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier, plot_importance
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd
import pickle
from pathlib import Path


def xgboost_training():

    OUT_DIR = Path("../models/XGBoost/")

    X_train = pd.read_csv("data/processed/train/X_train.csv")
    X_test = pd.read_csv("data/processed/test/X_test.csv")
    y_train = pd.read_csv("data/processed/train/y_train.csv")
    y_test = pd.read_csv("data/processed/test/y_test.csv")

    estimators = [
        'encoder', TargetEncoder(),
        'clf', XGBClassifier(random_state=42)
    ]

    pipe = Pipeline(steps=estimators)

    #Hyperparamters tuning
    search_space = {
        'clf__max_depth': Integer(2, 8),
        'clf__learning_rate': Real(0.001, 1.0, prior= 'log-uniform'),
        'clf__subsample': Real(0.5, 1.0),
        'clf__colsample_bytree': Real(0.5,1.0),
        'clf__colsample_bylevel': Real(0.5,1.0),
        'clf__colsample_bynode': Real(0.5,1.0),
        'clf__reg_alpha': Real(0.0,10.0),
        'clf__reg_lambda': Real(0.0,10.0),
        'clf__gamma': Real(0.0,10.0)
    }

    opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=100, scoring='roc_auc', random_state=8)

    opt.fit(X_train, y_train)
    opt.best_estimator_
    opt.best_score_
    opt.predict(X_test)
    opt.precit_proba(X_test)
    xgboost_step = opt.best_estimator_.steps[1]
    xgboost_model = xgboost_step[1]
    plot_importance(xgboost_model)    

    pickle.dump(opt, open(OUT_DIR / "xgboost_model.pkl", "wb"))


