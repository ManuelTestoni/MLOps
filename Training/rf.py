import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def random_forest_training(X_train, y_train, X_test, y_test):

    df = pd.read_csv("data/processed/data_encoded.csv")
    X = df.iloc[:, 0:34]

    #Hyper Parameters
    rf = RandomForestClassifier(n_estimators=1000, criterion= 'entropy', min_samples_split = 10, max_depth= 14, random_state = 42)
    rf.fit(X_train, y_train)
    print("Model Training Completed")
    print("-------------------------")
    print("Model Evaluation:")
    y_pred = rf.predict(X_test)
    print(rf.score(X_test, y_test))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))    

    print("-------------------------")
    print("Extracting Feature Importances...")
    feature = pd.DataFrame(rf.feature_importances_, index = X.columns)
    print("Feature Importances:")
    print(feature)