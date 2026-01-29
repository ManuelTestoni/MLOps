from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys

sys.path.append('/opt/airflow/mlops')

# --------------------------
#     1. TASK FUNCTIONS
# --------------------------

def preprocess_data():
    """
    Pre-processing function calls
    """
    from src.preprocessing.clean import clean_data 
    from src.preprocessing.encode import encode_features
    from src.preprocessing.scale import scale_data
    from src.preprocessing.split import split_data
    clean_data()
    encode_features()
    scale_data()
    split_data()



def train_random_forest():
    """
    Random Forest training and MLFlows logging function.
    """
    from src.training.rf import random_forest_training
    random_forest_training()
    

def train_xgboost():
    """
    Xgboost training and MLFlows logging function.
    """
    from src.training.train_xgboost import xgboost_training
    xgboost_training()


def compare_and_promote_best_model():
    """
    Function that compare different models run in MLFlow and promote the best one to production.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    #Using the docker internal interface
    TRACKING_URI = "http://host.docker.internal:5000"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Get all experiments
    experiments = client.list_experiments()

    best_run = None
    best_metric = 0.0
    metric_name = "f1_macro"
    winning_artifact_path = ""
    
    #Iterating over all experiments
    for exp_name in experiments:
        exp = client.get_experiment_by_name(exp_name.name)
        if not exp:
            print("Expriment not found:", exp_name.name)
            continue

        # Getting all the runs fot that experiment
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            continue

        run = runs[0]
        current_metric = run.data.metrics.get(metric_name, 0.0)
        model_type = run.data.tags.get("model","Unknown")  

        #Metric values comparison
        if current_metric > best_metric:
            best_metric = current_metric
            best_run = run
            if "Random_Forest" in model_type:
                winning_artifact_path = "random_forest_model"
            else:
                winning_artifact_path = "xgboost_model"

    # --- PROMOTING ---
    if best_run:
        print(f"Winner: {best_run.data.tags.get('model')} with {metric_name}: {best_metric}")
        
        # Nome univoco con cui registrare il modello in produzione
        registered_model_name = "Churn_Prediction_Prod"
        
        # Building model URI: runs:/<run_id>/<artifact_path>
        model_uri = f"runs:/{best_run.info.run_id}/{winning_artifact_path}"
        
        # 1. Registering the model
        reg_model = mlflow.register_model(model_uri, registered_model_name)
        
        # 2. Promoting to production
        client.transition_model_version_stage(
            name=registered_model_name,
            version=reg_model.version,
            stage="Production",
            archive_existing_versions=True # Archivia la vecchia produzione
        )
        print(f"Model {registered_model_name} (v{reg_model.version}) has been promoted.")
    else:
        print("No eligible model found.")
    



def notify():
    """
    Optional: invia una notifica console, email, slack...
    Per ora stampa un testo.
    """
    print("Pipeline completata. Controlla MLflow per il nuovo modello in Production.")


# --------------------------
#       2. DAG DEFINITION
# --------------------------

default_args = {
    "owner": "mlops",
    "start_date": datetime(2024, 1, 1),
}

with DAG(
    dag_id="model_training_pipeline",
    description="Complete Pipeline: preprocessing → training → confronto MLflow",
    schedule_interval=None,    # Nessuna schedulazione automatica
    catchup=False,
    default_args=default_args,
) as dag:

    # --------------- TASKS ----------------

    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    task_train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_random_forest,
    )

    task_train_xgb = PythonOperator(
        task_id="train_xgboost",
        python_callable=train_xgboost,
    )

    task_compare = PythonOperator(
        task_id="compare_and_promote_best_model",
        python_callable=compare_and_promote_best_model,
    )

    task_notify = PythonOperator(
        task_id="notify",
        python_callable=notify,
    )


    # ---------------------------
    #          DAG FLOW
    # ---------------------------

    task_preprocess >> [task_train_rf, task_train_xgb] >> task_compare >> task_notify