# Customer Churn Prediction MLOps Pipeline

This repository contains a complete MLOps pipeline designed to predict customer churn. The project demonstrates an end-to-end machine learning workflow, from data versioning to model deployment, ensuring reproducibility and automation.

## Project Overview

The goal of this project is to identify the best machine learning model for predicting customer churn and manage its lifecycle in a production-like environment. The pipeline trains multiple models, tracks their performance, and automatically selects the best candidate for deployment.

## Tech Stack

*   **Language**: Python
*   **Data Versioning**: [DVC (Data Version Control)](https://dvc.org/)
*   **Experiment Tracking & Model Registry**: [MLflow](https://mlflow.org/)
*   **Orchestration**: [Apache Airflow](https://airflow.apache.org/)
*   **Machine Learning Models**:
    *   Random Forest
    *   XGBoost

## Pipeline Architecture

The workflow is structured as follows:

1.  **Data Management**: The Churn dataset is versioned using DVC to ensure data reproducibility across different experiments.
2.  **Model Training**: Two primary classification algorithms, Random Forest and XGBoost, are trained on the processed data.
3.  **Experiment Tracking**: MLflow is used to log parameters, metrics (accuracy, precision, recall, F1-score), and artifacts for every run.
4.  **Model Selection & Deployment**:
    *   Apache Airflow orchestrates the periodic evaluation of trained models.
    *   The system compares performance metrics and automatically promotes the best-performing model to the "Production" stage in the MLflow Model Registry.

## Getting Started

*(Instructions on how to set up the environment, install dependencies, and run the pipeline will go here)*
