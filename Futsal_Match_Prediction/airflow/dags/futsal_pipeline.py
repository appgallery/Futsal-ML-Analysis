from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# 1. Point Airflow to your project directory
# Dynamically get the project root so it works in both Windows and Linux/WSL
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Set the Database URI for the production pipeline (Can also be set in Airflow UI Variables)
os.environ['DATABASE_URI'] = os.environ.get('DATABASE_URI', 'mysql+pymysql://root:root@localhost:3307/airflow_db')

# 2. Import your existing pipelines
from pipeline.data_pipeline import DataPipeline
from pipeline.training_pipeline import TrainingPipeline

# 3. Create wrapper functions that Airflow will call
def run_data_etl():
    # This runs components/data_ingestion.py and components/data_transformation.py
    pipeline = DataPipeline()
    matches, players = pipeline.run_pipeline()
    print(f"Data ETL complete. Processed {len(matches)} matches.")

def run_model_training():
    # This runs scripts/model_trainer.py and saves models to models/ folder.
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    print("Model training complete.")

# 4. Define the Airflow DAG properties
default_args = {
    'owner': 'futsal_admin',
    'start_date': datetime(2026, 6, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'futsal_nightly_db_pipeline',
    default_args=default_args,
    schedule='0 0 * * *', # Runs at midnight every day
    catchup=False
) as dag:

    # 5. Define Tasks
    task_etl = PythonOperator(
        task_id='extract_transform_load_db',
        python_callable=run_data_etl
    )

    task_train = PythonOperator(
        task_id='train_ml_models',
        python_callable=run_model_training
    )

    # 6. Set Execution Order (ETL must finish before Training starts)
    task_etl >> task_train
