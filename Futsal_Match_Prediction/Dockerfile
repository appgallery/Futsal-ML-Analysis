FROM apache/airflow:2.9.1-python3.11

# Switch to root to install any system dependencies if needed
USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Install python dependencies required by the pipeline
RUN pip install --no-cache-dir \
    "fastapi>=0.136.1" \
    "joblib>=1.5.3" \
    "numpy>=1.26.0" \
    "pandas<2.2" \
    "pydantic>=2.13.4" \
    "pymysql>=1.2.0" \
    "scikit-learn>=1.3.0" \
    "sqlalchemy<2.0" \
    "uvicorn>=0.47.0" \
    "xgboost>=2.1.0"
