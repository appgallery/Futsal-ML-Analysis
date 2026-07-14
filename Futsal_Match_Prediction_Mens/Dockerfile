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

# Cache bust: 2026-06-16
# Install python dependencies required by the pipeline
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml
