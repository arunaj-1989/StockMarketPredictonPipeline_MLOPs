FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MLFLOW_DISABLE_AGENT_HINT=1

# libgomp is required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data/raw /app/data/processed /app/docs /mlflow \
    && chown -R app:app /app /mlflow
USER app

EXPOSE 8000 8501 5000
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
