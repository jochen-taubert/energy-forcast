# Multi-Stage: "serving" für die API, "training" für den Retraining-Job
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

FROM base AS serving
RUN pip install ".[serving]"
USER app
ENV EF_MODEL_DIR=/models
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "energy_forecast.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS training
RUN pip install ".[training,agent]"
USER app
ENV EF_MODEL_DIR=/models EF_DATA_DIR=/data
CMD ["python", "-m", "energy_forecast.pipelines.flow"]
