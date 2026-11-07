.PHONY: install test lint train-synthetic train mlflow serve docker agent agent-rules eval eval-rules approvals

install:
	uv venv && uv pip install -e ".[dev]"

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

train-synthetic:
	uv run ef-train --synthetic --force

train:
	uv run python -m energy_forecast.pipelines.flow

mlflow:
	uv run mlflow ui --port 5000

serve:
	uv run uvicorn energy_forecast.serving.app:app --reload

docker:
	docker build --target serving -t energy-forecast:serving .
	docker build --target training -t energy-forecast:training .

agent:
	uv run ef-agent run

agent-rules:
	uv run ef-agent run --synthetic --rules

eval:
	uv run ef-agent eval --judge --out eval-results.json

eval-rules:
	uv run ef-agent eval --rules

approvals:
	uv run uvicorn energy_forecast.agent.api:app --port 8001
