test:
    uv run --frozen --extra dev pytest -m "not integration"

lint:
    uv run --frozen --extra dev ruff check .

fmt:
    uv run --frozen --extra dev ruff format .

format-check:
    uv run --frozen --extra dev ruff format --check .

typecheck:
    uv run --frozen --extra dev mypy src

check: lint format-check typecheck test

run:
    uv run --frozen uvicorn ner_service.main:app --host 127.0.0.1 --port 8000

build:
    docker build -t ner-service .

generate-client:
    uv run --frozen --extra dev python scripts/generate_client.py

check-client:
    uv run --frozen --extra dev python scripts/generate_client.py --check

observe-up:
    docker compose --profile observability up -d --build

observe-down:
    docker compose --profile observability down

observe-logs:
    docker compose --profile observability logs -f

smoke:
    uv run --frozen python scripts/smoke.py
