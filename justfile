test:
    uv run pytest -m "not integration" -v

test-integration:
    uv run pytest -m integration -v

lint:
    uv run ruff check .

fmt:
    uv run ruff format .

typecheck:
    uv run mypy src

check: lint fmt typecheck test

benchmark:
    uv run --extra dev python scripts/benchmark_conll.py --model llama3.1-8b --concurrency 40

profile:
    uv run python scripts/profile.py --provider cerebras --model llama3.1-8b --texts-count 4 --concurrency 2 --text-lengths 64,256,1024

clean:
    rm -rf .venv/ __pycache__/ *.egg-info/ dist/ build/
    rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete

run:
    uv run uvicorn ner_service.main:app --host 0.0.0.0 --port 8000

build:
    docker build -t ner-service .

generate-client:
    uv run python scripts/generate_client.py

observe-up:
    docker compose up -d --build

observe-down:
    docker compose down -v

observe-logs:
    docker compose logs -f ner-service prometheus grafana

bench-offsets:
    CEREBRAS_API_KEY={{ env_var('CEREBRAS_API_KEY') }} uv run --extra dev python scripts/benchmark_conll.py --model llama3.1-8b --concurrency 40 --require-offsets
