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

clean:
    rm -rf .venv/ __pycache__/ *.egg-info/ dist/ build/
    rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
    find . -name "*.pyc" -delete
    find . -name "*.pyo" -delete

run:
    uv run uvicorn ner_service.main:app --reload

build:
    docker build -t ner-service .

bench-offsets:
    CEREBRAS_API_KEY={{ env_var('CEREBRAS_API_KEY') }} uv run --extra dev python scripts/benchmark_conll.py --model llama3.1-8b --concurrency 40 --require-offsets
