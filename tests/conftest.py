from __future__ import annotations

import json
from pathlib import Path

import pytest

SAMPLES_PATH = Path(__file__).parent / "data" / "samples.jsonl"


@pytest.fixture(scope="session")
def samples() -> list[dict]:
    with SAMPLES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
