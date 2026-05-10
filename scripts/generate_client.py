from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

from ner_service.main import create_app

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "clients/python/ner-client"
CONFIG_PATH = ROOT / "clients/python/openapi-python-client.yml"
SPEC_PATH = ROOT / "clients/python/.openapi.json"
PYPROJECT = """[project]
name = \"ner-client\"
version = \"1.0.0\"
description = \"Typed Python client for NER Service\"
readme = \"README.md\"
requires-python = \">=3.10\"
dependencies = [
    \"attrs==26.1.0\",
    \"httpx==0.28.1\",
    \"python-dateutil==2.9.0.post0\",
]

[tool.uv.build-backend]
module-name = \"ner_client\"
module-root = \"\"

[build-system]
requires = [\"uv_build==0.11.11\"]
build-backend = \"uv_build\"

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = [\"F\", \"I\", \"UP\"]
"""
README = """# ner-client

Typed Python client for NER Service.

## Regenerate

```bash
uv run python scripts/generate_client.py
```

## Install

```bash
uv add ./clients/python/ner-client
```

## Sync example

```python
from ner_client import Client
from ner_client.api.default.extract_v1_extract_post import sync as extract_sync
from ner_client.models.entity_label import EntityLabel
from ner_client.models.extract_request import ExtractRequest
from ner_client.models.ner_config import NERConfig

client = Client(base_url=\"http://127.0.0.1:8000\")
result = extract_sync(
    client=client,
    body=ExtractRequest(
        text=\"Tim Cook visited Berlin.\",
        config=NERConfig(labels=[EntityLabel(name=\"PERSON\", description=\"People\")]),
    ),
)
```

## Async example

```python
import asyncio

from ner_client import Client
from ner_client.api.default.extract_v1_extract_post import asyncio as extract_async
from ner_client.models.entity_label import EntityLabel
from ner_client.models.extract_request import ExtractRequest
from ner_client.models.ner_config import NERConfig


async def main() -> None:
    async with Client(base_url=\"http://127.0.0.1:8000\") as client:
        result = await extract_async(
            client=client,
            body=ExtractRequest(
                text=\"Tim Cook visited Berlin.\",
                config=NERConfig(labels=[EntityLabel(name=\"PERSON\", description=\"People\")]),
            ),
        )
        print(result)


asyncio.run(main())
```
"""


def _strip_docstrings(node: ast.AST) -> None:
    if (
        isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
    ):
        value = node.body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            node.body.pop(0)
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)


def _rewrite_python_sources() -> None:
    for path in OUTPUT_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _strip_docstrings(tree)
        path.write_text(f"{ast.unparse(tree)}\n", encoding="utf-8")


def _rewrite_metadata() -> None:
    (OUTPUT_DIR / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (OUTPUT_DIR / "README.md").write_text(README, encoding="utf-8")
    shutil.rmtree(OUTPUT_DIR / ".ruff_cache", ignore_errors=True)


def main() -> None:
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    SPEC_PATH.write_text(
        json.dumps(create_app().openapi(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    subprocess.run(
        [
            "openapi-python-client",
            "generate",
            "--path",
            str(SPEC_PATH),
            "--meta",
            "uv",
            "--config",
            str(CONFIG_PATH),
            "--output-path",
            str(OUTPUT_DIR),
            "--overwrite",
        ],
        check=True,
        cwd=ROOT,
    )
    _rewrite_python_sources()
    _rewrite_metadata()
    subprocess.run(["ruff", "check", ".", "--fix-only"], check=True, cwd=OUTPUT_DIR)
    subprocess.run(["ruff", "format", "."], check=True, cwd=OUTPUT_DIR)
    shutil.rmtree(OUTPUT_DIR / ".ruff_cache", ignore_errors=True)
    SPEC_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
