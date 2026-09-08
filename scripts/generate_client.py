from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ner_service.main import create_app

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "clients/python/ner-client"
CONFIG_PATH = ROOT / "clients/python/openapi-python-client.yml"
PYPROJECT = """[project]
name = \"ner-client\"
version = \"1.1.0\"
description = \"Typed Python client for NER Service\"
readme = \"README.md\"
license = \"MIT\"
requires-python = \">=3.10\"
dependencies = [
    \"attrs==26.1.0\",
    \"httpx==0.28.1\",
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
README = (ROOT / "clients/python/README.md").read_text(encoding="utf-8")


def _rewrite_metadata(output: Path) -> None:
    (output / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (output / "README.md").write_text(README, encoding="utf-8")
    shutil.copyfile(ROOT / "LICENSE", output / "LICENSE")


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SDK without deleting it on failure.")
    parser.add_argument("--check", action="store_true", help="fail if committed SDK differs")
    args = parser.parse_args()
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ner-client-") as directory:
        staging = Path(directory)
        spec = staging / "openapi.json"
        app = create_app()
        spec.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2), encoding="utf-8")
        app.state.tracer_provider.shutdown()
        output = staging / "ner-client"
        subprocess.run(
            [
                "openapi-python-client",
                "generate",
                "--path",
                str(spec),
                "--meta",
                "uv",
                "--config",
                str(CONFIG_PATH),
                "--output-path",
                str(output),
            ],
            check=True,
            cwd=ROOT,
        )
        _rewrite_metadata(output)
        # Libraries use their consumer's lock; a generated environment lock is unnecessary.
        (output / "uv.lock").unlink(missing_ok=True)
        subprocess.run(["ruff", "check", ".", "--fix-only"], check=True, cwd=output)
        subprocess.run(["ruff", "format", "."], check=True, cwd=output)
        shutil.rmtree(output / ".ruff_cache", ignore_errors=True)
        if args.check:
            expected, actual = _files(output), _files(OUTPUT_DIR)
            changes = sorted(
                key
                for key in expected.keys() | actual.keys()
                if expected.get(key) != actual.get(key)
            )
            if changes:
                raise SystemExit("SDK is stale: " + ", ".join(changes))
            print("SDK matches OpenAPI.")
            return
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
        shutil.copytree(output, OUTPUT_DIR)


if __name__ == "__main__":
    main()
