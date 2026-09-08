"""Exercise a running NER API. May incur provider charges; uses only synthetic text."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import httpx


def run(base_url: str, api_key: str | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    config: dict[str, Any] = {
        "labels": [
            {"name": "PERSON", "description": "Names of people"},
            {"name": "LOCATION", "description": "Cities and countries"},
        ],
        "require_offsets": True,
        "max_tokens": 256,
        "retries": 2,
    }
    text = "Tim Cook visited Berlin last week."
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with httpx.Client(base_url=base_url, headers=headers, timeout=180) as client:

        def call(method: str, path: str, **kwargs: Any) -> Any:
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else None

        call("GET", "/v1/health")
        ready = call("GET", "/v1/ready")
        record = call("POST", "/v1/configs", json=config)
        config_id = record["id"]
        try:
            assert call("GET", f"/v1/configs/{config_id}")["id"] == config_id
            assert any(item["id"] == config_id for item in call("GET", "/v1/configs"))
            first = call("POST", "/v1/extract", json={"text": text, "config_id": config_id})
            results.append({"scenario": "stored config", "response": first})
            for entity in first["data"]["entities"]:
                assert text[entity["start"] : entity["end"]] == entity["text"]
            repeated = call("POST", "/v1/extract", json={"text": text, "config_id": config_id})
            results.append({"scenario": "repeat", "response": repeated})
            if repeated["meta"]["cache_hit"]:
                assert repeated["meta"]["attempts"] == 0
                assert "usage" not in repeated["data"]
                assert repeated["data"]["entities"] == first["data"]["entities"]
            call("PATCH", f"/v1/configs/{config_id}", json={"case_sensitive": False})
            call("PUT", f"/v1/configs/{config_id}", json=config)
            inline = call(
                "POST", "/v1/extract", json={"text": "No names appear here.", "config": config}
            )
            results.append({"scenario": "inline config", "response": inline})
            prompt_config = {
                **config,
                "system_prompt": (
                    "Extract only cities. Follow schema {cfg.schema}. Context: {payload.context}"
                ),
                "few_shot_examples": [
                    {"text": "I visited Rome.", "entities": [{"text": "Rome", "label": "LOCATION"}]}
                ],
            }
            for context in ("Travel", "Geography"):
                response = call(
                    "POST",
                    "/v1/extract",
                    json={
                        "text": "We visited Paris.",
                        "config": prompt_config,
                        "prompt_payload": {"context": context},
                    },
                )
                results.append(
                    {"scenario": f"prompt and few-shot: {context}", "response": response}
                )
            batch = call(
                "POST",
                "/v1/batch/extract",
                json={
                    "items": [
                        {"text": "Ada lives in London.", "config": config},
                        {"text": "Hello", "config_id": "missing-smoke-config"},
                    ]
                },
            )
            assert batch["meta"]["succeeded"] == 1 and batch["meta"]["failed"] == 1
            assert batch["items"][1]["error"]["code"] == "config_not_found"
            results.append({"scenario": "mixed batch", "response": batch})
            response = client.post("/v1/extract", json={"text": "Hello"})
            assert response.status_code == 422 and "error" in response.json()
            metrics = client.get("/metrics")
            metrics.raise_for_status()
            assert "ner_tokens_total" in metrics.text
        finally:
            call("DELETE", f"/v1/configs/{config_id}")
        assert client.get(f"/v1/configs/{config_id}").status_code == 404
    return {
        "ready": ready,
        "elapsed_s": time.perf_counter() - started,
        "results": results,
        "note": "Checks API behavior and source offsets, not NER accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(run(args.url, os.environ.get("NER_API_KEY")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
