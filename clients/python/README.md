# ner-client

Typed sync and async Python client for [NER Service](https://github.com/moon-strider/ner).
Generated from the service's OpenAPI schema. Python 3.10+.

From the repository root:

```bash
uv add ./clients/python/ner-client
```

## Sync

```python
from ner_client import Client
from ner_client.api.default.extract_v1_extract_post import sync as extract
from ner_client.models import EntityLabel, ErrorEnvelope, ExtractRequest, NERConfig

with Client(base_url="http://127.0.0.1:8000") as client:
    result = extract(
        client=client,
        body=ExtractRequest(
            text="I visited Berlin.",
            config=NERConfig(labels=[EntityLabel(name="LOCATION", description="Cities")]),
        ),
    )
    if isinstance(result, ErrorEnvelope):
        raise RuntimeError(f"{result.error.code}: {result.error.message}")
    if result is not None:
        print(result.data.entities)
```

## Async and authentication

```python
import asyncio
import os

from ner_client import AuthenticatedClient
from ner_client.api.default.extract_v1_extract_post import asyncio as extract
from ner_client.models import ExtractRequest

async def main():
    async with AuthenticatedClient(
        base_url="http://127.0.0.1:8000", token=os.environ["NER_API_KEY"]
    ) as client:
        result = await extract(
            client=client,
            body=ExtractRequest(text="I visited Berlin.", config_id="YOUR_SAVED_CONFIG_ID"),
        )
        print(result)

asyncio.run(main())
```

Use `Client` if service authentication is disabled. Use `sync_detailed` or
`asyncio_detailed` to inspect HTTP status and response headers. Known error statuses
parse as `ErrorEnvelope`; unexpected statuses return `None` unless
`raise_on_unexpected_status=True` is configured.

Omitted `model` and `max_tokens` fields are not serialized, allowing server defaults
to apply. A cache hit has `meta.cache_hit=True`, `attempts=0`, and no new usage.

## Regenerate

From the repository root:

```bash
uv sync --frozen --extra dev
uv run --frozen python scripts/generate_client.py
uv run --frozen python scripts/generate_client.py --check
```

Do not edit the generated files. Update the API models or generator instead.
