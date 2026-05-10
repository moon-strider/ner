# ner-client

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

client = Client(base_url="http://127.0.0.1:8000")
result = extract_sync(
    client=client,
    body=ExtractRequest(
        text="Tim Cook visited Berlin.",
        config=NERConfig(labels=[EntityLabel(name="PERSON", description="People")]),
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
    async with Client(base_url="http://127.0.0.1:8000") as client:
        result = await extract_async(
            client=client,
            body=ExtractRequest(
                text="Tim Cook visited Berlin.",
                config=NERConfig(labels=[EntityLabel(name="PERSON", description="People")]),
            ),
        )
        print(result)


asyncio.run(main())
```
