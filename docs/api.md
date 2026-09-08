# API contract

The machine-readable contract is served at `/openapi.json`; interactive docs are at `/docs`.
If `NER_API_KEY` is configured, send `Authorization: Bearer <key>` for the `/v1`
configuration, extraction, and provider endpoints. Health and readiness remain accessible.

## Extract

```json
{
  "text": "Tim Cook visited Berlin.",
  "config": {
    "labels": [{"name": "PERSON", "description": "Names of people"}],
    "require_offsets": true
  }
}
```

Exactly one of inline `config` or a saved `config_id` is required. Unknown fields
are rejected to catch configuration typos. Input text must be nonempty.

```json
{
  "data": {
    "entities": [{"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8}],
    "provider": "llama_cpp",
    "model": "smollm2-1.7b",
    "usage": {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170}
  },
  "meta": {
    "request_id": "example-1",
    "latency_ms": 750.0,
    "attempts": 1,
    "cache_hit": false,
    "warnings": []
  }
}
```

This is an illustrative response. `usage` is included when supplied by the model server.
`attempts` counts total model calls, including structured-output repair attempts.
`retries=3` therefore means **at most three calls**, not four. HTTPX transport retries
cover connection establishment failures only, not arbitrary HTTP errors or read timeouts.
There is no separate automatic retry of rate limits, quota failures, or refusals.

`X-Request-ID` is echoed if it contains 1–128 letters, digits, `.`, `_`, `:`, or `-`.
Otherwise a UUID is generated. Every response receives an ID, including errors.

## Matching semantics

- Entity labels match `^[A-Z][A-Z0-9_]*$` and must be unique in a configuration.
- Without offsets, only entities whose text occurs in the input are retained;
  duplicate `(text, label)` pairs are removed. Results follow model order.
- With offsets, each model-returned entity maps to the next available matching
  occurrence. One model item produces at most one result; the service does not
  expand a surface into every occurrence. Overlaps are resolved in model order.
- `start` is inclusive, `end` is exclusive. Positions count Unicode code points,
  not UTF-8 bytes, UTF-16 code units, or visible grapheme clusters:
  `text[start:end] == entity.text` in Python.
- `case_sensitive=false` uses Python regular-expression case-insensitive matching
  against the original string. It does not apply Unicode normalization or expand
  multi-character case folds such as `ß` to `ss`.
- Matching is by substring, not word boundaries. A surface may match inside a
  larger word. Repeated ambiguous surfaces are assigned left to right; the service
  cannot recover distinctions the model did not express.
- Dropped unmatched, duplicate, or overlapping items produce a count-only warning.
  Input text and entity surfaces are not logged by the matcher.

These rules ground the returned surfaces; they do not validate the semantic labels.
Nested and overlapping NER spans are not supported in offset mode.

## Configurations

`POST /v1/configs` accepts the same object as inline `config` and returns `{id, config}`.
Read/list/replace/patch/delete use `/v1/configs` and `/v1/configs/{id}`.
`PUT` replaces an existing record; it does not create an arbitrary ID.
`DELETE` returns 204. Missing records return 404.

| Field | Default / behavior |
| --- | --- |
| `labels` | Required list of `{name, description}` |
| `model` | `NER_MODEL` when omitted; saved configs retain the resolved value |
| `max_tokens` | `MAX_TOKENS` when omitted |
| `require_offsets` | `false` |
| `case_sensitive` | `true` |
| `retries` | `3` total attempts, bounded by `MAX_ATTEMPTS` |
| `reasoning_effort` | `null`; passed to providers that support it |
| `system_prompt` | `null`; use the default extraction prompt |
| `few_shot_examples` | `[]` |

PATCH distinguishes omission from `null`. Use `null` to clear `system_prompt` or
`reasoning_effort`; non-nullable fields reject explicit nulls. PATCH updates are
serialized within one service process. Multi-process concurrent edits are last-writer-wins.
The config list is unpaginated and intended for a small set of shared configurations.

## Templates and examples

Custom `system_prompt` is a full replacement template. It supports dotted dictionary
lookups rooted at `cfg` and `payload`; `cfg.schema` contains the compact JSON Schema.
Literal braces use `{{` and `}}`. Attribute access, format specifications, and conversion
flags are not supported. Braces inside label descriptions are literal with the default prompt.

```json
{
  "text": "We visited Paris.",
  "config": {
    "labels": [{"name": "LOCATION", "description": "Cities"}],
    "system_prompt": "Extract cities using {cfg.schema}. Context: {payload.context}",
    "few_shot_examples": [
      {"text": "I visited Rome.", "entities": [{"text": "Rome", "label": "LOCATION"}]}
    ]
  },
  "prompt_payload": {"context": "Travel"}
}
```

Few-shot entities must use configured labels and exact substrings of their example text.
Missing template values fail before a model request, even when an otherwise similar
result is cached. The rendered prompt has its own length limit.

## Cache

The per-process cache key includes the input text, resolved configuration, provider,
and rendered system prompt. Updating a configuration or a used payload value changes
the key. Cache hits return `meta.cache_hit=true`, `attempts=0`, and omit `data.usage`;
they do not increment token or estimated-cost counters. The cache is bounded by entry
count and TTL, uses LRU eviction, and disappears on restart. It is not tenant-isolated.
Concurrent identical misses may each call the provider; no single-flight deduplication is promised.

## Batch

`POST /v1/batch/extract` accepts `{"items": [<extract request>, ...]}` with 1–100 items.
Results preserve input order. A syntactically invalid item rejects the whole body with 422;
runtime failures such as a missing stored config are isolated to their item.

For compatibility, success is nested as `items[i].data = {data, meta}`;
item metadata is also available at `items[i].meta`. Failures contain
`items[i].error = {code, message}`. Batch HTTP status is 200 for mixed results;
check `meta.failed`. All items share the HTTP request ID and have distinct `index` values.
Batch summary includes `total`, `succeeded`, `failed`, and `latency_ms`.

## Errors

```json
{"error":{"code":"validation_error","message":"request validation failed","details":{},"request_id":"example-1"}}
```

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `provider_bad_request` | Upstream rejected model parameters |
| 401 | `http_error` | Missing or incorrect service Bearer token |
| 402 | `provider_quota_exhausted` | Upstream credit/quota failure |
| 403 | `provider_permission_denied` | Upstream permission failure |
| 404 | `config_not_found` | Unknown config ID |
| 413 | `request_too_large` | Body exceeds byte limit, including chunked bodies |
| 422 | `validation_error`, `invalid_request`, `prompt_template_error` | Request/config validation |
| 429 | `provider_rate_limited` | Upstream rate limit; safe retry headers are forwarded |
| 502 | `provider_auth_failed`, `provider_upstream_error`, `provider_error` | Upstream auth, transport, circuit, or output failure |
| 503 | `http_error` | Service or storage unavailable |
| 500 | `internal_error` | Unexpected internal failure |

Provider bodies and exception messages are not returned to API clients. Safe numeric
status/attempt details may be included. Batch errors use the same public codes and messages.
