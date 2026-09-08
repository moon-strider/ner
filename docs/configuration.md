# Configuration

Settings are read from environment variables and `.env` (environment wins).
Unknown environment variables are ignored. Unknown fields in API requests are rejected.
Restart the service after changing environment settings.

The application defaults remain Cerebras + `llama3.1-8b` for compatibility.
The checked-in `.env.example` overrides them for the local CPU quick start.
Select an actual model available to your cloud account; no cloud model availability
or quality is implied by the historical application default.

## Providers

| `NER_PROVIDER` | Credential | Default base URL |
| --- | --- | --- |
| `llama_cpp` | `LLAMA_CPP_API_KEY` (default `not-needed`) | `http://127.0.0.1:8080/v1` |
| `vllm` | `VLLM_API_KEY` (default `not-needed`) | Required: `VLLM_BASE_URL` |
| `openai` | Required: `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `cerebras` | Required: `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` |
| `openrouter` | Required: `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |

Each base URL can be overridden with the matching `<PROVIDER>_BASE_URL` variable.
URLs include `/v1` where applicable; the service appends `/chat/completions`.
They are trusted administrator configuration, never supplied by API callers.
Do not point them at untrusted servers or include credentials in the URL.

Example cloud configuration:

```dotenv
NER_PROVIDER=cerebras
NER_MODEL=your-available-model-id
CEREBRAS_API_KEY=your-api-key
REQUEST_TIMEOUT_S=30
PROVIDER_CONCURRENCY_LIMIT=10
BATCH_CONCURRENCY=5
MAX_TOKENS=1024
```

The adapter sends `response_format.type=json_schema`, strict schema mode, and
`temperature=0`. `llama_cpp` uses `max_tokens`; other adapters use
`max_completion_tokens`. Select a provider/model combination supporting these
parameters. Some reasoning models do not accept temperature or structured output;
there is no silent fallback to unconstrained text generation. `reasoning_effort`
is passed through only when specified in the request config.

## Runtime settings

Defaults below are the application defaults, before `.env` overrides.

| Variable | Default | Meaning |
| --- | --- | --- |
| `NER_PROVIDER` | `cerebras` | Provider identifier |
| `NER_MODEL` | `llama3.1-8b` | Default model when omitted by a request |
| `ALLOWED_MODELS` | `[]` | Optional JSON list of allowed model IDs; empty allows any |
| `NER_API_KEY` | Unset | Optional service Bearer token; must be nonempty when set |
| `REQUEST_TIMEOUT_S` | `30` | HTTPX upstream per-operation timeout, not a whole-request deadline |
| `TRANSPORT_RETRIES` | `2` | Connection-establishment retries |
| `MAX_TOKENS` | `1024` | Default output token budget |
| `RATE_LIMIT_RPS` | `100` | Token bucket refill per process |
| `RATE_LIMIT_BURST` | `200` | Token bucket burst capacity |
| `PROVIDER_CONCURRENCY_LIMIT` | `50` | In-flight provider calls per process |
| `BATCH_CONCURRENCY` | `10` | Concurrent items inside each batch |
| `MAX_REQUEST_BODY_BYTES` | `2000000` | HTTP body size cap, before JSON parsing |
| `MAX_TEXT_LENGTH` | `32000` | Maximum input/few-shot text length in characters |
| `MAX_LABELS` | `50` | Labels per configuration |
| `MAX_LABEL_DESCRIPTION_LENGTH` | `500` | Characters per label description |
| `MAX_SYSTEM_PROMPT_LENGTH` | `20000` | Custom template length |
| `MAX_RENDERED_PROMPT_LENGTH` | `100000` | Rendered system prompt length |
| `MAX_CONFIG_ID_LENGTH` | `128` | Config ID length in bodies and paths |
| `MAX_ATTEMPTS` | `10` | Maximum accepted `config.retries` |
| `MAX_OUTPUT_TOKENS` | `16384` | Maximum accepted output token budget |
| `MAX_FEW_SHOT_EXAMPLES` | `20` | Maximum examples per configuration |
| `CONFIG_DB_PATH` | `configs.db` | SQLite path; parent directory must exist and be writable |
| `CACHE_ENABLED` | `true` | Enable in-process extraction result cache |
| `CACHE_TTL_SECONDS` | `600` | Result TTL |
| `CACHE_MAX_SIZE` | `10000` | Maximum cache entries (not a byte-size cap) |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive upstream failures before opening |
| `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S` | `30` | Delay before permitting recovery probes |
| `CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS` | `1` | Concurrent recovery probes |
| `OTEL_ENDPOINT` | Unset | OTLP HTTP trace endpoint, including `/v1/traces` |
| `TOKEN_PRICING_JSON` | Unset | Model-to-price JSON map for estimated costs |

The API additionally caps batches at 100 items, label names at 64 characters,
model IDs at 128 characters, and raw model entities at 2048. Stored configurations
are revalidated against current limits when used. Save a new config to adopt a
changed default model; existing records preserve their resolved settings.

```dotenv
ALLOWED_MODELS=["smollm2-1.7b"]
TOKEN_PRICING_JSON={"example-model":{"input_per_million":0.1,"output_per_million":0.2}}
```

Prices are administrator-supplied estimates per million tokens, not live billing.
Only finite, non-negative numeric prices are accepted. Do not include currency symbols.
