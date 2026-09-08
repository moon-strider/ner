# Public-readiness audit

Audit date: 2026-09-08. Starting commit: `9c8d4d0` (`clean production ner runtime`).
Scope: application and provider code, config storage, offsets, concurrency controls,
HTTP/SDK contracts, packaging, Docker/Compose, CI, dependency advisories, repository
history, examples, and documentation. This is a source and runtime review, not a
formal security certification or a model-quality guarantee.

## Findings and resolution

| Area | Finding | Resolution |
| --- | --- | --- |
| Cache correctness | Different prompt payloads reused one cached response; a hit could bypass template validation | Key includes rendered prompt and provider; validation precedes lookup |
| Cache accounting | Cached responses repeated historical token usage and attempt counts | Explicit `cache_hit`, zero new attempts, no new usage; counters exclude hits |
| Cache memory | Shared mutable values, wall-clock expiry, linear eviction | Defensive copies, monotonic TTL, bounded LRU |
| Offsets | Trie expanded every occurrence, overwrote duplicate surfaces, selected conflicting prefixes, and shifted Unicode positions after lowercasing | One source-based matcher; consumes one non-overlapping occurrence per model item |
| Entity grounding | Dictionary mode kept hallucinated surfaces and duplicate pairs | Verify substrings, restore source casing, deduplicate, return count-only warnings |
| Prompt templates | Literal braces in label descriptions were interpreted as template placeholders | Render only explicitly configured templates; validate malformed formatting |
| Circuit breaker | Failed/cancelled recovery probes could leave it stuck; old in-flight completions could corrupt a newer state | Generation-aware transitions, consecutive failure counting, cancellation-safe probe slots |
| Rate limiter | Tokens were spent before waiting for concurrency admission | Acquire a slot before spending tokens; release slot if refill is cancelled |
| Public errors | Raw provider messages and batch exception strings could expose private data; 404/405 differed from other errors | Shared public classification, allowlisted provider metadata, consistent HTTP and batch errors |
| Input controls | Unbounded repair counts/output budgets; no body-size limit; loose examples and typo acceptance | Runtime limits, chunked-body enforcement, strict fields, validated few-shot entities, safe request IDs |
| Access and cost | No built-in authentication or model allowlist | Optional Bearer key and `ALLOWED_MODELS`; shared-trust limits documented |
| Storage | SQLite `:memory:` broke across per-operation connections; shallow in-memory copies; concurrent PATCH lost updates | Persistent serialized connection, WAL, defensive copies, per-process mutation lock |
| Readiness/lifecycle | Storage initialization was deferred; connection/exporter cleanup incomplete; default app missed environment tracing endpoint | Initialize storage before serving, return 503 on readiness failure, close owned resources, load settings before tracing |
| Provider parsing | Unexpected response types could become internal errors; truncated valid JSON was accepted; malformed usage broke metrics | Structural checks, repair truncated output, filter invalid usage, handle non-object error bodies |
| Metrics | Batch bypassed extraction metrics; cache hits inflated tokens | Record in the shared service layer |
| SDK | Error models disagreed with runtime responses; hard-coded SDK defaults overrode server model; generation deleted existing SDK before success | Regenerate typed errors, omit unresolved defaults, stage generation and add drift check |
| Dependencies | Initial advisory scan returned 29 entries across seven packages (some IDs duplicated) | Update compatible packages and lockfile; follow-up scan clean |
| Docker | Host networking, missing persistent config volume, partial env forwarding, different base distributions | Bridge network, localhost ports, env file, persistent SQLite volume, matching Debian images, non-root smoke gate |
| Repository | MIT declared without license file; generated client lock and unused runtime dependency/duplicate matcher; inaccurate sample offsets | Add license, remove redundant pieces, correct synthetic fixture offsets |
| Documentation | Quick start lacked a working keyless path; benchmarks had no raw evidence; boundaries unclear | Local CPU guide, rewritten README/API/operations, separate explicitly unverified historical benchmarks |

Nine regression scenarios were first executed against the original implementation
and all failed, covering cache isolation, cache validation, Unicode/duplicate/prefix
offsets, grounding, literal braces, and two circuit-breaker recovery failures.
The original 80 tests had passed, so the added tests exercise previously missed behavior.

## Validation evidence

- **149 passing unit/contract/SDK tests**, plus one opt-in live test skipped by default.
- The suite also passes on Python 3.13.14 (Python 3.12.13 used for coverage and live runs).
- Approximately **92% statement coverage** for `src/ner_service` in the local suite.
- Ruff lint, format checks, strict mypy, and deterministic SDK regeneration pass.
- Service and SDK wheel/source distributions build successfully.
- Actual CPU model + Uvicorn + HTTP smoke: six scenarios completed; sync and async
  SDK calls succeeded; SQLite configs survived restarting Uvicorn.
  [Raw CPU report](validation/cpu-smoke.json).
- `pip-audit`: no known vulnerabilities in the updated installed runtime/development
  environment. [Before/after advisory summary](validation/dependencies.json).
- Gitleaks 8.30.1: 32 original Git commits scanned, no detected leaks; current
  source scan also clean. Secret scanners cannot prove that no secret exists.
- Docker Compose 5.5.1 validates the complete observability configuration with
  `.env.example`. GitHub CI also built and ran the Docker image, verified the
  non-root user, received HTTP 200 from readiness, confirmed SQLite database
  creation, and validated Compose configuration. The complete monitoring stack
  was not run.
- [GitHub CI run 32](https://github.com/moon-strider/ner/actions/runs/34276605278)
  passed all three jobs for the audited code in
  [PR #1](https://github.com/moon-strider/ner/pull/1): Python 3.12/3.13 tests,
  lint/format/type checks, SDK drift, service/client distribution builds,
  dependency audit, and the container smoke check. Python 3.12 reported 149
  passing tests and 92.14% statement coverage; the dependency audit was clean.

Two third-party deprecation warnings from the current Starlette/httpx/AnyIO test
stack are visible; they do not affect test outcomes and were not hidden.

## Intentional limits

- The service has shared access, not tenant isolation. Public network deployment
  still needs TLS, gateway admission limits, and storage/access policies.
- Rate limits, cache, circuit breaker, metrics, and PATCH serialization are per process.
- Config listing is unpaginated; cache capacity counts entries rather than bytes.
- There is no complete billing ledger for failed model attempts, no single-flight
  cache coordination, and no whole-request deadline beyond upstream operation timeouts.
- Offsets are substring-based, non-overlapping, and model-order-dependent. They cannot
  resolve semantic ambiguity in repeated text or guarantee correct NER labels.
- Small-model smoke outputs are not accuracy claims. Cloud endpoints were tested
  through mocked protocol responses, without spending cloud API credits.
- Historical CoNLL results were retained with provenance caveats, not rerun or promoted
  as current-version results. Dataset revision pinning/raw prediction export remain
  work for a future reproducible accuracy benchmark.

## Compatibility notes for 1.1.0

Cache hits now report zero attempts and omit usage. Unknown request/config fields
and invalid few-shot examples are rejected. Dictionary entities must occur in the
source. Offset mode no longer expands all occurrences of a returned surface. Batch
error codes now match single-extraction codes. Regenerate or reinstall the Python
client alongside these changes. Existing saved configs are validated against new
runtime bounds when extracted; old configs exceeding limits may require adjustment.
