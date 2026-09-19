# Security

See [deployment boundaries](docs/operations.md) before exposing the service.

Set `NER_API_KEY` for shared access, restrict models with `ALLOWED_MODELS`, and
place remote deployments behind TLS, connection/rate limits, and appropriate
network controls. Configuration storage and cache are shared across callers.
This project does not implement per-user authorization or tenant isolation.

Provider URLs and credentials are trusted operator settings. API callers cannot
supply a provider URL. Model responses are untrusted structured data: the service
validates their shape and labels, verifies entity surfaces, and never executes
model output. Structured output cannot prevent semantic prompt-injection errors;
do not make authorization decisions solely from extracted entities.

The optional `span_pipeline` mode adds TypeSafe as a third party in the data-processing
path. With it enabled, input text and candidate windows, the configured label names and
descriptions, and the judgment criteria and instructions are sent to `TYPESAFE_BASE_URL`
using `TYPESAFE_API_KEY`. Review that processor's terms and retention before enabling the
mode on sensitive text; with the mode disabled no TypeSafe request is made.

Operators and integrations must not log input text, extracted entity surfaces, or label
values; log counts, codes, durations, and request IDs instead. Provider failures surface
as a public error code and message: the upstream response body is not returned to API
clients and must not be logged.

Never include credentials or personal text in public bug reports. If the repository's
private vulnerability reporting is enabled, use its Security tab. Otherwise open
an issue requesting a private contact without including exploit details or secrets.
Do not assume an unpublished security contact or response-time commitment.
