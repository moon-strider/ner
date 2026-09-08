# Benchmarks and profiling

## Methodology

`benchmark_conll.py` reconstructs CoNLL-2003 text by joining tokens with spaces.
Offset scoring uses exact `(label, start, end)` matches on that reconstruction.
Dictionary scoring uses unique `(label, text)` pairs and collapses repeated mentions.
These modes measure different tasks; do not compare their counts as identical.
Provider failures contribute false negatives and are counted separately.

```bash
uv sync --frozen --extra benchmark
# Set your provider and model in .env first. This may incur charges.
uv run --frozen --extra benchmark python scripts/benchmark_conll.py --limit 100 --concurrency 4
uv run --frozen --extra benchmark python scripts/benchmark_conll.py --limit 100 --concurrency 4 --require-offsets
```

The script downloads the `eriktks/conll2003` test split from Hugging Face's parquet
conversion ref and caches the reconstructed data in `data/benchmarks/`. The ref is
not immutable: retain a local cache and its SHA-256 for reproducible comparisons.
The cache and dataset are not redistributed with this repository. Check the
dataset's license and access conditions separately.

```bash
uv run --frozen python scripts/profile.py --texts-count 8 --concurrency 2 --text-lengths 64,256,1024 --output reports/profile.json
```

The profiler reports latency percentiles, request throughput, errors, and usage.
It measures the Python service pipeline directly, not HTTP gateway overhead.
The HTTP smoke check is described in [local inference](local-inference.md).

## Historical measurements (unverified)

The following table is preserved from commit `9c8d4d0`. Raw predictions, dataset
hashes, and complete runtime metadata were not checked in. It predates the audit's
offset and grounding fixes and must not be advertised as the current version's
validated performance. It was **not rerun** with cloud credentials during this audit.
Historical cost estimates are not current provider prices.

Full CoNLL-2003 `test` baseline results (3453 examples, concurrency 40, max_tokens 1024, retries 3):

| Model | Mode | Reasoning | micro-P | micro-R | micro-F1 | TP | FP | FN | Errors | Total s | Avg s | Min s | Max s | Examples/s | Total tokens |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.1-8b` | dictionary | N/A | 0.531 | 0.737 | 0.617 | 4115 | 3637 | 1469 | 2 | 68.163 | 0.772 | 0.350 | 5.083 | 50.658 | 1436014 |
| `llama3.1-8b` | offsets | N/A | 0.509 | 0.726 | 0.598 | 4098 | 3950 | 1550 | 1 | 114.363 | 1.314 | 0.342 | 63.124 | 30.193 | 1445445 |
| `gpt-oss-120b` | dictionary | low | 0.778 | 0.776 | 0.777 | 4335 | 1240 | 1249 | 0 | 204.446 | 2.343 | 0.298 | 66.743 | 16.890 | 1758947 |
| `gpt-oss-120b` | offsets | low | 0.771 | 0.774 | 0.773 | 4374 | 1297 | 1274 | 0 | 207.912 | 2.214 | 0.304 | 62.849 | 16.608 | 1815023 |

Cost to run:

- `llama3.1-8b`: ±0.3$ per run
- `gpt-oss-120b`: ±0.69$ per run
