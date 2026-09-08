# Local CPU inference

The local adapter targets llama.cpp's OpenAI-compatible `/v1/chat/completions`
endpoint and uses JSON Schema structured output. It does not import PyTorch,
Transformers, or model weights into the service process.

## Tested setup

| Component | Value |
| --- | --- |
| Runtime | llama.cpp `b10867`, commit `f3f1a8f27` |
| Binary | Official Ubuntu x64 CPU release |
| Model | `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF` |
| Model revision | `2d4a76a30b4af41ecd395c35725ac11688d4cfe4` |
| File | `smollm2-1.7b-instruct-q4_k_m.gguf` |
| Quantization | Q4_K_M |
| CPU | Virtualized AMD EPYC 9V74 |
| Settings | Four threads, 4096-token context, one slot, zero GPU layers |

Model SHA-256:

```text
decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33
```

Sources: [official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF),
[model revision](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF/tree/2d4a76a30b4af41ecd395c35725ac11688d4cfe4),
[llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b10867),
[server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

## Run

Install a llama.cpp build appropriate for your system. For a previously downloaded
model, this command avoids a network download:

```bash
llama-server -m /path/to/smollm2-1.7b-instruct-q4_k_m.gguf \
  --alias smollm2-1.7b --host 127.0.0.1 --port 8080 \
  -c 4096 -t 4 -ngl 0 -np 1
```

Use `.env.example` for the NER service. The alias must match `NER_MODEL`. Set
`LLAMA_CPP_API_KEY` if the model server requires authentication.

```bash
uv run --frozen uvicorn ner_service.main:app --host 127.0.0.1 --port 8000
# Another terminal:
uv run --frozen python scripts/smoke.py
```

The smoke script uses synthetic text and creates then deletes its own saved
configuration. It exercises extraction, cache metadata, CRUD, few-shot prompts,
payload changes, mixed batch results, validation, and metrics. Against a cloud
provider it may incur charges.

For the opt-in pytest entry point:

```bash
NER_TEST_BASE_URL=http://127.0.0.1:8000 uv run --frozen --extra dev pytest -m integration
```

Normal unit tests never require model downloads, an API key, or network access.

## Results and limits

The audit ran an actual llama.cpp process and Uvicorn process, then called the API
through HTTP and both generated SDK variants. All six smoke scenarios completed
in approximately 6.1 seconds on this host. The SQLite config also survived a
Uvicorn restart. [Raw validation output](validation/cpu-smoke.json) includes timings,
usage, model identity, entities, and SDK responses.

This is a smoke result, **not** an NER accuracy benchmark. The small model produced
schema-valid but incomplete or incorrectly labelled entities in exploratory runs.
Do not use its outputs as gold annotations. Context includes system instructions,
labels, examples, input, and output tokens; a 4096-token model server cannot process
every input allowed by the service's general 32000-character limit. Keep local
inputs short or increase the model context and memory budget deliberately.

Model files are excluded from Git and container builds. The model is distributed
under its own Apache-2.0 license; the NER service's MIT license does not replace it.
