import json

import pytest
from pydantic import ValidationError

from ner_service.config import Settings
from ner_service.config_store import PromptTemplateError, prepare_config, render_system_prompt
from ner_service.providers.base import ProviderError, ProviderUpstreamError
from ner_service.providers.openai_compatible import OpenAICompatibleProvider
from ner_service.providers.registry import get_provider
from ner_service.schemas import EntityLabel, NERConfig


def config(**kwargs):
    return NERConfig(labels=[EntityLabel(name="X", description="Names")], **kwargs)


@pytest.mark.parametrize(
    "completion",
    [
        [],
        "text",
        None,
        {"choices": "oops"},
        {"choices": [None]},
        {"choices": [{}]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": []}}]},
        {"choices": [{"message": {"content": "{}", "refusal": "private"}}]},
    ],
)
async def test_malformed_upstream_response_is_provider_error(httpx_mock, completion):
    provider = OpenAICompatibleProvider("test", "https://test.example/v1", "m")
    httpx_mock.add_response(json=completion)
    prepared = prepare_config(config(retries=1))
    with pytest.raises(ProviderError):
        await provider.extract("Hello", prepared=prepared, system_prompt="Extract")
    await provider.aclose()


@pytest.mark.parametrize("body", [[], None, "bad", {"error": None}])
async def test_non_object_upstream_error_body_is_handled(httpx_mock, body):
    provider = OpenAICompatibleProvider("test", "https://test.example/v1", "m")
    httpx_mock.add_response(status_code=503, json=body)
    with pytest.raises(ProviderUpstreamError):
        await provider.extract("Hello", prepared=prepare_config(config()), system_prompt="Extract")
    await provider.aclose()


@pytest.mark.parametrize(
    "usage",
    [[], "invalid", {"prompt_tokens": -5}, {"completion_tokens": True}, {"prompt_tokens": "42"}],
)
async def test_invalid_usage_is_not_counted(httpx_mock, usage):
    provider = OpenAICompatibleProvider("test", "https://test.example/v1", "m")
    httpx_mock.add_response(
        json={"choices": [{"message": {"content": '{"entities":[]}'}}], "usage": usage}
    )
    result = await provider.extract(
        "Hi", prepared=prepare_config(config()), system_prompt="Extract"
    )
    assert result.usage is None
    await provider.aclose()


async def test_truncated_valid_json_is_repaired(httpx_mock):
    provider = OpenAICompatibleProvider("test", "https://test.example/v1", "m")
    for reason in ["length", "stop"]:
        httpx_mock.add_response(
            json={
                "choices": [{"message": {"content": '{"entities":[]}'}, "finish_reason": reason}],
                "usage": {"total_tokens": 3},
            }
        )
    result = await provider.extract(
        "Hi", prepared=prepare_config(config()), system_prompt="Extract"
    )
    assert result.attempts == 2 and result.usage["total_tokens"] == 6
    await provider.aclose()


async def test_llama_cpp_uses_max_tokens(httpx_mock):
    provider = get_provider(Settings(ner_provider="llama_cpp"))
    httpx_mock.add_response(json={"choices": [{"message": {"content": '{"entities":[]}'}}]})
    await provider.extract(
        "Hi", prepared=prepare_config(config(max_tokens=42)), system_prompt="Extract"
    )
    body = json.loads(httpx_mock.get_request().content)
    assert body["max_tokens"] == 42 and "max_completion_tokens" not in body
    await provider.aclose()


@pytest.mark.parametrize(
    "name, key, url",
    [
        ("openai", "openai_api_key", "https://api.openai.com/v1"),
        ("cerebras", "cerebras_api_key", "https://api.cerebras.ai/v1"),
        ("openrouter", "openrouter_api_key", "https://openrouter.ai/api/v1"),
    ],
)
async def test_provider_registry_and_required_credentials(name, key, url, monkeypatch):
    monkeypatch.delenv(key.upper(), raising=False)
    with pytest.raises(RuntimeError, match="required"):
        get_provider(Settings(_env_file=None, ner_provider=name))
    provider = get_provider(Settings(ner_provider=name, **{key: "test"}))
    assert provider.name == name and provider._base_url == url
    await provider.aclose()


async def test_vllm_requires_explicit_endpoint():
    with pytest.raises(RuntimeError):
        get_provider(Settings(_env_file=None, ner_provider="vllm"))
    provider = get_provider(Settings(ner_provider="vllm", vllm_base_url="http://localhost:9000/v1"))
    assert provider.name == "vllm"
    await provider.aclose()


@pytest.mark.parametrize("template", ["{", "}", "{payload.x!r}", "{payload.x:10}", "{}"])
def test_malformed_templates_raise_prompt_error(template):
    with pytest.raises(PromptTemplateError):
        render_system_prompt(prepare_config(config(system_prompt=template)), {"x": "value"})


@pytest.mark.parametrize(
    "entity",
    [
        {"text": "Hi", "label": "OTHER"},
        {"text": "Absent", "label": "X"},
        {"text": "Hi", "label": "X", "extra": 1},
    ],
)
def test_invalid_few_shot_entities_are_rejected(entity):
    with pytest.raises(ValidationError):
        config(few_shot_examples=[{"text": "Hi", "entities": [entity]}])


def test_label_cannot_end_with_newline():
    with pytest.raises(ValidationError):
        EntityLabel(name="PERSON\n", description="People")
